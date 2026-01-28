"""Multi-task learning model for virtual staining + amyloid segmentation.

This model uses a shared encoder backbone with two decoder heads:
- Shared Encoder: Extracts features F from H&E images
- Head A (main): Virtual staining decoder (flow matching) - takes F + time t
- Head B (aux): Segmentation decoder - takes F, predicts mask M

The model is trained jointly with: L = L_FM + α * L_seg

Key insight: The segmentation task forces the shared encoder to learn amyloid-specific
features from H&E, which helps the flow matching decoder generate better IHC.
"""

import pdb
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

from src.models.components.shared_encoder import SharedEncoder, TimeEmbedding
from src.models.components.task_decoders import FlowMatchingDecoder, SegmentationDecoder


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss.
        
        Args:
            pred: Predictions of shape (B, 1, H, W) - logits
            target: Ground truth of shape (B, 1, H, W) - binary masks [0, 1]
            
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class MultiTaskFlowMatchingLitModule(LightningModule):
    """Multi-task learning with shared backbone for virtual staining and segmentation.
    
    Architecture:
        H&E → Shared Encoder → Features F
                                   ├→ Flow Decoder (Head A) → IHC
                                   └→ Seg Decoder (Head B) → Mask
    """

    def __init__(
        self,
        encoder: SharedEncoder,
        flow_decoder: FlowMatchingDecoder,
        seg_decoder: SegmentationDecoder,
        flow_matcher: ConditionalFlowMatcher,
        solver: Optional[NeuralODE] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        log_images: bool = True,
        seg_loss_weight: float = 1.0,  # α in L = L_FM + α * L_seg
        dice_weight: float = 0.5,  # Weight for Dice vs BCE in segmentation loss
        n_images_log: int = 5,
        time_emb_dim: int = 256,
    ) -> None:
        """Initialize the multi-task model with shared backbone.

        Args:
            encoder: Shared encoder backbone that extracts features from H&E
            flow_decoder: Decoder for flow matching (Head A: virtual staining)
            seg_decoder: Decoder for segmentation (Head B: amyloid mask)
            flow_matcher: Conditional flow matcher
            solver: ODE solver for inference
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            compile: Whether to compile the model with torch.compile
            log_images: Whether to log images to wandb
            seg_loss_weight: Weight α for segmentation loss (L = L_FM + α * L_seg)
            dice_weight: Weight for Dice loss in segmentation (1-dice_weight for BCE)
            n_images_log: Number of images to log per epoch
            time_emb_dim: Dimension of time embeddings
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        # Shared backbone and task-specific heads
        self.encoder = encoder
        self.flow_decoder = flow_decoder
        self.seg_decoder = seg_decoder
        
        # Time embedding for flow matching
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Flow matching components
        self.flow_matcher = flow_matcher
        self.solver = solver
        
        # Training components
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Loss components
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.seg_loss_weight = seg_loss_weight
        self.dice_weight = dice_weight
        
        # Logging
        self.log_images = log_images
        self.n_images_log = n_images_log
        
        if compile:
            self.encoder = torch.compile(self.encoder)
            self.flow_decoder = torch.compile(self.flow_decoder)
            self.seg_decoder = torch.compile(self.seg_decoder)

    def forward_flow(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for flow matching (Head A).
        
        Args:
            t: Time tensor (B,) or (B, 1)
            x: Interpolated state xt (B, 3, H, W)
            
        Returns:
            Predicted velocity field (B, 3, H, W)
        """
        # Encode the interpolated state to get features
        bottleneck, skip_connections = self.encoder(x)
        
        # Get time embeddings
        t_emb = self.time_embedding(t)
        
        # Decode with time conditioning
        velocity = self.flow_decoder(bottleneck, skip_connections, t_emb)
        
        return velocity

    def forward_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation (Head B).
        
        Args:
            x: Input H&E image tensor (B, 3, H, W)
            
        Returns:
            Predicted mask logits (B, 1, H, W)
        """
        # Encode H&E image to get features
        bottleneck, skip_connections = self.encoder(x)
        
        # Decode to get mask
        mask_logits = self.seg_decoder(bottleneck, skip_connections)
        
        return mask_logits

    def compute_segmentation_loss(
        self, pred_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined segmentation loss (Dice + BCE).
        
        Args:
            pred_mask: Predicted mask logits (B, 1, H, W)
            target_mask: Ground truth mask (B, 1, H, W)
            
        Returns:
            Combined loss and dictionary of individual loss components
        """
        # Convert target_mask to float if it's not already
        target_mask = target_mask.float()
    
        # Compute losses
        dice_loss = self.dice_loss(pred_mask, target_mask)
        bce_loss = self.bce_loss(pred_mask, target_mask)
        
        # Combined segmentation loss
        seg_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss
        
        loss_dict = {
            "dice": dice_loss,
            "bce": bce_loss,
            "seg_total": seg_loss,
        }
        
        return seg_loss, loss_dict

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform a single model step for training or validation.

        Args:
            batch: (source_img, target_img, target_seg_mask)
                - source_img: H&E images (B, 3, H, W)
                - target_img: IHC images (B, 3, H, W)
                - target_seg_mask: Amyloid masks (B, 1, H, W), ROI=1
                
        Returns:
            Total loss and dictionary of loss components
        """
        source_img, target_img, gt_mask = batch
        x0, x1 = source_img, target_img

        # ============ Task A: Flow Matching (Virtual Staining) ============
        # Sample time, interpolated state, and conditional flow
        # Note: We don't condition on mask here - encoder learns from H&E directly
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        
        # Predict velocity field from interpolated state xt
        # The encoder will learn features useful for this task
        vt = self.forward_flow(t, xt)
        
        # Flow matching MSE loss
        flow_loss = torch.mean((vt - ut) ** 2)

        # ============ Task B: Segmentation (Predict Amyloid Mask) ============
        # Predict mask from source H&E image
        # This forces the encoder to learn amyloid-specific features
        pred_mask_logits = self.forward_segmentation(source_img)
        
        # Compute segmentation loss
        seg_loss, seg_loss_dict = self.compute_segmentation_loss(
            pred_mask_logits, gt_mask
        )

        # ============ Combined Loss ============
        # L = L_FM + α * L_seg
        # Both losses backpropagate through the shared encoder
        total_loss = flow_loss + self.seg_loss_weight * seg_loss

        # Prepare loss dictionary for logging
        loss_dict = {
            "total": total_loss,
            "flow": flow_loss,
            "seg": seg_loss,
            "seg_dice": seg_loss_dict["dice"],
            "seg_bce": seg_loss_dict["bce"],
        }

        return total_loss, loss_dict

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.
        
        Args:
            batch: A batch of data (source_img, target_img, target_mask)
            batch_idx: The index of the batch
            
        Returns:
            The total loss value
        """
        total_loss, loss_dict = self.model_step(batch)
        
        # Log all losses
        self.log(
            "train/loss", loss_dict["total"],
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "train/flow_loss", loss_dict["flow"],
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            "train/seg_loss", loss_dict["seg"],
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "train/seg_dice", loss_dict["seg_dice"],
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            "train/seg_bce", loss_dict["seg_bce"],
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        
        return total_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step.
        
        Args:
            batch: A batch of data (source_img, target_img, target_mask)
            batch_idx: The index of the batch
        """
        total_loss, loss_dict = self.model_step(batch)
        
        # Log all losses
        self.log(
            "val/loss", loss_dict["total"],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "val/flow_loss", loss_dict["flow"],
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            "val/seg_loss", loss_dict["seg"],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "val/seg_dice", loss_dict["seg_dice"],
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            "val/seg_bce", loss_dict["seg_bce"],
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )
        
        # Compute segmentation metrics (IoU, Dice coefficient)
        source_img, target_img, mask = batch
        pred_mask_logits = self.forward_segmentation(source_img)
        pred_mask = torch.sigmoid(pred_mask_logits) > 0.5  # Threshold at 0.5
        
        # Dice coefficient
        intersection = (pred_mask * mask).sum()
        union = pred_mask.sum() + mask.sum()
        dice_coef = (2.0 * intersection + 1e-7) / (union + 1e-7)
        
        # IoU
        intersection = (pred_mask * mask).sum()
        union = (pred_mask | mask).float().sum()
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        self.log("val/dice_coef", dice_coef, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step.
        
        Args:
            batch: A batch of data (source_img, target_img, target_mask)
            batch_idx: The index of the batch
        """
        total_loss, loss_dict = self.model_step(batch)
        
        # Log all losses
        self.log(
            "test/loss", loss_dict["total"],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "test/flow_loss", loss_dict["flow"],
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log(
            "test/seg_loss", loss_dict["seg"],
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        
        # Compute segmentation metrics
        source_img, target_img, mask = batch
        pred_mask_logits = self.forward_segmentation(source_img)
        pred_mask = torch.sigmoid(pred_mask_logits) > 0.5
        
        # Dice coefficient
        intersection = (pred_mask * mask).sum()
        union = pred_mask.sum() + mask.sum()
        dice_coef = (2.0 * intersection + 1e-7) / (union + 1e-7)
        
        # IoU
        intersection = (pred_mask * mask).sum()
        union = (pred_mask | mask).float().sum()
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        self.log("test/dice_coef", dice_coef, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            A dictionary with optimizer(s) and scheduler(s)
        """
        # Optimize all components: encoder + both decoders
        params = (
            list(self.encoder.parameters())
            + list(self.flow_decoder.parameters())
            + list(self.seg_decoder.parameters())
        )
        optimizer = self.optimizer(params=params)
        
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}

    @torch.no_grad()
    def generate(
        self,
        source_img: torch.Tensor,
        num_steps: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate target image and predicted mask from source image.
        
        Args:
            source_img: Source H&E image tensor of shape (B, C, H, W) or (C, H, W)
            num_steps: Number of ODE integration steps
            
        Returns:
            Tuple of (generated IHC image, predicted mask probabilities)
        """
        if self.solver is None:
            raise ValueError("Solver is not initialized. Cannot perform inference.")
        
        self.eval()
        
        # Handle single image case
        if source_img.dim() == 3:
            source_img = source_img.unsqueeze(0)
        
        device = source_img.device
        batch_size = source_img.shape[0]
        
        # ============ Generate Segmentation Mask ============
        pred_mask_logits = self.forward_segmentation(source_img)
        pred_mask = torch.sigmoid(pred_mask_logits)
        
        # ============ Generate Virtual Staining ============
        # Create a wrapper for ODE integration that uses the encoder+decoder
        class FlowWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, t, x, **kwargs):
                # Ensure t is the right shape
                if t.dim() == 0:
                    t = t.unsqueeze(0).expand(x.shape[0])
                elif t.dim() == 1 and t.shape[0] == 1:
                    t = t.expand(x.shape[0])
                
                # Forward through encoder and flow decoder
                return self.model.forward_flow(t, x)
        
        wrapped_net = FlowWrapper(self)

        node = NeuralODE(
            wrapped_net,
            solver=self.solver.solver if hasattr(self.solver, 'solver') else "dopri5",
            sensitivity=self.solver.sensitivity if hasattr(self.solver, 'sensitivity') else "adjoint",
            atol=self.solver.atol if hasattr(self.solver, 'atol') else 1e-4,
            rtol=self.solver.rtol if hasattr(self.solver, 'rtol') else 1e-4,
        )
        
        # Integrate from t=0 to t=1
        t_span = torch.linspace(0, 1, num_steps, device=device)
        traj = node.trajectory(source_img, t_span=t_span)
        
        # Return final image and predicted mask
        generated_ihc = traj[-1]
        
        return generated_ihc, pred_mask

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log images if requested
        if self.log_images and self.current_epoch % 1 == 0:
            self._log_images("train")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Log images if requested
        if self.log_images and self.current_epoch % 1 == 0:
            self._log_images("val")

    def _log_images(self, split: str = "train") -> None:
        """Helper method to log images to wandb.
        
        Args:
            split: Either 'train' or 'val' to specify which split we're logging
        """
        # Check if wandb logger is available
        wandb_logger = None
        for logger_ in self.loggers:
            if hasattr(logger_, 'experiment'):
                if 'wandb' in str(type(logger_.experiment)).lower():
                    wandb_logger = logger_
                    break
        
        if wandb_logger is None:
            return
        
        # Get a batch from the appropriate dataloader
        if split == "train":
            dataloader = self.trainer.train_dataloader
        else:
            dataloader = self.trainer.val_dataloaders
        
        try:
            batch = next(iter(dataloader))
            source_imgs, target_imgs, masks = batch
            
            # Move to device
            source_imgs = source_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)
            masks = masks.to(self.device)
            
            # Select images to log
            num_images = min(self.n_images_log, len(source_imgs))
            indices = torch.randperm(len(source_imgs))[:num_images]
            
            wandb_images = []
            for idx in indices:
                idx_int = idx.item()
                source_img = source_imgs[idx_int].unsqueeze(0)
                target_img = target_imgs[idx_int]
                gt_mask = masks[idx_int]
                
                # Generate predictions
                gen_img, pred_mask = self.generate(source_img, num_steps=2)
                
                # Convert tensors to numpy for logging
                source_np = source_img[0].cpu().permute(1, 2, 0).numpy()
                target_np = target_img.cpu().permute(1, 2, 0).numpy()
                gen_np = gen_img[0].cpu().permute(1, 2, 0).numpy()
                gt_mask_np = gt_mask[0].cpu().numpy()
                pred_mask_np = pred_mask[0, 0].cpu().numpy()
                
                # Create wandb image
                wandb_images.append(
                    wandb.Image(
                        source_np,
                        caption=f"{split} - Epoch {self.current_epoch} - Source H&E",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        target_np,
                        caption=f"{split} - Epoch {self.current_epoch} - Target IHC",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        gen_np,
                        caption=f"{split} - Epoch {self.current_epoch} - Generated IHC",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        gt_mask_np,
                        caption=f"{split} - Epoch {self.current_epoch} - GT Mask",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        pred_mask_np,
                        caption=f"{split} - Epoch {self.current_epoch} - Pred Mask",
                    )
                )
            
            # Log to wandb
            wandb_logger.experiment.log({f"{split}/images": wandb_images})
            
        except Exception as e:
            print(f"Failed to log images: {e}")
