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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

from src.models.components.shared_encoder import SharedEncoder, TimeEmbedding
from src.models.components.task_decoders import FlowMatchingDecoder, SegmentationDecoder



class MulticlassDiceLoss(nn.Module):
    """Dice loss for multiclass segmentation."""

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multiclass Dice loss.
        
        Args:
            pred: Predictions of shape (B, C, H, W) - logits for C classes
            target: Ground truth of shape (B, H, W) - class indices [0, C-1]
            
        Returns:
            Dice loss value (averaged over classes)
        """
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)  # (B, C, H, W)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Create mask for valid pixels (ignore index)
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index).float()  # (B, H, W)
            valid_mask = valid_mask.unsqueeze(1)  # (B, 1, H, W)
        else:
            valid_mask = torch.ones_like(target).unsqueeze(1).float()
        
        # Compute Dice for each class
        dice_scores = []
        for c in range(self.num_classes):
            pred_c = pred[:, c:c+1, :, :]  # (B, 1, H, W)
            target_c = target_one_hot[:, c:c+1, :, :]  # (B, 1, H, W)
            
            # Apply valid mask
            pred_c = pred_c * valid_mask
            target_c = target_c * valid_mask
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average dice across classes
        mean_dice = torch.stack(dice_scores).mean()
        
        return 1 - mean_dice


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
        num_classes: int = 2,  # Number of segmentation classes
        solver: Optional[NeuralODE] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        log_images: bool = True,
        seg_loss_weight: float = 1.0,  # α in L = L_FM + α * L_seg
        dice_weight: float = 0.5,  # Weight for Dice vs CE in segmentation loss
        n_images_log: int = 5,
        time_emb_dim: int = 256,
        ignore_index: int = -100,  # Ignore index for segmentation loss
    ) -> None:
        """Initialize the multi-task model with shared backbone.

        Args:
            encoder: Shared encoder backbone that extracts features from H&E
            flow_decoder: Decoder for flow matching (Head A: virtual staining)
            seg_decoder: Decoder for segmentation (Head B: multiclass segmentation)
            flow_matcher: Conditional flow matcher
            num_classes: Number of segmentation classes
            solver: ODE solver for inference
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            compile: Whether to compile the model with torch.compile
            log_images: Whether to log images to wandb
            seg_loss_weight: Weight α for segmentation loss (L = L_FM + α * L_seg)
            dice_weight: Weight for Dice loss in segmentation (1-dice_weight for CE)
            n_images_log: Number of images to log per epoch
            time_emb_dim: Dimension of time embeddings
            ignore_index: Class index to ignore in segmentation loss
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
        
        # Segmentation parameters
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Loss components
        self.dice_loss = MulticlassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
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
            Predicted mask logits (B, num_classes, H, W) for multiclass segmentation
        """
        # Encode H&E image to get features
        bottleneck, skip_connections = self.encoder(x)
        
        # Decode to get mask
        mask_logits = self.seg_decoder(bottleneck, skip_connections)
        
        return mask_logits

    def compute_segmentation_loss(
        self, pred_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined segmentation loss (Dice + CrossEntropy).
        
        Args:
            pred_mask: Predicted mask logits (B, num_classes, H, W)
            target_mask: Ground truth mask (B, H, W) - class indices
            
        Returns:
            Combined loss and dictionary of individual loss components
        """
        # Remove channel dimension if present (B, 1, H, W) -> (B, H, W)
        if target_mask.dim() == 4 and target_mask.shape[1] == 1:
            target_mask = target_mask.squeeze(1)
        
        # Ensure target is long tensor for CrossEntropyLoss
        target_mask = target_mask.long()
    
        # Compute losses
        dice_loss = self.dice_loss(pred_mask, target_mask)
        ce_loss = self.ce_loss(pred_mask, target_mask)
        
        # Combined segmentation loss
        seg_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss
        
        loss_dict = {
            "dice": dice_loss,
            "ce": ce_loss,
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
                - target_seg_mask: Multiclass masks (B, H, W) with class indices [0, num_classes-1]
                
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
            "seg_ce": seg_loss_dict["ce"],
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
            "train/seg_ce", loss_dict["seg_ce"],
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
            "val/seg_ce", loss_dict["seg_ce"],
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )
        
        # Compute segmentation metrics (IoU, Dice coefficient) for multiclass
        source_img, target_img, mask = batch
        pred_mask_logits = self.forward_segmentation(source_img)
        pred_mask = torch.argmax(pred_mask_logits, dim=1)  # (B, H, W) - class predictions
        
        # Remove channel dimension from target if present
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        mask = mask.long()
        
        # # Compute metrics per class and average (excluding ignore_index)
        # dice_per_class = []
        # iou_per_class = []
        
        # for c in range(self.num_classes):
        #     # Binary mask for class c
        #     pred_c = (pred_mask == c)
        #     target_c = (mask == c)
            
        #     # Skip if ignore_index
        #     if c == self.ignore_index:
        #         continue
            
        #     # Dice coefficient for class c
        #     intersection = (pred_c & target_c).float().sum()
        #     union = pred_c.float().sum() + target_c.float().sum()
        #     dice_c = (2.0 * intersection + 1e-7) / (union + 1e-7)
        #     dice_per_class.append(dice_c)
            
        #     # IoU for class c
        #     intersection = (pred_c & target_c).float().sum()
        #     union = (pred_c | target_c).float().sum()
        #     iou_c = (intersection + 1e-7) / (union + 1e-7)
        #     iou_per_class.append(iou_c)
        
        # # Average across classes
        # mean_dice = torch.stack(dice_per_class).mean() if dice_per_class else torch.tensor(0.0)
        # mean_iou = torch.stack(iou_per_class).mean() if iou_per_class else torch.tensor(0.0)
        
        # self.log("val/dice_coef", mean_dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("val/iou", mean_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
        
        # Compute segmentation metrics for multiclass
        source_img, target_img, mask = batch
        pred_mask_logits = self.forward_segmentation(source_img)
        pred_mask = torch.argmax(pred_mask_logits, dim=1)  # (B, H, W)
        
        # Remove channel dimension from target if present
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        mask = mask.long()
        
        # Compute metrics per class and average
        dice_per_class = []
        iou_per_class = []
        
        for c in range(self.num_classes):
            pred_c = (pred_mask == c)
            target_c = (mask == c)
            
            if c == self.ignore_index:
                continue
            
            # Dice coefficient for class c
            intersection = (pred_c & target_c).float().sum()
            union = pred_c.float().sum() + target_c.float().sum()
            dice_c = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_per_class.append(dice_c)
            
            # IoU for class c
            intersection = (pred_c & target_c).float().sum()
            union = (pred_c | target_c).float().sum()
            iou_c = (intersection + 1e-7) / (union + 1e-7)
            iou_per_class.append(iou_c)
        
        mean_dice = torch.stack(dice_per_class).mean() if dice_per_class else torch.tensor(0.0)
        mean_iou = torch.stack(iou_per_class).mean() if iou_per_class else torch.tensor(0.0)
        
        self.log("test/dice_coef", mean_dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/iou", mean_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
        # For multiclass: get probabilities and class predictions
        pred_mask_probs = F.softmax(pred_mask_logits, dim=1)  # (B, num_classes, H, W)
        pred_mask = torch.argmax(pred_mask_probs, dim=1, keepdim=True)  # (B, 1, H, W)
        
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
                
                # Denormalize images from [-1, 1] to [0, 1] for visualization
                def denormalize(img):
                    return (img * 0.5 + 0.5).clamp(0, 1)
                
                # Convert tensors to numpy for logging
                source_np = denormalize(source_img[0]).cpu().permute(1, 2, 0).numpy()
                target_np = denormalize(target_img).cpu().permute(1, 2, 0).numpy()
                gen_np = denormalize(gen_img[0]).cpu().permute(1, 2, 0).numpy()
                
                # Handle multiclass masks
                # After indexing masks[idx_int], gt_mask has dim==3 if masks are (B, 1, H, W) or dim==2 if (B, H, W)
                if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
                    gt_mask_np = gt_mask[0].cpu().numpy()
                elif gt_mask.dim() == 2:
                    gt_mask_np = gt_mask.cpu().numpy()
                else:
                    # Unexpected dimension - squeeze all leading dimensions
                    gt_mask_np = gt_mask.squeeze().cpu().numpy()
                pred_mask_np = pred_mask[0, 0].cpu().numpy()
                
                # Create color map for multiclass visualization
                # Define distinct colors for each class (RGB values in [0, 1])
                color_map = np.array([
                    [0.0, 0.0, 0.0],      # Class 0: Black
                    [1.0, 0.0, 0.0],      # Class 1: Red
                    [0.0, 1.0, 0.0],      # Class 2: Green
                    [0.0, 0.0, 1.0],      # Class 3: Blue
                    [1.0, 1.0, 0.0],      # Class 4: Yellow
                    [1.0, 0.0, 1.0],      # Class 5: Magenta (if needed)
                    [0.0, 1.0, 1.0],      # Class 6: Cyan (if needed)
                ])
                
                # Convert masks to color images (H, W) -> (H, W, 3)
                gt_mask_colored = color_map[gt_mask_np.astype(np.int32)]
                pred_mask_colored = color_map[pred_mask_np.astype(np.int32)]
                
                # Create wandb image
                wandb_images.append(
                    wandb.Image(
                        source_np,
                        caption=f"{split} - Epoch {self.current_epoch} - Source",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        target_np,
                        caption=f"{split} - Epoch {self.current_epoch} - Target",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        gen_np,
                        caption=f"{split} - Epoch {self.current_epoch} - Generated Target",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        gt_mask_colored,
                        caption=f"{split} - Epoch {self.current_epoch} - GT Mask",
                    )
                )
                wandb_images.append(
                    wandb.Image(
                        pred_mask_colored,
                        caption=f"{split} - Epoch {self.current_epoch} - Pred Mask",
                    )
                )
            
            # Log to wandb
            wandb_logger.experiment.log({f"{split}/images": wandb_images})
            
        except Exception as e:
            print(f"Failed to log images: {e}")
