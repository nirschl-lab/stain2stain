from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

class ConditionalFlowMatchingLitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        flow_matcher: ConditionalFlowMatcher,
        solver: Optional[NeuralODE] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        log_images: bool = True,
        aux_loss_weight: float = 0.1,
    ) -> None:
        """Initialize a `ConditionalFlowMatchingLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param aux_loss_weight: Weight for auxiliary amyloid fraction loss.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.flow_matcher = flow_matcher
        self.solver = solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_images = log_images
        self.aux_loss_weight = aux_loss_weight
        
        # Regression head for amyloid fraction prediction
        # Assumes velocity field output has 3 channels (RGB)
        C = 3
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global avg pooling over H, W
            nn.Flatten(),  # shape: (B, C)
            nn.Linear(C, 1),  # output: (B, 1)
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        if compile:
            self.net = torch.compile(self.net)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        :param t: Time tensor
        :param x: Input tensor
        :return: Tuple of (predicted velocity field, amyloid fraction prediction)
        """
        vt = self.net(t, x)
        frac_pred = self.regression_head(vt)
        return vt, frac_pred.squeeze(-1)  # vt: (B, C, H, W), frac_pred: (B,)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single model step for training or validation.
        
        :param batch: A batch of data (source_img, target_img, target_seg_mask)
        :return: The loss value
        """
        source_img, target_img, mask = batch  # mask: DAB+ segmentation (B, 1, H, W)

        x0 = source_img
        x1 = target_img

        # Sample interpolated point and conditional velocity from flow matcher
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)

        # Predict vector field and amyloid fraction
        vt, frac_pred = self.forward(t, xt)  # Returns: velocity field + amyloid fraction

        # --- Flow matching loss ---
        # Compute squared error
        squared_error = (vt - ut) ** 2  # (B, C, H, W)

        # Broadcast mask to match channels
        mask_expanded = mask.expand_as(squared_error)

        # Apply segmentation weighting to the squared error
        loss_masked = torch.mean(mask_expanded * squared_error)  # Segmentation-weighted loss
        loss_unmasked = torch.mean(squared_error)       # Standard flow matching loss

        # Combine weighted and unweighted losses
        lambda_weight = 0.5
        flow_matching_loss = lambda_weight * loss_masked + (1 - lambda_weight) * loss_unmasked

        # --- Amyloid fraction auxiliary loss ---
        with torch.no_grad():
            B, _, H, W = mask.shape
            frac_true = mask.view(B, -1).float().mean(dim=1)  # Shape: (B,)

        aux_loss = F.mse_loss(frac_pred, frac_true)

        # --- Total loss ---
        total_loss = flow_matching_loss + self.aux_loss_weight * aux_loss

        return total_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :param batch_idx: The index of the batch
        :return: The loss value
        """
        source_img, target_img, mask = batch
        x0 = source_img
        x1 = target_img

        # Sample interpolated point and conditional velocity from flow matcher
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)

        # Predict vector field and amyloid fraction
        vt, frac_pred = self.forward(t, xt)

        # --- Flow matching loss ---
        squared_error = (vt - ut) ** 2
        mask_expanded = mask.expand_as(squared_error)
        loss_masked = torch.mean(mask_expanded * squared_error)
        loss_unmasked = torch.mean(squared_error)
        lambda_weight = 0.5
        flow_matching_loss = lambda_weight * loss_masked + (1 - lambda_weight) * loss_unmasked

        # --- Amyloid fraction auxiliary loss ---
        with torch.no_grad():
            B, _, H, W = mask.shape
            frac_true = mask.view(B, -1).float().mean(dim=1)
        aux_loss = F.mse_loss(frac_pred, frac_true)

        # --- Total loss ---
        total_loss = flow_matching_loss + self.aux_loss_weight * aux_loss
        
        # Log losses
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/flow_matching_loss", flow_matching_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/aux_loss", aux_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :param batch_idx: The index of the batch
        """
        source_img, target_img, mask = batch
        x0 = source_img
        x1 = target_img

        # Sample interpolated point and conditional velocity from flow matcher
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)

        # Predict vector field and amyloid fraction
        vt, frac_pred = self.forward(t, xt)

        # --- Flow matching loss ---
        squared_error = (vt - ut) ** 2
        mask_expanded = mask.expand_as(squared_error)
        loss_masked = torch.mean(mask_expanded * squared_error)
        loss_unmasked = torch.mean(squared_error)
        lambda_weight = 0.5
        flow_matching_loss = lambda_weight * loss_masked + (1 - lambda_weight) * loss_unmasked

        # --- Amyloid fraction auxiliary loss ---
        with torch.no_grad():
            B, _, H, W = mask.shape
            frac_true = mask.view(B, -1).float().mean(dim=1)
        aux_loss = F.mse_loss(frac_pred, frac_true)

        # --- Total loss ---
        total_loss = flow_matching_loss + self.aux_loss_weight * aux_loss
        
        # Log validation loss
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/flow_matching_loss", flow_matching_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aux_loss", aux_loss, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :param batch_idx: The index of the batch
        """
        loss = self.model_step(batch)
        
        # Log test loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        :return: A dictionary with optimizer(s) and scheduler(s)
        """
        optimizer = self.optimizer(params=self.parameters())
        
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

    def _velocity_field_only(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Wrapper that returns only velocity field for ODE solver.
        
        :param t: Time tensor
        :param x: Input tensor
        :param kwargs: Additional keyword arguments from NeuralODE (ignored)
        :return: Predicted velocity field only
        """
        vt, _ = self.forward(t, x)
        return vt

    @torch.no_grad()
    def generate(
        self,
        source_img: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Generate target image from source image conditioned on target class.
        
        :param source_img: Source image tensor of shape (B, C, H, W) or (C, H, W)
        :param num_steps: Number of ODE integration steps
        :return: Generated target image
        """
        if self.solver is None:
            raise ValueError("Solver is not initialized. Cannot perform inference.")
        
        self.eval()
        
        # Handle single image case
        if source_img.dim() == 3:
            source_img = source_img.unsqueeze(0)
        
        batch_size = source_img.shape[0]
        device = source_img.device        

        node = NeuralODE(
            self._velocity_field_only,  # Use wrapper that returns only velocity field
            solver=self.solver.solver if hasattr(self.solver, 'solver') else "dopri5",
            sensitivity=self.solver.sensitivity if hasattr(self.solver, 'sensitivity') else "adjoint",
            atol=self.solver.atol if hasattr(self.solver, 'atol') else 1e-4,
            rtol=self.solver.rtol if hasattr(self.solver, 'rtol') else 1e-4,
        )
        
        # Integrate from t=0 to t=1
        t_span = torch.linspace(0, 1, num_steps, device=device)
        traj = node.trajectory(source_img, t_span=t_span)
        
        # Return final image
        return traj[-1]
    
    def _log_images_to_wandb(self, batch_images: torch.Tensor, split: str = "train") -> None:
        """Helper method to log images to wandb.
        
        :param batch_images: List of tuples (source_img, target_img) from dataloader
        :param split: Either 'train' or 'val' to specify which split we're logging
        """
        # Check if wandb logger is available
        wandb_logger = None
        for logger in self.loggers:
            if isinstance(logger, type(self.loggers[0])) and hasattr(logger, 'experiment'):
                # Check if it's wandb by checking the experiment type
                if 'wandb' in str(type(logger.experiment)).lower():
                    wandb_logger = logger
                    break
        
        if wandb_logger is None:
            return  # No wandb logger found, skip logging
        
        # Select up to 10 random images from the batch
        source_imgs, target_imgs = batch_images
        num_images = min(10, len(source_imgs))
        indices = torch.randperm(len(source_imgs))[:num_images]
        
        wandb_images = []
        for idx in indices:
            source_img = source_imgs[idx].unsqueeze(0)  # Add batch dimension
            target_img = target_imgs[idx]
            
            # Generate prediction using the model
            with torch.no_grad():
                generated_img = self.generate(source_img, num_steps=100)
            
            # Denormalize images from [-1, 1] to [0, 1]
            source_img_vis = (source_img.squeeze(0).cpu() + 1) / 2
            target_img_vis = (target_img.cpu() + 1) / 2
            generated_img_vis = (generated_img.squeeze(0).cpu() + 1) / 2
            
            # Clip to [0, 1] range
            source_img_vis = torch.clamp(source_img_vis, 0, 1)
            target_img_vis = torch.clamp(target_img_vis, 0, 1)
            generated_img_vis = torch.clamp(generated_img_vis, 0, 1)
            
            # Convert to numpy and transpose to (H, W, C)
            source_np = source_img_vis.permute(1, 2, 0).numpy()
            target_np = target_img_vis.permute(1, 2, 0).numpy()
            generated_np = generated_img_vis.permute(1, 2, 0).numpy()
            
            # Create wandb image with source, generated, and target
            wandb_images.append(
                wandb.Image(
                    source_np,
                    caption=f"{split}_source_{idx.item()}"
                )
            )
            wandb_images.append(
                wandb.Image(
                    generated_np,
                    caption=f"{split}_generated_{idx.item()}"
                )
            )
            wandb_images.append(
                wandb.Image(
                    target_np,
                    caption=f"{split}_target_{idx.item()}"
                )
            )
        
        # Log to wandb
        wandb_logger.experiment.log({f"{split}/images": wandb_images})
    
    def on_train_epoch_end(self) -> None:
        """Hook called at the end of training epoch to log images."""
        # Get train dataloader
        if self.log_images is False:
            return
        train_dataloader = self.trainer.train_dataloader
        
        if train_dataloader is None:
            return
        
        # Get one batch from train dataloader
        try:
            # Get the first batch
            batch = next(iter(train_dataloader))
            
            # Move batch to device
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                source_imgs = batch[0].to(self.device)
                target_imgs = batch[1].to(self.device)
                batch_images = (source_imgs, target_imgs)
                
                # Log images
                self._log_images_to_wandb(batch_images, split="train")
        except Exception as e:
            print(f"Error logging train images: {e}")
    
    def on_validation_epoch_end(self) -> None:
        """Hook called at the end of validation epoch to log images."""
        if self.log_images is False:
            return
        # Get val dataloader
        val_dataloaders = self.trainer.val_dataloaders
        
        if val_dataloaders is None:
            return
        
        # Handle single dataloader or list of dataloaders
        val_dataloader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders
        
        # Get one batch from val dataloader
        try:
            # Get the first batch
            batch = next(iter(val_dataloader))
            
            # Move batch to device
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                source_imgs = batch[0].to(self.device)
                target_imgs = batch[1].to(self.device)
                batch_images = (source_imgs, target_imgs)
                
                # Log images
                self._log_images_to_wandb(batch_images, split="val")
        except Exception as e:
            print(f"Error logging val images: {e}")
