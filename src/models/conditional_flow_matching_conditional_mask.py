import pdb
from typing import Any, Dict, Optional, Tuple
from venv import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
# from loguru import logger

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
        n_images_log: int = 5,
    ) -> None:
        """Initialize a `ConditionalFlowMatchingLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param aux_loss_weight: Weight for auxiliary amyloid fraction loss.
        :param n_images_log: Number of images to log per epoch.
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
        self.n_images_log = n_images_log
        
        if compile:
            self.net = torch.compile(self.net)

    def forward(self, t: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        :param t: Time tensor
        :param x: Input tensor (3 channels RGB)
        :param mask: Mask tensor (1 channel) to condition generation
        :return: Predicted velocity field (3 channels)
        """
        # Concatenate mask as 4th channel: (B, 3, H, W) + (B, 1, H, W) -> (B, 4, H, W)
        x_with_mask = torch.cat([x, mask], dim=1)
        vt = self.net(t, x_with_mask)
        return vt

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single model step for training or validation.

        :param batch: (source_img, target_img, target_seg_mask)
        :return: loss
        """
        source_img, target_img, mask = batch          # mask: (B,1,H,W), ROI=1
        x0, x1 = source_img, target_img

        # ---- Flow-matching MSE (vector field supervision) ----
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        # Pass mask to condition the generation
        vt = self.forward(t, xt, mask)
        loss = torch.mean((vt - ut) ** 2)

        return loss

        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :param batch_idx: The index of the batch
        :return: The loss value
        """
        loss = self.model_step(batch)
        
        # Log training loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :param batch_idx: The index of the batch
        """
        loss = self.model_step(batch)
        
        # Log validation loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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


    @torch.no_grad()
    def generate(
        self,
        source_img: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Generate target image from source image conditioned on mask.
        
        :param source_img: Source image tensor of shape (B, C, H, W) or (C, H, W)
        :param mask: Mask tensor of shape (B, 1, H, W) or (1, H, W) for conditioning
        :param num_steps: Number of ODE integration steps
        :return: Generated target image
        """
        if self.solver is None:
            raise ValueError("Solver is not initialized. Cannot perform inference.")
        
        self.eval()
        
        # Handle single image case
        if source_img.dim() == 3:
            source_img = source_img.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        batch_size = source_img.shape[0]
        device = source_img.device
        
        # Create a wrapper that concatenates mask with the state during ODE integration
        class MaskConditionedWrapper(torch.nn.Module):
            def __init__(self, net, mask):
                super().__init__()
                self.net = net
                self.mask = mask
            
            def forward(self, t, x, **kwargs):
                # Concatenate mask as 4th channel
                x_with_mask = torch.cat([x, self.mask], dim=1)
                return self.net(t, x_with_mask)
        
        wrapped_net = MaskConditionedWrapper(self.net, mask)

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
        
        # Return final image
        return traj[-1]
    
    def _log_images_to_wandb(self, batch_images: torch.Tensor, split: str = "train", epoch: int = None) -> None:
        """Helper method to log images to wandb.
        
        :param batch_images: Tuple of (source_img, target_img, mask) from dataloader
        :param split: Either 'train' or 'val' to specify which split we're logging
        :param epoch: Current epoch number to include in captions
        """
        # Check if wandb logger is available
        wandb_logger = None
        for logger_ in self.loggers:
            if isinstance(logger_, type(self.loggers[0])) and hasattr(logger_, 'experiment'):
                # Check if it's wandb by checking the experiment type
                if 'wandb' in str(type(logger_.experiment)).lower():
                    wandb_logger = logger_
                    break
        
        if wandb_logger is None:
            return  # No wandb logger found, skip logging
        
        # Select up to 10 random images from the batch
        source_imgs, target_imgs, masks = batch_images
        num_images = min(5, len(source_imgs))
        indices = torch.randperm(len(source_imgs))[:num_images]
        logger.debug(f"Logging {num_images} images to wandb for {split} split.")
        
        wandb_images = []
        for idx in indices:
            idx_int = idx.item()  # Convert tensor to int
            source_img = source_imgs[idx_int].unsqueeze(0)  # Add batch dimension
            target_img = target_imgs[idx_int]
            mask = masks[idx_int]
            
            # Generate prediction using the model
            with torch.no_grad():
                mask_input = mask.unsqueeze(0)  # Add batch dimension
                generated_img = self.generate(source_img, mask_input, num_steps=2)
            
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
            
            # Process mask for visualization
            mask_vis = mask.cpu().squeeze()  # Remove channel dim if (1, H, W)
            if mask_vis.dim() == 3:
                mask_vis = mask_vis[0]  # Take first channel if multiple
            mask_np = mask_vis.numpy()*255.0  # Scale to [0, 255] for visibility

            #resize the all images to 128x128 for logging
            # import cv2
            # source_np = cv2.resize(source_np, (128, 128))
            # target_np = cv2.resize(target_np, (128, 128))
            # generated_np = cv2.resize(generated_np, (128, 128))
            # mask_np = cv2.resize(mask_np, (128, 128))
            
            # Create wandb image with source, generated, target, and mask
            epoch_label = f"epoch{epoch}_" if epoch is not None else ""
            wandb_images.append(
                wandb.Image(
                    source_np,
                    caption=f"{epoch_label}{split}_source_{idx_int}"
                )
            )
            wandb_images.append(
                wandb.Image(
                    generated_np,
                    caption=f"{epoch_label}{split}_generated_{idx_int}"
                )
            )
            wandb_images.append(
                wandb.Image(
                    target_np,
                    caption=f"{epoch_label}{split}_target_{idx_int}"
                )
            )
            wandb_images.append(
                wandb.Image(
                    mask_np,
                    caption=f"{epoch_label}{split}_mask_{idx_int}"
                )
            )
        
        # Log to wandb
        wandb_logger.experiment.log({f"{split}/images": wandb_images})
    
    def on_train_epoch_end(self) -> None:
        """Hook called at the end of training epoch to log images."""
        # Get train dataloader
        if self.log_images is False:
            return
        
        # Only rank 0 should log images to avoid deadlock
        if self.trainer.is_global_zero:
            train_dataloader = self.trainer.train_dataloader
            
            if train_dataloader is None:
                return
            
            # pdb.set_trace()
            
            # Get one batch from train dataloader
            # try:
            # Get the first batch
            batch = next(iter(train_dataloader))
            
            # Move batch to device
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                source_imgs = batch[0].to(self.device)
                target_imgs = batch[1].to(self.device)
                masks = batch[2].to(self.device)
                batch_images = (source_imgs, target_imgs, masks)
                
                # Log images
                self._log_images_to_wandb(batch_images, split="train", epoch=self.current_epoch)
        # except Exception as e:
            #     print(f"Error logging train images: {e}")
        
        # Ensure all ranks wait for rank 0 to finish logging
        if self.trainer.world_size > 1:
            torch.distributed.barrier()
    
    def on_validation_epoch_end(self) -> None:
        """Hook called at the end of validation epoch to log images."""
        if self.log_images is False:
            return
        
        # Only rank 0 should log images to avoid deadlock
        if self.trainer.is_global_zero:
            # Get val dataloader
            val_dataloaders = self.trainer.val_dataloaders
            
            if val_dataloaders is None:
                return
            
            # Handle single dataloader or list of dataloaders
            val_dataloader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders
            
            # Get one batch from val dataloader
            # try:
            # Get the first batch
            batch = next(iter(val_dataloader))
            
            # Move batch to device
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                source_imgs = batch[0].to(self.device)
                target_imgs = batch[1].to(self.device)
                masks = batch[2].to(self.device)
                batch_images = (source_imgs, target_imgs, masks)

                # Log images
                self._log_images_to_wandb(batch_images, split="val", epoch=self.current_epoch)
            # except Exception as e:
            #     print(f"Error logging val images: {e}")
        
        # Ensure all ranks wait for rank 0 to finish logging
        if self.trainer.world_size > 1:
            torch.distributed.barrier()
