from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

class ClassConditionalFlowMatchingLitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        flow_matcher: ConditionalFlowMatcher,
        solver: Optional[NeuralODE] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ) -> None:
        """Initialize a `ClassConditionalFlowMatchingLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
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
        if compile:
            self.net = torch.compile(self.net)

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        :param t: Time tensor
        :param x: Input tensor
        :param y: Class labels
        :return: Predicted velocity field
        """
        return self.net(t, x, y=y)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        """Perform a single model step for training or validation.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :return: The loss value
        """
        source_img, target_img, target_label = batch
        
        x0 = source_img  # paired source
        x1 = target_img  # paired target (same filename)
        y = target_label.long()  # condition: target domain
        
        # Sample along probability path between x0 and x1
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        
        # Predict vector field conditioned on target label
        vt = self.forward(t, xt, y)
        
        # Compute MSE loss
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
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :param batch_idx: The index of the batch
        """
        loss = self.model_step(batch)
        
        # Log validation loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step.
        
        :param batch: A batch of data (source_img, target_img, target_label)
        :param batch_idx: The index of the batch
        """
        loss = self.model_step(batch)
        
        # Log test loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
        target_class: int,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """Generate target image from source image conditioned on target class.
        
        :param source_img: Source image tensor of shape (B, C, H, W) or (C, H, W)
        :param target_class: Target class label
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
        
        # Create target class labels
        if isinstance(target_class, int):
            y = torch.tensor([target_class] * batch_size, device=device)
        else:
            y = target_class.to(device)
        
        # Wrap model with class conditioning
        class ConditionalWrapper(torch.nn.Module):
            def __init__(self, model, y):
                super().__init__()
                self.model = model
                self.y = y
            
            def forward(self, t, x, **kwargs):
                batch_size = x.shape[0]
                y_expanded = self.y.expand(batch_size) if self.y.dim() == 0 else self.y
                if y_expanded.shape[0] != batch_size:
                    y_expanded = y_expanded[:batch_size]
                return self.model(t, x, y=y_expanded)
        
        conditional_model = ConditionalWrapper(self.net, y)
        conditional_node = NeuralODE(
            conditional_model,
            solver=self.solver.solver if hasattr(self.solver, 'solver') else "dopri5",
            sensitivity=self.solver.sensitivity if hasattr(self.solver, 'sensitivity') else "adjoint",
            atol=self.solver.atol if hasattr(self.solver, 'atol') else 1e-4,
            rtol=self.solver.rtol if hasattr(self.solver, 'rtol') else 1e-4,
        )
        
        # Integrate from t=0 to t=1
        t_span = torch.linspace(0, 1, num_steps, device=device)
        traj = conditional_node.trajectory(source_img, t_span=t_span)
        
        # Return final image
        return traj[-1]
