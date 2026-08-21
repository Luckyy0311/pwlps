import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union


# =====================================================================
# Helpers
# =====================================================================

def _call(block, x):
    """
    Call a block with tensor, tuple, or dict input.
    """
    if isinstance(x, tuple):
        return block(*x)
    if isinstance(x, dict):
        return block(**x)
    return block(x)


def _as_tensor(output):
    """
    Convert common PyTorch model outputs into a single tensor.
    Supports tensor, tuple, list, and dict outputs.
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        return output[0]

    if isinstance(output, dict):
        for key in (
            "last_hidden_state",
            "hidden_states",
            "logits",
            "output",
            "x",
        ):
            if key in output:
                return output[key]

        return next(iter(output.values()))

    raise TypeError(
        f"Unsupported model output type: {type(output)}. "
        "Wrap your model so it returns a tensor, tuple, list, or dict."
    )


def _detach(x):
    """
    Detach tensor, tuple of tensors, or dict of tensors.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()

    if isinstance(x, tuple):
        return tuple(_detach(v) for v in x)

    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}

    return x


def pool_tensor(x: torch.Tensor, pool_type: str = "auto") -> torch.Tensor:
    """
    Convert an arbitrary block output into shape (batch_size, feature_dim).

    Supported pool_type:
        - "identity": keep tensor if already 2D
        - "flatten": flatten all non-batch dims
        - "mean_sequence": mean over sequence dim, expects (B, S, D)
        - "mean_spatial": mean over spatial dims, expects (B, C, ...)
        - "auto": automatically choose based on tensor dims
    """

    if pool_type == "identity":
        z = x

    elif pool_type == "flatten":
        z = x.flatten(1)

    elif pool_type == "mean_sequence":
        if x.dim() >= 3:
            z = x.mean(dim=1)
        else:
            z = x

    elif pool_type == "mean_spatial":
        if x.dim() > 2:
            z = x.mean(dim=tuple(range(2, x.dim())))
        else:
            z = x

    elif pool_type == "auto":
        if x.dim() == 2:
            # Already (B, D)
            z = x
        elif x.dim() == 3:
            # Assume (B, sequence_length, features)
            z = x.mean(dim=1)
        elif x.dim() >= 4:
            # Assume (B, channels, spatial...)
            z = x.mean(dim=tuple(range(2, x.dim())))
        else:
            z = x

    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

    # Ensure output is (B, D)
    if z.dim() == 0:
        z = z.view(1, 1)
    elif z.dim() == 1:
        z = z.unsqueeze(-1)
    elif z.dim() > 2:
        z = z.flatten(1)

    return z


# =====================================================================
# Universal Local Predictive Settling Block
# =====================================================================

class LocalSettleBlock(nn.Module):
    """
    Universal local predictive settling block.

    This can wrap any differentiable PyTorch block:
        - Linear/MLP block
        - Convolutional block
        - Transformer encoder layer
        - RNN block
        - Residual block
        - Graph neural block

    The wrapped block produces:
        h_hat = F(x)

    The local activity h settles by minimizing:
        E(h) = pi/2 ||h - h_hat||^2 + lambda * CE(C(Pool(h)), y)

    Then the block weights are updated so that F(x) matches h*.
    """

    def __init__(
        self,
        block: nn.Module,
        feature_dim: int,
        num_classes: int,
        pool_type: str = "auto",
        learn_precision: bool = False,
        fixed_precision: float = 1.0,
        precision_reg_weight: float = 0.01,
        clamp_min: Optional[float] = -20.0,
        clamp_max: Optional[float] = 20.0,
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        self.block = block
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.pool_type = pool_type

        self.head = nn.Linear(feature_dim, num_classes)

        self.learn_precision = learn_precision
        self.fixed_precision = float(fixed_precision)
        self.precision_reg_weight = float(precision_reg_weight)

        if self.learn_precision:
            self.log_precision = nn.Parameter(torch.zeros(1))
        else:
            self.log_precision = None

        # Explicit clamp bounds.
        # For ReLU outputs, set clamp_min=0.0 via project_fn or explicitly.
        # For general blocks, symmetric clamping is a safe default.
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.project_fn = project_fn

    def precision(self):
        """
        Returns precision scalar pi.
        """
        if self.learn_precision:
            return F.softplus(self.log_precision).mean() + 1e-6
        return self.fixed_precision

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool block output into shape (B, feature_dim).
        """
        return pool_tensor(x, self.pool_type)

    def transform(self, x):
        """
        Apply the wrapped block and return a tensor.
        """
        out = _call(self.block, x)
        return _as_tensor(out)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """
        Local classifier logits from activity h.

        Note:
            h should be the RAW block output, not pre-pooled.
            This method applies pooling internally.
        """
        z = self.pool(h)
        return self.head(z)

    def _detached_logits(self, h: torch.Tensor) -> torch.Tensor:
        """
        Logits with detached head weights.

        Used during settling so gradients are taken only with respect to h.
        """
        z = self.pool(h)

        return F.linear(
            z,
            self.head.weight.detach(),
            self.head.bias.detach() if self.head.bias is not None else None,
        )

    def settle(
        self,
        x,
        y: torch.Tensor,
        T: int = 5,
        step_size: float = 0.05,
        cls_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Local activity settling.

        Finds h* such that h*:
            1. stays close to F(x)
            2. helps the local classifier predict y

        Optimization:
            h_hat = F(x) is computed ONCE before the loop since it is
            constant across settling steps.
        """

        # Compute feedforward prediction once.
        # This is constant during settling because weights are frozen.
        with torch.no_grad():
            h_hat = self.transform(x)
            h = h_hat.clone()

        pi = self.precision()
        if torch.is_tensor(pi):
            pi = pi.detach().item()

        T = max(0, int(T))

        for _ in range(T):
            h = h.detach().requires_grad_(True)

            logits = self._detached_logits(h)

            pred_error = (
                (h - h_hat)
                .pow(2)
                .flatten(1)
                .mean(dim=1)
                .mean()
            )

            cls_loss = F.cross_entropy(logits, y)

            energy = 0.5 * pi * pred_error + cls_weight * cls_loss

            grad_h = torch.autograd.grad(
                energy,
                h,
                retain_graph=False,
                create_graph=False,
            )[0]

            with torch.no_grad():
                h = h - step_size * grad_h

                if self.project_fn is not None:
                    h = self.project_fn(h)

                # Explicit clamp bounds
                if self.clamp_min is not None or self.clamp_max is not None:
                    h = torch.clamp(
                        h,
                        min=self.clamp_min,
                        max=self.clamp_max,
                    )

        return h.detach()

    def local_loss(
        self,
        x,
        y: torch.Tensor,
        T: int = 5,
        step_size: float = 0.05,
        cls_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Local weight-update loss.

        1. Settle h* using current block.
        2. Train block so F(x) matches h*.
        3. Train local classifier on F(x) and h*.
        4. Optionally train precision.
        """

        h_star = self.settle(
            x=x,
            y=y,
            T=T,
            step_size=step_size,
            cls_weight=cls_weight,
        )

        h_pred = self.transform(x)

        error = (
            (h_star - h_pred)
            .pow(2)
            .flatten(1)
            .mean(dim=1)
            .mean()
        )

        pi = self.precision()
        pi_for_weights = pi.detach() if torch.is_tensor(pi) else pi

        pred_loss = 0.5 * pi_for_weights * error

        logits = self.logits(h_pred)
        cls_loss = F.cross_entropy(logits, y)

        settled_logits = self.logits(h_star)
        settled_cls_loss = F.cross_entropy(settled_logits, y)

        total_loss = pred_loss + cls_weight * (cls_loss + settled_cls_loss)

        if self.learn_precision:
            target_precision = self.fixed_precision

            precision_loss = (
                0.5 * pi * error.detach()
                + self.precision_reg_weight * (pi - target_precision).pow(2)
            )

            total_loss = total_loss + precision_loss

        return total_loss


# =====================================================================
# Universal Local Predictive Settling Model
# =====================================================================

class LocalSettleSequential(nn.Module):
    """
    Universal sequential local predictive settling model.

    This processes blocks sequentially.
    Each block learns locally.
    No global backpropagation graph is kept across blocks.
    """

    def __init__(self, blocks: List[LocalSettleBlock]):
        super().__init__()

        if len(blocks) == 0:
            raise ValueError("blocks must contain at least one LocalSettleBlock")

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        Inference path.

        Step 1: Pass input through all block transformations.
                After this loop, x is the RAW output of the final block.

        Step 2: Apply the final block's local classifier head.
                Note: logits() expects RAW block output and applies
                pooling internally. Do NOT pre-pool x here.
        """
        for block in self.blocks:
            x = block.transform(x)

        # x is now raw output of the final block.
        # logits() will pool it internally.
        return self.blocks[-1].logits(x)

    def local_train_step(
        self,
        x,
        y: torch.Tensor,
        optimizers: List[torch.optim.Optimizer],
        T: int = 5,
        step_size: float = 0.05,
        cls_weight: float = 1.0,
        grad_clip: float = 1.0,
    ) -> float:
        """
        One universal local training step.

        Each block:
            1. receives detached input
            2. settles locally
            3. updates its own weights
            4. passes detached output forward
        """

        if len(optimizers) != len(self.blocks):
            raise ValueError("Number of optimizers must match number of blocks")

        total_loss = 0.0

        for block, optimizer in zip(self.blocks, optimizers):
            x = _detach(x)

            optimizer.zero_grad()

            loss = block.local_loss(
                x=x,
                y=y,
                T=T,
                step_size=step_size,
                cls_weight=cls_weight,
            )

            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    block.parameters(),
                    grad_clip,
                )

            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                x = block.transform(x)

        return total_loss / float(len(self.blocks))


# =====================================================================
# Builder
# =====================================================================

def build_local_sequential(
    blocks: List[nn.Module],
    num_classes: int,
    sample_input: Union[torch.Tensor, tuple, dict],
    pool_types: Optional[List[str]] = None,
    learn_precision: bool = False,
    fixed_precision: float = 1.0,
    precision_reg_weight: float = 0.01,
    clamp_min: Optional[float] = -20.0,
    clamp_max: Optional[float] = 20.0,
    project_fns: Optional[List[Optional[Callable[[torch.Tensor], torch.Tensor]]]] = None,
) -> LocalSettleSequential:
    """
    Build a universal local predictive settling model from ordinary PyTorch blocks.

    Important:
        During construction, blocks are temporarily set to eval() mode and
        run under torch.no_grad(). This avoids mutating stateful layers
        like BatchNorm running statistics during dimension inference.

        If your blocks are stateful and you need specific behavior during
        construction, wrap them accordingly before passing to this builder.

    Example:
        blocks = [
            nn.Conv2d(...),
            nn.TransformerEncoderLayer(...),
            nn.Linear(...),
        ]

        model = build_local_sequential(
            blocks=blocks,
            num_classes=10,
            sample_input=torch.randn(2, 1, 28, 28),
            pool_types=["mean_spatial", "mean_sequence", "identity"],
        )
    """

    if pool_types is None:
        pool_types = ["auto"] * len(blocks)

    if project_fns is None:
        project_fns = [None] * len(blocks)

    if len(pool_types) != len(blocks):
        raise ValueError("pool_types must have the same length as blocks")

    if len(project_fns) != len(blocks):
        raise ValueError("project_fns must have the same length as blocks")

    wrapped_blocks = []
    x = sample_input

    for block, pool_type, project_fn in zip(blocks, pool_types, project_fns):
        # ----------------------------------------------------------
        # Infer feature_dim and advance x using a SINGLE forward pass.
        # Use eval() mode to avoid mutating stateful layers like BatchNorm.
        # ----------------------------------------------------------
        training = getattr(block, "training", False)

        if hasattr(block, "eval"):
            block.eval()

        with torch.no_grad():
            out = _as_tensor(_call(block, x))
            pooled = pool_tensor(out, pool_type)
            feature_dim = pooled.shape[-1]

        if hasattr(block, "train"):
            block.train(training)

        # ----------------------------------------------------------
        # Wrap the block
        # ----------------------------------------------------------
        wrapped_blocks.append(
            LocalSettleBlock(
                block=block,
                feature_dim=feature_dim,
                num_classes=num_classes,
                pool_type=pool_type,
                learn_precision=learn_precision,
                fixed_precision=fixed_precision,
                precision_reg_weight=precision_reg_weight,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                project_fn=project_fn,
            )
        )

        # Advance x using the eval-mode output
        x = out

    return LocalSettleSequential(wrapped_blocks)
