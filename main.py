import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =====================================================================
# Utilities
# =====================================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# Local Predictive Layer
# =====================================================================

class LocalPredictiveLayer(nn.Module):
    """
    A layer that learns using local predictive settling.

    It contains:
    - feedforward weights W_l
    - local classification head C_l
    - optional precision parameter pi_l

    The layer activity h_l settles by minimizing:

        E_l =
            (pi_l / 2) * || h_l - h_hat_l ||^2
            + cls_weight * CrossEntropy(C_l(h_l), y)

    where h_hat_l is the feedforward prediction from the previous layer.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_classes,
        learn_precision=False,
        fixed_precision=1.0,
        precision_reg_weight=0.01,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(in_dim, out_dim)
        self.head = nn.Linear(out_dim, num_classes)
        self.act = nn.ReLU()

        self.learn_precision = learn_precision
        self.fixed_precision = float(fixed_precision)
        self.precision_reg_weight = float(precision_reg_weight)

        if self.learn_precision:
            # Initialized so softplus(0) ~= 0.693, close to 1.
            self.log_precision = nn.Parameter(torch.zeros(1))
        else:
            self.log_precision = None

    def precision(self):
        """
        Returns precision scalar pi.

        If learn_precision=True, pi is a learnable positive scalar.
        If learn_precision=False, pi is fixed.
        """
        if self.learn_precision:
            return F.softplus(self.log_precision).mean() + 1e-6
        return self.fixed_precision

    def forward(self, x):
        """
        Normal feedforward transformation.
        """
        return self.act(self.linear(x))

    def _detached_head_logits(self, h):
        """
        Compute logits using detached head parameters.

        This is used during activity settling so that gradients are taken
        only with respect to the activity h, not the head weights.
        """
        return F.linear(
            h,
            self.head.weight.detach(),
            self.head.bias.detach() if self.head.bias is not None else None,
        )

    def settle(
        self,
        x,
        y,
        T=5,
        step_size=0.05,
        cls_weight=1.0,
    ):
        """
        Local activity settling.

        Given previous layer activity x, find a local activity h that minimizes:

            E(h) =
                pi/2 ||h - f(W x)||^2
                + cls_weight * CE(C(h), y)

        This is done by local gradient descent on h only.
        We do not update weights during settling.
        """

        # Initial feedforward guess
        with torch.no_grad():
            h = self.forward(x).clone()

        pi = self.precision()
        if torch.is_tensor(pi):
            pi = pi.detach().item()

        T = max(0, int(T))

        for _ in range(T):
            h = h.detach().requires_grad_(True)

            # Feedforward prediction from previous layer.
            # Detached because during settling we update h, not W.
            with torch.no_grad():
                h_hat = self.forward(x)

            # Local classification logits.
            # Head parameters are detached because we only want dE/dh here.
            logits = self._detached_head_logits(h)

            pred_error = ((h - h_hat) ** 2).sum(dim=1).mean()
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
                h = self.act(h)

                # Simple stability clamp.
                # ReLU already enforces min 0, but max clamp prevents explosion.
                h = torch.clamp(h, min=0.0, max=20.0)

        return h.detach()

    def local_loss(
        self,
        x,
        y,
        T=5,
        step_size=0.05,
        cls_weight=1.0,
    ):
        """
        Local weight-update loss.

        1. Settle h* using current weights.
        2. Update feedforward weights so that f(W x) matches h*.
        3. Update local classifier using current and settled activities.
        4. Optionally update precision.
        """

        # Settled activity target
        h_star = self.settle(
            x=x,
            y=y,
            T=T,
            step_size=step_size,
            cls_weight=cls_weight,
        )

        # Current feedforward prediction
        h_pred = self.forward(x)

        # Prediction error between settled activity and feedforward output
        error = ((h_star - h_pred) ** 2).sum(dim=1).mean()

        pi = self.precision()

        # Use precision as a scalar weight for the predictive loss.
        # Detach it for weight updates so precision does not collapse
        # trivially through the main weights.
        pi_for_weights = pi.detach() if torch.is_tensor(pi) else pi

        pred_loss = 0.5 * pi_for_weights * error

        # Local classification on current feedforward activity
        logits = self.head(h_pred)
        cls_loss = F.cross_entropy(logits, y)

        # Also train head on settled activity
        settled_logits = self.head(h_star)
        settled_cls_loss = F.cross_entropy(settled_logits, y)

        total_loss = pred_loss + cls_weight * (cls_loss + settled_cls_loss)

        # Optional precision learning
        if self.learn_precision:
            # Precision adapts to detached prediction error.
            # Regularization keeps it near fixed_precision and avoids collapse.
            target_precision = self.fixed_precision
            precision_loss = (
                0.5 * pi * error.detach()
                + self.precision_reg_weight * (pi - target_precision).pow(2)
            )
            total_loss = total_loss + precision_loss

        return total_loss


# =====================================================================
# Local Predictive MLP
# =====================================================================

class LocalPredictiveMLP(nn.Module):
    """
    Multi-layer local predictive network.

    Each layer has:
    - local feedforward weights
    - local classifier head
    - local predictive settling

    During training, layers are processed sequentially.
    No global backpropagation graph is kept across layers.
    """

    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
        num_layers=2,
        learn_precision=False,
        fixed_precision=1.0,
        precision_reg_weight=0.01,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []

        layers.append(
            LocalPredictiveLayer(
                in_dim=input_dim,
                out_dim=hidden_dim,
                num_classes=num_classes,
                learn_precision=learn_precision,
                fixed_precision=fixed_precision,
                precision_reg_weight=precision_reg_weight,
            )
        )

        for _ in range(num_layers - 1):
            layers.append(
                LocalPredictiveLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_classes=num_classes,
                    learn_precision=learn_precision,
                    fixed_precision=fixed_precision,
                    precision_reg_weight=precision_reg_weight,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Inference path.

        Uses all feedforward layers and the final local head as classifier.
        """
        x = x.view(x.size(0), -1)

        for layer in self.layers:
            x = layer(x)

        logits = self.layers[-1].head(x)
        return logits

    def local_train_step(
        self,
        x,
        y,
        optimizers,
        T=5,
        step_size=0.05,
        cls_weight=1.0,
        grad_clip=1.0,
    ):
        """
        One local training step over all layers.

        Important:
        - x is detached before every layer.
        - each layer is updated independently.
        - no global computational graph is retained.
        """

        x = x.view(x.size(0), -1)
        total_loss = 0.0

        for layer, optimizer in zip(self.layers, optimizers):
            x = x.detach()

            optimizer.zero_grad()

            loss = layer.local_loss(
                x=x,
                y=y,
                T=T,
                step_size=step_size,
                cls_weight=cls_weight,
            )

            loss.backward()

            # Stability for local learning
            torch.nn.utils.clip_grad_norm_(layer.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()

            # Pass activity forward using updated layer.
            # No graph is kept.
            with torch.no_grad():
                x = layer(x)

        return total_loss / float(len(self.layers))


# =====================================================================
# Standard Backprop MLP Baseline
# =====================================================================

class StandardMLP(nn.Module):
    """
    Standard MLP trained with global backpropagation.
    """

    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        num_classes=10,
        num_layers=2,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        modules = [
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]

        for _ in range(num_layers - 1):
            modules += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]

        modules.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


# =====================================================================
# Training functions
# =====================================================================

def train_bp_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(1, total_samples)


def train_local_epoch(
    model,
    loader,
    optimizers,
    device,
    T=5,
    step_size=0.05,
    cls_weight=1.0,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        loss = model.local_train_step(
            x=x,
            y=y,
            optimizers=optimizers,
            T=T,
            step_size=step_size,
            cls_weight=cls_weight,
        )

        total_loss += loss * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(1, total)


# =====================================================================
# Main experiment
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Precision-Weighted Local Predictive Settling"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="local",
        choices=["bp", "local"],
        help="bp = standard backprop baseline, local = predictive settling",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=2)

    parser.add_argument(
        "--T",
        type=int,
        default=5,
        help="Number of local settling steps",
    )

    parser.add_argument(
        "--settle-lr",
        type=float,
        default=0.05,
        help="Step size for local activity settling",
    )

    parser.add_argument(
        "--cls-weight",
        type=float,
        default=1.0,
        help="Weight of local classification loss",
    )

    parser.add_argument(
        "--learn-precision",
        action="store_true",
        help="Learn layer-wise precision. If not set, precision is fixed.",
    )

    parser.add_argument(
        "--fixed-precision",
        type=float,
        default=1.0,
        help="Fixed precision value, or regularization target if learned.",
    )

    parser.add_argument(
        "--precision-reg",
        type=float,
        default=0.01,
        help="Regularization weight for learned precision.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="./data")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"Arguments: {args}")
    print("-" * 80)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    if args.method == "bp":
        model = StandardMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        print(f"StandardMLP parameters: {count_parameters(model)}")
        print("-" * 80)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            reset_memory_stats()
            train_loss = train_bp_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            peak_mem = get_peak_memory_mb()

            test_acc = evaluate(model, test_loader, device)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Test Acc {test_acc:.4f} | "
                f"Peak Train Mem {peak_mem:.2f} MB | "
                f"Time {elapsed:.2f}s"
            )

    elif args.method == "local":
        model = LocalPredictiveMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
            learn_precision=args.learn_precision,
            fixed_precision=args.fixed_precision,
            precision_reg_weight=args.precision_reg,
        ).to(device)

        optimizers = [
            torch.optim.Adam(layer.parameters(), lr=args.lr)
            for layer in model.layers
        ]

        print(f"LocalPredictiveMLP parameters: {count_parameters(model)}")
        print("-" * 80)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            reset_memory_stats()
            train_loss = train_local_epoch(
                model=model,
                loader=train_loader,
                optimizers=optimizers,
                device=device,
                T=args.T,
                step_size=args.settle_lr,
                cls_weight=args.cls_weight,
            )
            peak_mem = get_peak_memory_mb()

            test_acc = evaluate(model, test_loader, device)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Test Acc {test_acc:.4f} | "
                f"Peak Train Mem {peak_mem:.2f} MB | "
                f"Time {elapsed:.2f}s"
            )


if __name__ == "__main__":
    main()
