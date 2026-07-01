"""
MNIST 0-9 classification with the same MLP trained by:
1. SGD with momentum
2. Damped mini-batch Natural Gradient (NG) solved by Fisher-vector products + conjugate gradient

Run:
    pip install torch torchvision matplotlib kagglehub
    python Q6.py

For a faster CPU smoke test, set TRAIN_LIMIT = 10000 and EPOCHS = 2 below.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.matplotlib_cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -------------------------
# Config
# -------------------------
SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = "./data"
OUTPUT_DIR = SCRIPT_DIR / "Q6_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_COLORS = {
    "blue": "#6a8caf",
    "orange": "#d08c60",
    "green": "#6b9f71",
    "red": "#c44e52",
    "purple": "#8172b3",
    "brown": "#937860",
}

EPOCHS = 5
BATCH_SIZE = 512
TEST_BATCH_SIZE = 1024
TRAIN_LIMIT = None  # set to 10000 for a quick CPU test; None means full MNIST training set

SGD_LR = 0.10
SGD_MOMENTUM = 0.90

# NG hyperparameters. NG is much more expensive per mini-batch than SGD.
NG_LR = 0.05
NG_DAMPING = 0.10
NG_CG_ITERS = 8
NG_CG_TOL = 1e-10
NG_MAX_STEP_NORM = 10.0  # safety clipping for the natural-gradient direction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False


# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Data
# -------------------------
MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def mnist_raw_dir() -> Path:
    return Path(DATA_DIR) / "MNIST" / "raw"


def has_mnist_raw_files() -> bool:
    raw_dir = mnist_raw_dir()
    return all((raw_dir / name).is_file() and (raw_dir / name).stat().st_size > 0 for name in MNIST_RAW_FILES)


def extract_mnist_archive(archive_path: Path) -> bool:
    if not archive_path.is_file():
        return False

    raw_dir = mnist_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting MNIST files from: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for target_name in MNIST_RAW_FILES:
            exact_matches = [name for name in names if name == target_name]
            nested_matches = [name for name in names if name.endswith(f"/{target_name}")]
            matches = exact_matches or nested_matches
            if not matches:
                raise FileNotFoundError(f"{target_name} not found in {archive_path}")

            with archive.open(matches[0]) as source, (raw_dir / target_name).open("wb") as destination:
                shutil.copyfileobj(source, destination)

    return has_mnist_raw_files()


def copy_mnist_files_from_dir(source_dir: Path) -> bool:
    raw_dir = mnist_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    for target_name in MNIST_RAW_FILES:
        matches = [path for path in source_dir.rglob(target_name) if path.is_file()]
        if not matches:
            return False
        shutil.copy2(matches[0], raw_dir / target_name)

    return has_mnist_raw_files()


def download_mnist_with_kagglehub() -> bool:
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed; skipping KaggleHub MNIST download.")
        return False

    print("Downloading MNIST dataset with kagglehub...")
    path = Path(kagglehub.dataset_download("hojjatk/mnist-dataset"))
    print(f"Path to dataset files: {path}")

    if copy_mnist_files_from_dir(path):
        return True

    for archive_path in path.rglob("*.zip"):
        if extract_mnist_archive(archive_path):
            return True

    return False


def prepare_mnist_data() -> None:
    if has_mnist_raw_files():
        return

    local_archive = mnist_raw_dir() / "archive.zip"
    if extract_mnist_archive(local_archive):
        return

    if download_mnist_with_kagglehub():
        return

    print("MNIST raw files were not found locally; falling back to torchvision download.")


def make_datasets() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    prepare_mnist_data()
    download = not has_mnist_raw_files()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=download, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=download, transform=transform)

    if TRAIN_LIMIT is not None:
        train_set = Subset(train_set, range(TRAIN_LIMIT))

    return train_set, test_set


def make_loaders(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    # Re-create the train loader for each optimizer so both see the same shuffle order.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )
    return train_loader, test_loader


# -------------------------
# Utilities
# -------------------------
def trainable_params(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_grads(
    grads: Iterable[torch.Tensor | None],
    params: Iterable[torch.nn.Parameter],
) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.contiguous().reshape(-1))
    return torch.cat(pieces)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_count += y.numel()

    return total_loss / total_count, total_correct / total_count


def conjugate_gradient(
    Avp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    n_steps: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Approximately solve A x = b for SPD A using conjugate gradient."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)

    for _ in range(n_steps):
        Ap = Avp(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-12)
        p = r + beta * p
        rdotr = new_rdotr

    return x


def fisher_vector_product(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """
    Compute (F + damping * I) v without forming F explicitly.

    F is approximated by the Hessian of KL[p_old(.|x) || p_theta(.|x)] at theta = theta_old.
    This is the standard Fisher/Gauss-Newton vector product used in natural-gradient methods.
    """
    params = trainable_params(model)

    logits = model(x)
    prob_old = F.softmax(logits, dim=1).detach()
    log_prob = F.log_softmax(logits, dim=1)
    log_prob_old = torch.log(prob_old + 1e-8)

    kl = (prob_old * (log_prob_old - log_prob)).sum(dim=1).mean()
    kl_grads = torch.autograd.grad(kl, params, create_graph=True)
    flat_kl_grad = flatten_grads(kl_grads, params)

    grad_v = torch.dot(flat_kl_grad, v)
    hvp = torch.autograd.grad(grad_v, params, retain_graph=False)
    flat_hvp = flatten_grads(hvp, params).detach()

    return flat_hvp + damping * v


# -------------------------
# Training loops
# -------------------------
def train_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.SGD(model.parameters(), lr=SGD_LR, momentum=SGD_MOMENTUM)
    history: List[Dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * y.numel()
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_count += y.numel()

        test_loss, test_acc = evaluate(model, test_loader, device)
        sec = time.perf_counter() - start
        row = {
            "method": "SGD",
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "train_acc": train_correct / train_count,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "seconds": sec,
        }
        history.append(row)
        print(
            f"SGD | epoch {epoch:02d} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | {sec:.1f}s"
        )

    return history


def train_natural_gradient(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    params = trainable_params(model)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)
            grads = torch.autograd.grad(loss, params)
            flat_grad = flatten_grads(grads, params).detach()

            def Avp(v: torch.Tensor) -> torch.Tensor:
                return fisher_vector_product(model, x, v, damping=NG_DAMPING)

            # Solve (F + lambda I) d = grad, then theta <- theta - lr * d
            nat_dir = conjugate_gradient(
                Avp,
                flat_grad,
                n_steps=NG_CG_ITERS,
                residual_tol=NG_CG_TOL,
            )

            if NG_MAX_STEP_NORM is not None:
                step_norm = torch.linalg.vector_norm(nat_dir)
                if step_norm > NG_MAX_STEP_NORM:
                    nat_dir = nat_dir * (NG_MAX_STEP_NORM / (step_norm + 1e-12))

            with torch.no_grad():
                current = parameters_to_vector(params)
                vector_to_parameters(current - NG_LR * nat_dir, params)

            train_loss_sum += float(loss.item()) * y.numel()
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_count += y.numel()

        test_loss, test_acc = evaluate(model, test_loader, device)
        sec = time.perf_counter() - start
        row = {
            "method": "NG",
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "train_acc": train_correct / train_count,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "seconds": sec,
        }
        history.append(row)
        print(
            f"NG  | epoch {epoch:02d} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | {sec:.1f}s"
        )

    return history


# -------------------------
# Output
# -------------------------
def save_results(rows: List[Dict[str, float]]) -> None:
    configure_chinese_font()

    csv_path = OUTPUT_DIR / "sgd_vs_ng_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_method: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    line_styles = {
        "SGD": {"marker": "o", "linestyle": "-", "color": PLOT_COLORS["blue"]},
        "NG": {"marker": "s", "linestyle": "--", "color": PLOT_COLORS["orange"]},
    }
    fallback_markers = ["^", "D", "v", "P", "X", "*"]
    fallback_colors = [
        PLOT_COLORS["green"],
        PLOT_COLORS["red"],
        PLOT_COLORS["purple"],
        PLOT_COLORS["brown"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, metric, title in (
        (axes[0], "train_acc", "训练准确率"),
        (axes[1], "test_acc", "测试准确率"),
    ):
        for idx, (method, hist) in enumerate(by_method.items()):
            xs = [int(r["epoch"]) for r in hist]
            ys = [float(r[metric]) for r in hist]
            style = line_styles.get(
                method,
                {
                    "marker": fallback_markers[idx % len(fallback_markers)],
                    "linestyle": ":" if idx % 2 else "-.",
                    "color": fallback_colors[idx % len(fallback_colors)],
                },
            )
            ax.plot(xs, ys, label=method, linewidth=1.8, markersize=5.5, **style)
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("准确率")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = OUTPUT_DIR / "accuracy_comparison.png"
    plt.savefig(acc_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, metric, title in (
        (axes[0], "train_loss", "训练损失"),
        (axes[1], "test_loss", "测试损失"),
    ):
        for idx, (method, hist) in enumerate(by_method.items()):
            xs = [int(r["epoch"]) for r in hist]
            ys = [float(r[metric]) for r in hist]
            style = line_styles.get(
                method,
                {
                    "marker": fallback_markers[idx % len(fallback_markers)],
                    "linestyle": ":" if idx % 2 else "-.",
                    "color": fallback_colors[idx % len(fallback_colors)],
                },
            )
            ax.plot(xs, ys, label=method, linewidth=1.8, markersize=5.5, **style)
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("损失")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = OUTPUT_DIR / "loss_comparison.png"
    plt.savefig(loss_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {acc_path}")
    print(f"Saved plot: {loss_path}")


def main() -> None:
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    train_set, test_set = make_datasets()

    base_model = MLP().to(DEVICE)
    n_params = count_parameters(base_model)
    fisher_gib = (n_params**2 * 4) / (1024**3)
    print(f"Trainable parameters: {n_params:,}")
    print(f"Full Fisher matrix size in float32 would be about {fisher_gib:.1f} GiB.")
    print("This script avoids storing F by using Fisher-vector products and conjugate gradient.\n")

    initial_state = copy.deepcopy(base_model.state_dict())
    all_rows: List[Dict[str, float]] = []

    # SGD run
    sgd_model = MLP().to(DEVICE)
    sgd_model.load_state_dict(copy.deepcopy(initial_state))
    train_loader, test_loader = make_loaders(train_set, test_set, seed=SEED)
    all_rows.extend(train_sgd(sgd_model, train_loader, test_loader, DEVICE))

    print("")

    # NG run, same initialization and same shuffle order
    ng_model = MLP().to(DEVICE)
    ng_model.load_state_dict(copy.deepcopy(initial_state))
    train_loader, test_loader = make_loaders(train_set, test_set, seed=SEED)
    all_rows.extend(train_natural_gradient(ng_model, train_loader, test_loader, DEVICE))

    save_results(all_rows)


if __name__ == "__main__":
    main()
