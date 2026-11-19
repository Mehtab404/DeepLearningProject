"""The four publication-quality figures (300 DPI PNG)."""
# Figures produced (saved in outputs/):
#   1) acc_vs_epsilon.png              accuracy vs attack strength, one line per model
#   2) clean_vs_robust_tradeoff.png    bar chart: accuracy before vs after adversarial training
#   3) transferability_heatmap.png     4x4 coloured table of the transfer attack results
#   4) adversarial_examples_grid.png   real images, the added noise, and the adversarial images
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; notebooks display the saved PNGs explicitly
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .attacks import pgd_attack
from .models import CIFAR10_CLASSES

DPI = 300    # resolution of saved images (300 = print quality)
COLORS = {"CNN": "#1f77b4", "ViT": "#d62728"}    # blue for CNN, red for ViT


def _style(name):
    # Chooses line colour, line style and marker from the model name, e.g. "ViT-AdvTrained":
    #   colour by architecture (CNN blue / ViT red)
    #   dashed line + square marker for adversarially trained models, solid + circle for clean ones
    arch = name.split("-")[0]
    return COLORS[arch], ("--" if "Adv" in name else "-"), ("s" if "Adv" in name else "o")


def _tag(fig, tag):
    # Writes a small grey note at the bottom of the figure (used to label quick-test plots as "not a result").
    if tag:
        fig.text(0.5, 0.005, tag, ha="center", va="bottom", fontsize=8, color="gray")


def plot_acc_vs_epsilon(df, path, tag=""):
    # Two side-by-side plots (FGSM on the left, PGD-10 on the right). x = eps, y = accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, attack in zip(axes, ("FGSM", "PGD-10")):
        for name in df["model"].unique():
            # Select the rows for this model and attack, ordered by eps.
            d = df[(df.model == name) & (df.attack == attack)].sort_values("eps_255")
            c, ls, mk = _style(name)
            ax.plot(d.eps_255, d.accuracy, color=c, linestyle=ls, marker=mk, linewidth=2, label=name)
        ax.set_title(f"{attack} attack")
        ax.set_xlabel(r"$\epsilon$ (in /255 units)")
        ax.set_xticks(sorted(df.eps_255.unique()))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].legend(loc="upper right")
    fig.suptitle("Accuracy vs. perturbation budget on CIFAR-10 (solid = clean, dashed = adversarially fine-tuned)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _tag(fig, tag)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)    # free memory


def _acc(df, model, attack, eps):
    # Look up one accuracy value in the results table.
    return float(df[(df.model == model) & (df.attack == attack) & (df.eps_255 == eps)].accuracy.iloc[0])


def plot_tradeoff(df, path, tag=""):
    # Shows what adversarial training COSTS (clean accuracy usually drops)
    # and what it GAINS (accuracy under attack goes up). One panel for CNN, one for ViT.
    # Each entry: (label on the x-axis, attack name, eps). FGSM with eps=0 is simply the clean accuracy.
    metrics = [("Clean (eps=0)", "FGSM", 0), ("FGSM (eps=8/255)", "FGSM", 8), ("PGD-10 (eps=8/255)", "PGD-10", 8)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, arch in zip(axes, ("CNN", "ViT")):
        before = [_acc(df, f"{arch}-Clean", a, e) for _, a, e in metrics]
        after = [_acc(df, f"{arch}-AdvTrained", a, e) for _, a, e in metrics]
        xs = np.arange(len(metrics))
        w = 0.36    # bar width
        # Two bars per group: light colour = before, strong colour = after adversarial training.
        ax.bar(xs - w / 2, before, w, label="Before (clean model)", color="#9ecae1" if arch == "CNN" else "#fcae91")
        ax.bar(xs + w / 2, after, w, label="After 10-epoch adv. fine-tuning", color=COLORS[arch])
        for i in range(len(metrics)):
            # delta = change in percentage points (pp). Green if it improved, red if it got worse.
            delta = after[i] - before[i]
            ax.text(xs[i] - w / 2, before[i] + 1, f"{before[i]:.1f}", ha="center", fontsize=8)
            ax.text(xs[i] + w / 2, after[i] + 1, f"{after[i]:.1f}", ha="center", fontsize=8)
            label = r"$\Delta$Acc$_{clean}$" if i == 0 else r"$\Delta$Acc$_{robust}$"
            ax.text(xs[i], max(before[i], after[i]) + 8, f"{label} = {delta:+.1f} pp", ha="center", fontsize=9,
                    fontweight="bold", color="darkgreen" if delta >= 0 else "darkred")
        ax.set_xticks(xs)
        ax.set_xticklabels([m[0] for m in metrics])
        ax.set_title(f"{arch}: cost of adversarial training vs. robustness gain")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _tag(fig, tag)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_transferability(matrix, path, tag=""):
    # Heatmap of the 4x4 transfer table. Colour scale: red = low accuracy (attack worked),
    # green = high accuracy (attack failed). annot=True writes the number in each cell.
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "Target accuracy (%) - lower = stronger attack"})
    ax.set_xlabel("Target model (evaluated on)")
    ax.set_ylabel("Source model (adversarial examples crafted on)")
    ax.set_title(r"Transferability of PGD-10 examples ($\epsilon$ = 8/255); diagonal = white-box")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _tag(fig, tag)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_adversarial_examples(models, X, Y, device, path, eps=8 / 255, source="CNN-Clean", n=4,
                              pgd_steps=10, tag=""):
    """Column 1: clean x + label. Column 2: magnified perturbation. Column 3: x_adv + predictions of all models."""
    src = models[source].eval()
    # pick n images of distinct classes that the source model classifies correctly
    # (an attack only makes sense on images the model gets right to begin with)
    chosen, seen = [], set()
    with torch.no_grad():
        for i in range(len(X)):
            y = int(Y[i])
            if y in seen:
                continue    # already have an image of this class
            if src(X[i:i + 1].to(device)).argmax(1).item() == y:
                chosen.append(i)
                seen.add(y)
            if len(chosen) == n:
                break
    if not chosen:
        chosen = list(range(min(n, len(X))))    # fallback: just take the first images
    x = X[chosen].to(device)
    y = Y[chosen].to(device)
    torch.manual_seed(42)
    x_adv = pgd_attack(src, x, y, eps, alpha=max(eps / 4, 1 / 255), steps=pgd_steps, random_start=True)
    # The real noise is at most 8/255 and invisible. To SHOW it, rescale [-eps, eps] to [0, 1]:
    # grey = no change, bright/dark = pixel pushed up/down.
    pert = ((x_adv - x + eps) / (2 * eps)).clamp(0, 1)

    # Prediction of EVERY model on the adversarial images (made only on the source model).
    preds = {}
    with torch.no_grad():
        for name, m in models.items():
            preds[name] = m.eval()(x_adv).argmax(1).cpu().tolist()

    rows = len(chosen)
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, 3.1 * rows))
    axes = np.atleast_2d(axes)    # keeps axes[r, c] indexing working even when there is only one row
    for r in range(rows):
        # permute(1, 2, 0): PyTorch stores images as (channels, H, W), matplotlib needs (H, W, channels).
        axes[r, 0].imshow(x[r].permute(1, 2, 0).cpu().numpy())
        axes[r, 0].set_title(f"Clean: {CIFAR10_CLASSES[int(y[r])]}", fontsize=10)
        axes[r, 1].imshow(pert[r].permute(1, 2, 0).cpu().numpy())
        axes[r, 1].set_title(r"PGD-10 perturbation, magnified $(x_{adv}-x+\epsilon)/2\epsilon$", fontsize=8)
        axes[r, 2].imshow(x_adv[r].permute(1, 2, 0).detach().cpu().numpy())
        lines = []
        for name in models:
            p = preds[name][r]
            ok = "OK" if p == int(y[r]) else "WRONG"
            lines.append(f"{name}: {CIFAR10_CLASSES[p]} [{ok}]")
        axes[r, 2].set_title(f"x_adv (eps=8/255, crafted on {source})\n" + "\n".join(lines), fontsize=7)
        for c in range(3):
            axes[r, c].axis("off")    # hide the axis ticks around images
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _tag(fig, tag)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def make_all_plots(sweep_df, transfer_df, models, X, Y, device, out_dir, pgd_steps=10, tag=""):
    # Called once from main.py: creates all four figures and returns their file paths.
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "acc_vs_epsilon": os.path.join(out_dir, "acc_vs_epsilon.png"),
        "tradeoff": os.path.join(out_dir, "clean_vs_robust_tradeoff.png"),
        "transfer": os.path.join(out_dir, "transferability_heatmap.png"),
        "examples": os.path.join(out_dir, "adversarial_examples_grid.png"),
    }
    plot_acc_vs_epsilon(sweep_df, paths["acc_vs_epsilon"], tag)
    plot_tradeoff(sweep_df, paths["tradeoff"], tag)
    plot_transferability(transfer_df, paths["transfer"], tag)
    plot_adversarial_examples(models, X, Y, device, paths["examples"], pgd_steps=pgd_steps, tag=tag)
    for p in paths.values():
        print(f"[plots] saved {p}")
    return paths
