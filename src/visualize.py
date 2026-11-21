"""Pictures: draws the 4 result graphs and saves them as image files."""
import os  # work with file and folder paths

import matplotlib  # drawing library
matplotlib.use("Agg")  # draw into files only (no pop-up window), so it also works on a server
import matplotlib.pyplot as plt  # the drawing commands
import numpy as np  # fast number lists
import seaborn as sns  # nicer-looking graphs (we use it for the colour table)
import torch  # main library for deep learning

from .attacks import pgd_attack  # the strong attack, used to make the example pictures
from .models import CIFAR10_CLASSES  # names of the 10 picture types

DPI = 300  # picture sharpness (300 = print quality)
COLORS = {"CNN": "#1f77b4", "ViT": "#d62728"}  # blue for CNN, red for ViT


def _style(name):  # picks line colour, line style and marker shape for a model name
    arch = name.split("-")[0]  # first part of the name: "CNN" or "ViT"
    return COLORS[arch], ("--" if "Adv" in name else "-"), ("s" if "Adv" in name else "o")  # colour by type; dashed line + square for defended models, solid line + circle for normal ones


def _tag(fig, tag):  # writes a small grey note at the bottom of a graph
    if tag:  # only if a note was given (used to mark quick-test graphs as "not a real result")
        fig.text(0.5, 0.005, tag, ha="center", va="bottom", fontsize=8, color="gray")  # place the note in the bottom middle


def plot_acc_vs_epsilon(df, path, tag=""):  # graph 1: accuracy as the attack gets stronger (FGSM on the left, PGD-10 on the right)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)  # a canvas with two graphs side by side that share the up-down scale
    for ax, attack in zip(axes, ("FGSM", "PGD-10")):  # one graph for each attack
        for name in df["model"].unique():  # one line for each model
            d = df[(df.model == name) & (df.attack == attack)].sort_values("eps_255")  # keep this model's rows for this attack, ordered by strength
            c, ls, mk = _style(name)  # colour, line style and marker for this model
            ax.plot(d.eps_255, d.accuracy, color=c, linestyle=ls, marker=mk, linewidth=2, label=name)  # draw the line: attack strength across, accuracy up
        ax.set_title(f"{attack} attack")  # graph title
        ax.set_xlabel(r"$\epsilon$ (in /255 units)")  # label under the graph: attack strength
        ax.set_xticks(sorted(df.eps_255.unique()))  # show a mark for each strength we tested
        ax.grid(alpha=0.3)  # faint grid lines
    axes[0].set_ylabel("Accuracy (%)")  # label on the left side
    axes[0].set_ylim(0, 100)  # the up-down scale goes from 0 to 100 percent
    axes[1].legend(loc="upper right")  # box that says which line is which model
    fig.suptitle("Accuracy vs. perturbation budget on CIFAR-10 (solid = clean, dashed = adversarially fine-tuned)")  # main title
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # fit everything neatly, leaving room at the bottom
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


def _acc(df, model, attack, eps):  # looks up one accuracy value in the results table
    return float(df[(df.model == model) & (df.attack == attack) & (df.eps_255 == eps)].accuracy.iloc[0])  # find the row for this model, attack and strength; take its accuracy


def plot_tradeoff(df, path, tag=""):  # graph 2: what defence training costs (normal accuracy) and gains (accuracy under attack)
    metrics = [("Clean (eps=0)", "FGSM", 0), ("FGSM (eps=8/255)", "FGSM", 8), ("PGD-10 (eps=8/255)", "PGD-10", 8)]  # three tests: no attack, FGSM, PGD-10 (FGSM with eps=0 is just normal accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)  # a canvas with two graphs side by side: CNN and ViT
    for ax, arch in zip(axes, ("CNN", "ViT")):  # one graph for each model type
        before = [_acc(df, f"{arch}-Clean", a, e) for _, a, e in metrics]  # accuracies of the normal model
        after = [_acc(df, f"{arch}-AdvTrained", a, e) for _, a, e in metrics]  # accuracies of the defended model
        xs = np.arange(len(metrics))  # positions of the bar groups: 0, 1, 2
        w = 0.36  # width of one bar
        ax.bar(xs - w / 2, before, w, label="Before (clean model)", color="#9ecae1" if arch == "CNN" else "#fcae91")  # light bars = before defence training
        ax.bar(xs + w / 2, after, w, label="After 10-epoch adv. fine-tuning", color=COLORS[arch])  # strong-colour bars = after defence training
        for i in range(len(metrics)):  # go through the three tests
            delta = after[i] - before[i]  # change in accuracy (in percentage points)
            ax.text(xs[i] - w / 2, before[i] + 1, f"{before[i]:.1f}", ha="center", fontsize=8)  # write the "before" number above its bar
            ax.text(xs[i] + w / 2, after[i] + 1, f"{after[i]:.1f}", ha="center", fontsize=8)  # write the "after" number above its bar
            label = r"$\Delta$Acc$_{clean}$" if i == 0 else r"$\Delta$Acc$_{robust}$"  # name of the change: normal-accuracy change or attacked-accuracy change
            ax.text(xs[i], max(before[i], after[i]) + 8, f"{label} = {delta:+.1f} pp", ha="center", fontsize=9,  # write the change above the bars
                    fontweight="bold", color="darkgreen" if delta >= 0 else "darkred")  # bold; green if accuracy went up, red if it went down
        ax.set_xticks(xs)  # one mark per bar group
        ax.set_xticklabels([m[0] for m in metrics])  # name each group
        ax.set_title(f"{arch}: cost of adversarial training vs. robustness gain")  # graph title
        ax.set_ylim(0, 115)  # the up-down scale (a bit above 100 to leave room for the text)
        ax.grid(axis="y", alpha=0.3)  # faint horizontal grid lines
        ax.legend(loc="upper right", fontsize=8)  # box that says which bar is which
    axes[0].set_ylabel("Accuracy (%)")  # label on the left side
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # fit everything neatly
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


def plot_transferability(matrix, path, tag=""):  # graph 3: colour table showing which model's tricks fool which model
    fig, ax = plt.subplots(figsize=(7.5, 6))  # a canvas for one graph
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax,  # coloured squares with the number written inside; red = low accuracy, green = high
                cbar_kws={"label": "Target accuracy (%) - lower = stronger attack"})  # label of the colour bar
    ax.set_xlabel("Target model (evaluated on)")  # label under the table: the model being tested
    ax.set_ylabel("Source model (adversarial examples crafted on)")  # label on the left: the model that made the tricked images
    ax.set_title(r"Transferability of PGD-10 examples ($\epsilon$ = 8/255); diagonal = white-box")  # title; the diagonal is where the same model is both attacked and tested
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")  # tilt the names under the table so they do not overlap
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # fit everything neatly
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


def plot_adversarial_examples(models, X, Y, device, path, eps=8 / 255, source="CNN-Clean", n=4,  # graph 4: shows real pictures, the added noise, and the tricked pictures
                              pgd_steps=10, tag=""):  # pgd_steps = how many small steps the attack takes
    src = models[source].eval()  # the model that makes the tricked images, in test mode
    chosen, seen = [], set()  # chosen = picked picture positions; seen = classes already picked
    with torch.no_grad():  # no learning information needed here
        for i in range(len(X)):  # go through the test pictures
            y = int(Y[i])  # the true label of this picture
            if y in seen:  # we already have a picture of this class
                continue  # skip it
            if src(X[i:i + 1].to(device)).argmax(1).item() == y:  # the model guesses this picture right (an attack only makes sense then)
                chosen.append(i)  # pick it
                seen.add(y)  # remember this class is done
            if len(chosen) == n:  # we have enough pictures
                break  # stop looking
    if not chosen:  # nothing suitable was found
        chosen = list(range(min(n, len(X))))  # just take the first few pictures
    x = X[chosen].to(device)  # the chosen pictures, on the GPU/CPU
    y = Y[chosen].to(device)  # their true labels
    torch.manual_seed(42)  # fixed starting number so the random start is the same every time
    x_adv = pgd_attack(src, x, y, eps, alpha=max(eps / 4, 1 / 255), steps=pgd_steps, random_start=True)  # make the tricked version of each picture
    pert = ((x_adv - x + eps) / (2 * eps)).clamp(0, 1)  # the real noise is too small to see, so stretch it to the range 0..1 (grey = no change)
    preds = {}  # what each model guesses for the tricked pictures
    with torch.no_grad():  # no learning information needed here
        for name, m in models.items():  # go through all the models
            preds[name] = m.eval()(x_adv).argmax(1).cpu().tolist()  # the guess of this model for each tricked picture
    rows = len(chosen)  # one row of the graph per picture
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, 3.1 * rows))  # a canvas with 3 columns: original, noise, tricked picture
    axes = np.atleast_2d(axes)  # makes sure indexing works even when there is only one row
    for r in range(rows):  # go through each picture
        axes[r, 0].imshow(x[r].permute(1, 2, 0).cpu().numpy())  # show the original picture (colour layers moved to the order the drawing tool wants)
        axes[r, 0].set_title(f"Clean: {CIFAR10_CLASSES[int(y[r])]}", fontsize=10)  # title: its true class name
        axes[r, 1].imshow(pert[r].permute(1, 2, 0).cpu().numpy())  # show the stretched noise
        axes[r, 1].set_title(r"PGD-10 perturbation, magnified $(x_{adv}-x+\epsilon)/2\epsilon$", fontsize=8)  # title: this is the magnified noise
        axes[r, 2].imshow(x_adv[r].permute(1, 2, 0).detach().cpu().numpy())  # show the tricked picture
        lines = []  # text lines with each model's guess
        for name in models:  # go through all the models
            p = preds[name][r]  # this model's guess
            ok = "OK" if p == int(y[r]) else "WRONG"  # is the guess right?
            lines.append(f"{name}: {CIFAR10_CLASSES[p]} [{ok}]")  # write: model name, guessed class, right or wrong
        axes[r, 2].set_title(f"x_adv (eps=8/255, crafted on {source})\n" + "\n".join(lines), fontsize=7)  # title: all models' guesses
        for c in range(3):  # go through the 3 columns
            axes[r, c].axis("off")  # hide the numbers around the picture
    fig.tight_layout(rect=(0, 0.02, 1, 1))  # fit everything neatly
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


def make_all_plots(sweep_df, transfer_df, models, X, Y, device, out_dir, pgd_steps=10, tag=""):  # draws all four graphs and gives back their file paths
    os.makedirs(out_dir, exist_ok=True)  # create the output folder if it is missing
    paths = {  # file name for each graph
        "acc_vs_epsilon": os.path.join(out_dir, "acc_vs_epsilon.png"),  # graph 1
        "tradeoff": os.path.join(out_dir, "clean_vs_robust_tradeoff.png"),  # graph 2
        "transfer": os.path.join(out_dir, "transferability_heatmap.png"),  # graph 3
        "examples": os.path.join(out_dir, "adversarial_examples_grid.png"),  # graph 4
    }
    plot_acc_vs_epsilon(sweep_df, paths["acc_vs_epsilon"], tag)  # draw graph 1
    plot_tradeoff(sweep_df, paths["tradeoff"], tag)  # draw graph 2
    plot_transferability(transfer_df, paths["transfer"], tag)  # draw graph 3
    plot_adversarial_examples(models, X, Y, device, paths["examples"], pgd_steps=pgd_steps, tag=tag)  # draw graph 4
    for p in paths.values():  # go through the saved files
        print(f"[plots] saved {p}")  # tell the user where each one was saved
    return paths  # give back the file paths
