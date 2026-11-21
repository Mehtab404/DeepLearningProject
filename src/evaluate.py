"""Evaluation: epsilon sweep (FGSM / PGD-10), cross-model transferability, result tables.

All attacks run in full fp32 (no autocast) so the reported numbers are not affected by fp16 noise.
"""
# Two experiments are run here:
#   1) Epsilon sweep: how does accuracy fall as the attack budget eps grows (0, 1, 2, 4, 8, 16 /255)?
#      eps = 0 is the clean accuracy. The "robust accuracy" is the accuracy under attack.
#   2) Transferability: do adversarial images made for one model also fool a DIFFERENT model?
#      This is a "black-box" attack: the attacker does not need the target model's gradients.
import os

import pandas as pd
import torch

from .attacks import fgsm_attack, pgd_attack

# ===================================================================================================
# EASY-LANGUAGE SUMMARY OF THIS FILE (read this first)
# ---------------------------------------------------------------------------------------------------
# What is this file?   It is the EXAMINER. It measures how good each model is, with and without attack.
# We compare 4 models: CNN-Clean, ViT-Clean (normal) and CNN-AdvTrained, ViT-AdvTrained (defended).
# Experiment 1 (epsilon sweep): attack with eps = 0,1,2,4,8,16 (/255). Bigger eps = stronger attack.
#     eps = 0 means no attack, so that number is the normal accuracy.
# Experiment 2 (transferability): make fooled pictures using model A, then test them on model B.
#     If B is fooled too, the attack "transfers". Result is a 4x4 table.
# "Accuracy" = (number of correct predictions / total pictures) * 100.
# "White-box" = attacker knows the model (can use its gradients). Here we attack the same model we test.
# ===================================================================================================

EPS_LIST = [0, 1, 2, 4, 8, 16]            # in units of 1/255
MODEL_ORDER = ["CNN-Clean", "ViT-Clean", "CNN-AdvTrained", "ViT-AdvTrained"]


# @torch.no_grad(): no gradients are needed just to count correct predictions (faster, less memory).
@torch.no_grad()
def _correct(model, x, y):
    # Number of images in the batch where the predicted class (highest score) equals the true label.
    return (model(x).argmax(1) == y).sum().item()
    # EASY: model(x) gives 10 scores per image; argmax picks the biggest = the model's guess.
    #       "== y" checks if the guess is right; sum() counts the right ones.


def accuracy(model, X, Y, device, batch_size=100):
    # Accuracy in % on the images X, processed in batches of 100 so the GPU memory is not exceeded.
    model.eval()
    c = 0
    for i in range(0, len(X), batch_size):
        x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        c += _correct(model, x, y)
    return 100.0 * c / len(X)


def _attack_fn(name, eps, pgd_steps):
    # Returns the attack function for the given name, with the evaluation settings filled in.
    if name == "FGSM":
        return lambda m, x, y: fgsm_attack(m, x, y, eps)
    if name == "PGD-10":
        # alpha = max(eps/4, 1/255), 10 steps, random start (the evaluation defaults from the spec)
        return lambda m, x, y: pgd_attack(m, x, y, eps, alpha=max(eps / 4, 1 / 255), steps=pgd_steps,
                                          random_start=True)
    raise ValueError(name)


def evaluate_epsilon_sweep(models, X, Y, device, eps_list=EPS_LIST, pgd_steps=10, batch_size=100, seed=42):
    """White-box accuracy of every model vs. epsilon under FGSM and PGD. Returns a long-format DataFrame."""
    # Three nested loops: every model x every attack x every eps value.
    # Result: one row per combination, e.g. {"model": "CNN-Clean", "attack": "PGD-10", "eps_255": 8, "accuracy": 0.0}
    rows = []
    for name, model in models.items():
        model.eval()
        for attack in ("FGSM", "PGD-10"):
            for e in eps_list:
                torch.manual_seed(seed)                      # reproducible random starts
                # White-box: the attack uses the SAME model it then evaluates.
                eps = e / 255.0    # convert e.g. 8 -> 8/255 (pixel values are in [0, 1])
                fn = _attack_fn(attack, eps, pgd_steps)
                correct = 0
                for i in range(0, len(X), batch_size):
                    x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
                    x_adv = fn(model, x, y)               # 1) attack the batch
                    correct += _correct(model, x_adv, y)  # 2) count how many are still classified correctly
                acc = 100.0 * correct / len(X)
                rows.append({"model": name, "attack": attack, "eps_255": e, "accuracy": acc})
                print(f"[eval] {name:15s} {attack:6s} eps={e:2d}/255 -> {acc:5.1f}%")
    return pd.DataFrame(rows)


def evaluate_transferability(models, X, Y, device, eps=8 / 255, pgd_steps=10, batch_size=100, seed=42):
    """matrix.loc[source, target] = accuracy of `target` on PGD examples crafted against `source`.

    The diagonal is the white-box accuracy. Adversarial examples are generated once per source model.
    """
    # Builds a 4x4 table. Row = model used to CREATE the adversarial images (source).
    # Column = model being TESTED on them (target). Low accuracy = the attack transfers well.
    names = list(models.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)    # empty 4x4 table
    for src in names:
        torch.manual_seed(seed)
        models[src].eval()
        adv_batches = []
        for i in range(0, len(X), batch_size):
            x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
            # .cpu() moves the result off the GPU to save GPU memory.
            adv_batches.append(pgd_attack(models[src], x, y, eps, alpha=max(eps / 4, 1 / 255),
                                          steps=pgd_steps, random_start=True).cpu())
        # Craft once on the source model, then test those same images on every target model.
        X_adv = torch.cat(adv_batches)
        for tgt in names:
            mat.loc[src, tgt] = accuracy(models[tgt], X_adv, Y, device, batch_size)
        print(f"[transfer] source={src:15s} -> " + ", ".join(f"{t}: {mat.loc[src, t]:.1f}%" for t in names))
    mat.index.name = "source \\ target"
    return mat


def to_markdown(df, floatfmt="{:.1f}"):
    """Minimal DataFrame -> GitHub markdown table (avoids the optional `tabulate` dependency)."""
    # Produces text like:
    #   | model | eps=0/255 | ... |
    #   | --- | --- | ... |
    #   | CNN-Clean | 92.1 | ... |
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export_results(sweep_df, transfer_df, out_dir, note=""):
    """Deliverable 5: results_summary.csv / .md (+ transferability.csv)."""
    # Saves the results as CSV files (for Excel/pandas) and one Markdown file (readable on GitHub).
    os.makedirs(out_dir, exist_ok=True)
    sweep_df.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    # pivot_table: turn the long list (one row per eps) into a wide table (one column per eps).
    wide = sweep_df.pivot_table(index=["attack", "model"], columns="eps_255", values="accuracy", sort=False)
    wide.columns = [f"eps={c}/255" for c in wide.columns]
    wide.index = [f"{a}, {m}" for a, m in wide.index]   # no '|' here: it would break the markdown table
    wide.index.name = "attack, model"
    md = ["# Results summary (accuracy %, white-box)", ""]
    if note:
        md += [f"> {note}", ""]
    md += [to_markdown(wide), "", "## Transferability: PGD-10 at eps=8/255 (rows = source, cols = target; accuracy %)", "",
           to_markdown(transfer_df), ""]
    with open(os.path.join(out_dir, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    transfer_df.to_csv(os.path.join(out_dir, "transferability.csv"))
    print("\n".join(md))
