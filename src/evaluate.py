"""Evaluation: epsilon sweep (FGSM / PGD-10), cross-model transferability, result tables.

All attacks run in full fp32 (no autocast) so the reported numbers are not affected by fp16 noise.
"""
import os

import pandas as pd
import torch

from .attacks import fgsm_attack, pgd_attack

EPS_LIST = [0, 1, 2, 4, 8, 16]            # in units of 1/255
MODEL_ORDER = ["CNN-Clean", "ViT-Clean", "CNN-AdvTrained", "ViT-AdvTrained"]


@torch.no_grad()
def _correct(model, x, y):
    return (model(x).argmax(1) == y).sum().item()


def accuracy(model, X, Y, device, batch_size=100):
    model.eval()
    c = 0
    for i in range(0, len(X), batch_size):
        x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        c += _correct(model, x, y)
    return 100.0 * c / len(X)


def _attack_fn(name, eps, pgd_steps):
    if name == "FGSM":
        return lambda m, x, y: fgsm_attack(m, x, y, eps)
    if name == "PGD-10":
        # alpha = max(eps/4, 1/255), 10 steps, random start (the evaluation defaults from the spec)
        return lambda m, x, y: pgd_attack(m, x, y, eps, alpha=max(eps / 4, 1 / 255), steps=pgd_steps,
                                          random_start=True)
    raise ValueError(name)


def evaluate_epsilon_sweep(models, X, Y, device, eps_list=EPS_LIST, pgd_steps=10, batch_size=100, seed=42):
    """White-box accuracy of every model vs. epsilon under FGSM and PGD. Returns a long-format DataFrame."""
    rows = []
    for name, model in models.items():
        model.eval()
        for attack in ("FGSM", "PGD-10"):
            for e in eps_list:
                torch.manual_seed(seed)                      # reproducible random starts
                eps = e / 255.0
                fn = _attack_fn(attack, eps, pgd_steps)
                correct = 0
                for i in range(0, len(X), batch_size):
                    x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
                    x_adv = fn(model, x, y)
                    correct += _correct(model, x_adv, y)
                acc = 100.0 * correct / len(X)
                rows.append({"model": name, "attack": attack, "eps_255": e, "accuracy": acc})
                print(f"[eval] {name:15s} {attack:6s} eps={e:2d}/255 -> {acc:5.1f}%")
    return pd.DataFrame(rows)


def evaluate_transferability(models, X, Y, device, eps=8 / 255, pgd_steps=10, batch_size=100, seed=42):
    """matrix.loc[source, target] = accuracy of `target` on PGD examples crafted against `source`.

    The diagonal is the white-box accuracy. Adversarial examples are generated once per source model.
    """
    names = list(models.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for src in names:
        torch.manual_seed(seed)
        models[src].eval()
        adv_batches = []
        for i in range(0, len(X), batch_size):
            x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
            adv_batches.append(pgd_attack(models[src], x, y, eps, alpha=max(eps / 4, 1 / 255),
                                          steps=pgd_steps, random_start=True).cpu())
        X_adv = torch.cat(adv_batches)
        for tgt in names:
            mat.loc[src, tgt] = accuracy(models[tgt], X_adv, Y, device, batch_size)
        print(f"[transfer] source={src:15s} -> " + ", ".join(f"{t}: {mat.loc[src, t]:.1f}%" for t in names))
    mat.index.name = "source \\ target"
    return mat


def to_markdown(df, floatfmt="{:.1f}"):
    """Minimal DataFrame -> GitHub markdown table (avoids the optional `tabulate` dependency)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export_results(sweep_df, transfer_df, out_dir, note=""):
    """Deliverable 5: results_summary.csv / .md (+ transferability.csv)."""
    os.makedirs(out_dir, exist_ok=True)
    sweep_df.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)
    wide = sweep_df.pivot_table(index=["attack", "model"], columns="eps_255", values="accuracy", sort=False)
    wide.columns = [f"eps={c}/255" for c in wide.columns]
    wide.index = [f"{a} | {m}" for a, m in wide.index]
    wide.index.name = "attack | model"
    md = ["# Results summary (accuracy %, white-box)", ""]
    if note:
        md += [f"> {note}", ""]
    md += [to_markdown(wide), "", "## Transferability: PGD-10 at eps=8/255 (rows = source, cols = target; accuracy %)", "",
           to_markdown(transfer_df), ""]
    with open(os.path.join(out_dir, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    transfer_df.to_csv(os.path.join(out_dir, "transferability.csv"))
    print("\n".join(md))
