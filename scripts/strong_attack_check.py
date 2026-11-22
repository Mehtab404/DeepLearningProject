"""Stronger-attack check: PGD with more steps and random restarts on the saved checkpoints.

Purpose: PGD-10 with one random start can overstate robustness. This script re-measures the
robust accuracy with PGD-N and R restarts. An image counts as robust only if it is classified
correctly after EVERY restart (worst case over restarts).

Run from the repository root (a GPU is strongly recommended):
    python scripts/strong_attack_check.py                       # eps=8/255, 50 steps, 5 restarts
    python scripts/strong_attack_check.py --eps 4 8 16          # several budgets
    python scripts/strong_attack_check.py --models CNN-AdvTrained ViT-AdvTrained
Results are printed and saved to outputs_strong_attack/strong_attack_results.csv
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make "src" importable

from src.attacks import check_perturbation, pgd_attack  # noqa: E402
from src.data import get_eval_tensors, prepare_data  # noqa: E402
from src.evaluate import accuracy  # noqa: E402
from src.models import load_checkpoint, load_cnn_clean, load_vit_clean  # noqa: E402

CKPT = {  # model name -> (kind, checkpoint file)
    "CNN-Clean": ("cnn", "cnn_clean.pth"),
    "ViT-Clean": ("vit", "vit_clean.pth"),
    "CNN-AdvTrained": ("cnn", "cnn_adv.pth"),
    "ViT-AdvTrained": ("vit", "vit_adv.pth"),
}


def load_model(name, ckpt_dir, device):
    kind, fname = CKPT[name]
    if kind == "cnn":
        shell = load_cnn_clean(device, pretrained=False)
    else:
        shell, _ = load_vit_clean(device, "timm-tiny", pretrained=False)
    return load_checkpoint(shell, os.path.join(ckpt_dir, fname), device)


def robust_accuracy(model, X, Y, device, eps, steps, restarts, alpha, batch_size, seed):
    """Accuracy (%) under PGD(steps) with `restarts` random starts; worst case over restarts."""
    still_ok = torch.zeros(len(X), dtype=torch.bool)  # True = never fooled so far
    with torch.no_grad():
        for i in range(0, len(X), batch_size):  # start from images that are classified correctly when clean
            x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
            still_ok[i:i + batch_size] = (model(x).argmax(1) == y).cpu()
    for r in range(restarts):
        torch.manual_seed(seed + r)  # a different random start for each restart
        idx_all = torch.nonzero(still_ok).flatten()  # only attack images that are still robust (saves time)
        for j in range(0, len(idx_all), batch_size):
            idx = idx_all[j:j + batch_size]
            x, y = X[idx].to(device), Y[idx].to(device)
            x_adv = pgd_attack(model, x, y, eps, alpha=alpha, steps=steps, random_start=True)
            check_perturbation(x, x_adv, eps)  # safety: stays inside the eps-ball and [0, 1]
            with torch.no_grad():
                fooled = (model(x_adv).argmax(1) != y).cpu()
            still_ok[idx[fooled]] = False
        print(f"    restart {r + 1}/{restarts}: robust accuracy so far {100.0 * still_ok.float().mean():.1f}%", flush=True)
    return 100.0 * still_ok.float().mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["CNN-AdvTrained", "ViT-AdvTrained"], choices=list(CKPT))
    p.add_argument("--eps", nargs="+", type=float, default=[8.0], help="budgets in /255 units")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--n", type=int, default=1000, help="number of stratified test images (same subset as the report)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="outputs_strong_attack")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[strong] device={device}")
    if device.type == "cpu":
        print("[strong] WARNING: no GPU found; this will be very slow (try --n 100 for a quick test).")
    prepare_data(a.data_root)
    X, Y = get_eval_tensors(a.n, a.data_root)

    rows = []
    for name in a.models:
        model = load_model(name, a.ckpt_dir, device)
        clean = accuracy(model, X, Y, device, a.batch_size)
        print(f"[strong] {name}: clean accuracy {clean:.1f}% (report value should match)")
        for e in a.eps:
            eps = e / 255.0
            print(f"  eps={e:g}/255")
            base = robust_accuracy(model, X, Y, device, eps, 10, 1, max(eps / 4, 1 / 255), a.batch_size, a.seed)
            strong = robust_accuracy(model, X, Y, device, eps, a.steps, a.restarts, eps / 10, a.batch_size, a.seed)
            print(f"[strong] {name} eps={e:g}/255: PGD-10 x1 = {base:.1f}%   PGD-{a.steps} x{a.restarts} = {strong:.1f}%")
            rows.append({"model": name, "eps_255": e, "clean": clean, "pgd10_1restart": base,
                         f"pgd{a.steps}_{a.restarts}restarts": strong})

    df = pd.DataFrame(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "strong_attack_results.csv")
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[strong] saved {out}")


if __name__ == "__main__":
    main()
