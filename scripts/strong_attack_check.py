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
# WHAT: Imports command-line, file, table and PyTorch libraries.
# HOW: argparse reads options, pandas writes the CSV, torch runs the attacks.
# WHY: This script re-attacks saved checkpoints.
# PROS & CONS: + Standard. - None.
# EXAMPLE: python scripts/strong_attack_check.py --eps 8
import argparse
import os
import sys

import pandas as pd
import torch

# WHAT: Makes the project's src/ package importable.
# HOW: Adds the repository root to the module search path.
# WHY: The script lives in scripts/, one level below src/.
# PROS & CONS: + Works from any folder. - Path hack instead of an installed package.
# EXAMPLE: import src.attacks works from python scripts/strong_attack_check.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make "src" importable

# WHAT: Imports the project's attack, data, evaluation and model functions.
# HOW: Reuses tested code.
# WHY: The check must use exactly the same attacks and data as the main results.
# PROS & CONS: + Consistent. - None.
# EXAMPLE: pgd_attack, get_eval_tensors, accuracy.
from src.attacks import check_perturbation, pgd_attack  # noqa: E402
from src.data import get_eval_tensors, prepare_data  # noqa: E402
from src.evaluate import accuracy  # noqa: E402
from src.models import load_checkpoint, load_cnn_clean, load_vit_clean  # noqa: E402

# WHAT: Maps a model name to its architecture and checkpoint file.
# HOW: Dictionary.
# WHY: One place to extend with new models.
# PROS & CONS: + Simple. - Hard-coded file names.
# EXAMPLE: 'ViT-AdvTrained': ('vit', 'vit_adv.pth').
CKPT = {  # model name -> (kind, checkpoint file)
    "CNN-Clean": ("cnn", "cnn_clean.pth"),
    "ViT-Clean": ("vit", "vit_clean.pth"),
    "CNN-AdvTrained": ("cnn", "cnn_adv.pth"),
    "ViT-AdvTrained": ("vit", "vit_adv.pth"),
}


# WHAT: Builds a model and loads its saved weights.
# HOW: Creates an empty ResNet-20 or ViT-Tiny and loads the checkpoint.
# WHY: Downloading pretrained weights again is unnecessary.
# PROS & CONS: + Fast. - Needs the checkpoint files.
# EXAMPLE: load_model('CNN-AdvTrained', 'checkpoints', device).
def load_model(name, ckpt_dir, device):
    kind, fname = CKPT[name]
    if kind == "cnn":
        shell = load_cnn_clean(device, pretrained=False)
    else:
        shell, _ = load_vit_clean(device, "timm-tiny", pretrained=False)
    return load_checkpoint(shell, os.path.join(ckpt_dir, fname), device)


# WHAT: Accuracy under a STRONG attack: PGD with many steps and several random restarts, worst case per image.
# HOW: An image counts as robust only if it is classified correctly at the start and after every restart. Only
#      still-robust images are attacked again.
# WHY: PGD-10 with one start can overstate robustness; restarts and more steps give a tighter estimate (lower
#      accuracy).
# PROS & CONS: + Tighter estimate; skipping fooled images saves time. - Still not AutoAttack.
# EXAMPLE: CNN-AdvTrained at 8/255: PGD-10 18.3% -> PGD-50 x5 8.2%.
def robust_accuracy(model, X, Y, device, eps, steps, restarts, alpha, batch_size, seed):
    """Accuracy (%) under PGD(steps) with `restarts` random starts; worst case over restarts."""
    # WHAT: Marks images that are still robust.
    # HOW: Boolean vector, filled with the clean-classification result first.
    # WHY: Misclassified clean images can never count as robust.
    # PROS & CONS: + Correct bookkeeping. - None.
    # EXAMPLE: 88.4% of images start as True for the CNN.
    still_ok = torch.zeros(len(X), dtype=torch.bool)  # True = never fooled so far
    # WHAT: Finds which images are correct before any attack.
    # HOW: Forward pass in batches.
    # WHY: Starting point of the worst-case calculation.
    # PROS & CONS: + Fast. - None.
    # EXAMPLE: clean accuracy 88.4%.
    with torch.no_grad():
        for i in range(0, len(X), batch_size):  # start from images that are classified correctly when clean
            x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
            still_ok[i:i + batch_size] = (model(x).argmax(1) == y).cpu()
    # WHAT: Loop over restarts.
    # HOW: Each restart begins from a new random point.
    # WHY: Different starts find different adversarial examples.
    # PROS & CONS: + Stronger attack. - Cost grows with the number of restarts.
    # EXAMPLE: 5 restarts.
    for r in range(restarts):
        # WHAT: Different random start for every restart.
        # HOW: Seed 42 + r.
        # WHY: Restarts must differ but stay repeatable.
        # PROS & CONS: + Reproducible. - None.
        # EXAMPLE: seeds 42, 43, 44, 45, 46.
        torch.manual_seed(seed + r)  # a different random start for each restart
        # WHAT: Selects only images that are still robust.
        # HOW: Indices of True entries.
        # WHY: No need to attack images already fooled; saves most of the time.
        # PROS & CONS: + Big speed-up. - None.
        # EXAMPLE: After restart 1 only 108 of 1000 CNN images remain.
        idx_all = torch.nonzero(still_ok).flatten()  # only attack images that are still robust (saves time)
        # WHAT: Attacks the selected images in batches.
        # HOW: Slices of 100.
        # WHY: Memory limit.
        # PROS & CONS: + Works on small GPUs. - None.
        # EXAMPLE: 10 batches at most.
        for j in range(0, len(idx_all), batch_size):
            idx = idx_all[j:j + batch_size]
            x, y = X[idx].to(device), Y[idx].to(device)
            # WHAT: PGD attack with many steps.
            # HOW: Random start, 50 steps, alpha = eps/10.
            # WHY: More steps with smaller alpha follow the loss surface more precisely.
            # PROS & CONS: + Stronger than PGD-10. - Five times the cost per restart.
            # EXAMPLE: 50 forward and backward passes per batch.
            x_adv = pgd_attack(model, x, y, eps, alpha=alpha, steps=steps, random_start=True)
            # WHAT: Safety check for the strong attack.
            # HOW: Asserts L-infinity <= eps and the [0, 1] range.
            # WHY: A stronger attack must still be a legal attack.
            # PROS & CONS: + Trustworthy numbers. - Small cost.
            # EXAMPLE: Raises if a pixel moved more than 8/255.
            check_perturbation(x, x_adv, eps)  # safety: stays inside the eps-ball and [0, 1]
            with torch.no_grad():
                # WHAT: Finds which images are now misclassified.
                # HOW: argmax differs from the label.
                # WHY: Those images are no longer robust.
                # PROS & CONS: + Direct. - None.
                # EXAMPLE: boolean vector.
                fooled = (model(x_adv).argmax(1) != y).cpu()
            # WHAT: Removes fooled images from the robust set.
            # HOW: Index assignment.
            # WHY: Worst-case over restarts.
            # PROS & CONS: + Monotone: accuracy can only fall. - None.
            # EXAMPLE: CNN: 10.8% after restart 1, 8.2% after restart 5.
            still_ok[idx[fooled]] = False
        # WHAT: Progress message.
        # HOW: Prints accuracy after each restart.
        # WHY: Shows how quickly the numbers stop falling.
        # PROS & CONS: + Informative. - None.
        # EXAMPLE: restart 1/5: robust accuracy so far 10.8%
        print(f"    restart {r + 1}/{restarts}: robust accuracy so far {100.0 * still_ok.float().mean():.1f}%", flush=True)
    # WHAT: Final robust accuracy.
    # HOW: Share of images never fooled, in percent.
    # WHY: Same unit as the report tables.
    # PROS & CONS: + Comparable. - None.
    # EXAMPLE: 8.2
    return 100.0 * still_ok.float().mean().item()


# WHAT: Runs the check for the chosen models and budgets.
# HOW: Loads data and models, evaluates PGD-10 x1 and PGD-50 x5, saves a CSV.
# WHY: Evidence for Table 5 of the report.
# PROS & CONS: + One command. - Needs a GPU for the ViT.
# EXAMPLE: python scripts/strong_attack_check.py --eps 8
def main():
    # WHAT: Options of the script.
    # HOW: --models, --eps, --steps, --restarts, --n, --ckpt-dir, --data-root, --out-dir, --seed.
    # WHY: Reusable for other budgets.
    # PROS & CONS: + Flexible. - None.
    # EXAMPLE: --eps 4 16
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
    # WHAT: Reads the options.
    # HOW: argparse.
    # WHY: Used below.
    # PROS & CONS: + Standard. - None.
    # EXAMPLE: a.eps == [8.0]
    a = p.parse_args()

    # WHAT: Chooses the device and warns on CPU.
    # HOW: cuda if available.
    # WHY: The ViT at 224x224 is very slow on a CPU.
    # PROS & CONS: + Clear warning. - None.
    # EXAMPLE: [strong] device=cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[strong] device={device}")
    if device.type == "cpu":
        print("[strong] WARNING: no GPU found; this will be very slow (try --n 100 for a quick test).")
    # WHAT: Makes sure the data exists, then builds the same 1000-image subset.
    # HOW: prepare_data and get_eval_tensors(1000).
    # WHY: Results must be comparable with the main tables.
    # PROS & CONS: + Same subset. - None.
    # EXAMPLE: (1000, 3, 32, 32).
    prepare_data(a.data_root)
    X, Y = get_eval_tensors(a.n, a.data_root)

    # WHAT: Result list.
    # HOW: One row per model and eps.
    # WHY: Becomes the CSV.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: []
    rows = []
    # WHAT: Loops over the models.
    # HOW: Loads each and prints its clean accuracy.
    # WHY: The clean accuracy must match the report (88.4% and 94.1%) - it proves the right checkpoint was
    #      loaded.
    # PROS & CONS: + Built-in sanity check. - None.
    # EXAMPLE: [strong] CNN-AdvTrained: clean accuracy 88.4%
    for name in a.models:
        model = load_model(name, a.ckpt_dir, device)
        clean = accuracy(model, X, Y, device, a.batch_size)
        print(f"[strong] {name}: clean accuracy {clean:.1f}% (report value should match)")
        # WHAT: Loops over the budgets.
        # HOW: Converts eps to pixel units.
        # WHY: One result per budget.
        # PROS & CONS: + Flexible. - Cost grows with the number of budgets.
        # EXAMPLE: 8/255.
        for e in a.eps:
            eps = e / 255.0
            print(f"  eps={e:g}/255")
            # WHAT: Two measurements: PGD-10 with one start (baseline) and PGD-50 with 5 restarts.
            # HOW: Both use the same function.
            # WHY: Direct comparison of weak and strong attack.
            # PROS & CONS: + Clear. - None.
            # EXAMPLE: 18.3 and 8.2.
            base = robust_accuracy(model, X, Y, device, eps, 10, 1, max(eps / 4, 1 / 255), a.batch_size, a.seed)
            strong = robust_accuracy(model, X, Y, device, eps, a.steps, a.restarts, eps / 10, a.batch_size, a.seed)
            print(f"[strong] {name} eps={e:g}/255: PGD-10 x1 = {base:.1f}%   PGD-{a.steps} x{a.restarts} = {strong:.1f}%")
            # WHAT: Stores the result.
            # HOW: Dictionary row.
            # WHY: Feeds the CSV.
            # PROS & CONS: + Simple. - None.
            # EXAMPLE: {'model': 'ViT-AdvTrained', 'eps_255': 8.0, ...}
            rows.append({"model": name, "eps_255": e, "clean": clean, "pgd10_1restart": base,
                         f"pgd{a.steps}_{a.restarts}restarts": strong})

    # WHAT: Saves and prints the table.
    # HOW: CSV in outputs_strong_attack/.
    # WHY: Evidence file for the report.
    # PROS & CONS: + Reproducible. - None.
    # EXAMPLE: outputs_strong_attack/strong_attack_results.csv
    df = pd.DataFrame(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "strong_attack_results.csv")
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[strong] saved {out}")


# WHAT: Starts main() when the file is run.
# HOW: Python sets __name__.
# WHY: Importable by extra_cnn_experiments.py without running.
# PROS & CONS: + Standard. - None.
# EXAMPLE: python scripts/strong_attack_check.py
if __name__ == "__main__":
    main()
