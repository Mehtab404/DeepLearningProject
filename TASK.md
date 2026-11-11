# Autonomous Agent Task List & GitHub Auto-Push Protocol

> **CRITICAL DIRECTIVE FOR THE AI AGENT:**
> You must execute the tasks below sequentially following `AGENT_PROJECT_SPEC.md`. 
> **GitHub Auto-Push Rule:** Immediately after completing **ANY** numbered Milestone (or every 15 minutes during training/evaluation), you **MUST**:
> 1. Mark the completed checkboxes `[x]` in this `TASK.md` file.
> 2. Stage all newly created/modified code, logs, metrics, and plots.
> 3. Commit with the specified semantic commit message.
> 4. Push directly to the remote GitHub repository (`git push origin main`).

---

## 0. Git Auto-Push Helper Script (Setup First)

Before starting Milestone 1, create an executable helper script `scripts/auto_push.sh` (and a Python equivalent inside the Colab notebook) so both local agents and Google Colab sessions can push automatically after every task:

```bash
#!/usr/bin/env bash
# Usage: ./scripts/auto_push.sh "feat: completed milestone X"
COMMIT_MSG=${1:-"chore: automated task checkpoint update"}

git add .
if ! git diff --cached --quiet; then
    git commit -m "$COMMIT_MSG"
    git push origin HEAD
    echo "[AUTO-PUSH] Successfully pushed to GitHub: $COMMIT_MSG"
else
    echo "[AUTO-PUSH] No file changes to commit."
fi
```

*Note for `.gitignore`*: Keep `outputs/*.png`, `outputs/*.csv`, and `outputs/*.md` tracked in Git so plots appear on GitHub! If `.pth` checkpoints exceed GitHub's 100MB file limit, track `checkpoints/training_log.json` in Git and save heavy `.pth` files to Google Drive (or Git LFS).

---

## Milestone 1: Repository Initialization & Environment Setup
- [x] **1.1.** Initialize Git repository (if not already initialized) and verify remote `origin` URL.
- [x] **1.2.** Create directory structure: `src/`, `checkpoints/`, `outputs/`, `scripts/`.
- [x] **1.3.** Create `.gitignore` (ignoring `__pycache__/`, `.venv/`, `data/`, and single `.pth` files $>95\text{MB}$ while keeping `outputs/` and `TASK.md` tracked).
- [x] **1.4.** Create `requirements.txt` (`torch`, `torchvision`, `timm`, `transformers`, `matplotlib`, `seaborn`, `tqdm`, `pandas`, `numpy`).
- [x] **1.5.** Create `scripts/auto_push.sh` and make it executable (`chmod +x scripts/auto_push.sh`).
- [x] **1.6. [AUTO-PUSH TRIGGER]:** Update `TASK.md` checkboxes `[x]` and run:
  ```bash
  ./scripts/auto_push.sh "chore(init): setup project structure, requirements, and auto-push script [Milestone 1]"
  ```

---

## Milestone 2: Pretrained CIFAR-10 Models & Normalization Wrapper (`src/models.py`)
- [x] **2.1.** Implement `NormalizedModelWrapper(nn.Module)` in `src/models.py` so input tensors $x$ remain strictly in $[0.0, 1.0]$ and $(x - \mu)/\sigma$ normalization happens inside `forward()`.
- [x] **2.2.** Implement loader for **`CNN-Clean`**: Load pretrained CIFAR-10 ResNet (`cifar10_resnet20` via `torch.hub.load("chenyaofo/pytorch-cifar-models", ...)`).
- [x] **2.3.** Implement loader for **`ViT-Clean`**: Load pretrained CIFAR-10 Vision Transformer (`aaraki/vit-base-patch16-224-in21k-finetuned-cifar10` or `timm` small ViT with bilinear upsampling inside wrapper).
- [x] **2.4.** Write a quick sanity test verifying both models output shape `(batch_size, 10)` on a dummy $[0, 1]$ tensor of shape `(4, 3, 32, 32)`. _(verified: `python -m src.models` passes in a CPU Docker container)_
- [x] **2.5. [AUTO-PUSH TRIGGER]:** Update `TASK.md` checkboxes `[x]` and run:
  ```bash
  ./scripts/auto_push.sh "feat(models): implement NormalizedModelWrapper and pretrained CIFAR-10 CNN & ViT loaders [Milestone 2]"
  ```

---

## Milestone 3: Native PyTorch FGSM & PGD Attacks (`src/attacks.py`)
- [x] **3.1.** Implement `fgsm_attack(model, x, y, eps)` in `src/attacks.py` following Goodfellow et al. (2015):
  $$x_{\text{adv}} = \text{clip}_{[0, 1]}\big(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))\big)$$
- [x] **3.2.** Implement `pgd_attack(model, x, y, eps, alpha, steps=10, random_start=True)` in `src/attacks.py` following Madry et al. (2018) with $\ell_\infty$ projection to $[x-\epsilon, x+\epsilon]$ and clamping to $[0, 1]$.
- [x] **3.3.** Implement `rs_fgsm_attack(model, x, y, eps, alpha)` (Fast Adversarial Training step with random initialization in $[-\epsilon, \epsilon]$).
- [x] **3.4.** Verify via unit assertion that `torch.max(torch.abs(x_adv - x)) <= eps + 1e-6` and `0.0 <= x_adv.min() <= x_adv.max() <= 1.0`. _(verified: `python -m src.attacks` self-test passes in a CPU Docker container)_
- [x] **3.5. [AUTO-PUSH TRIGGER]:** Update `TASK.md` checkboxes `[x]` and run:
  ```bash
  ./scripts/auto_push.sh "feat(attacks): implement FGSM, PGD-10, and RS-FGSM L-inf attacks in PyTorch [Milestone 3]"
  ```

---

## Milestone 4: 10-Epoch Fast Adversarial Fine-Tuning Loop (`src/defense.py`)
- [x] **4.1.** Implement `adversarial_finetune()` using Automatic Mixed Precision (`torch.cuda.amp.autocast()` + `GradScaler`) to keep total training time between **20 and 35 minutes** on Colab T4 (strictly $< 4\text{ hours}$). _(implemented; the 20-35 min runtime is a Colab T4 estimate and has NOT been measured yet)_
- [x] **4.2.** Use combined loss $\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{CE}}(x_{\text{clean}}, y) + 0.5 \cdot \mathcal{L}_{\text{CE}}(x_{\text{adv}}, y)$ at $\epsilon = 8/255$ for 10 epochs so clean accuracy trade-off is controlled and measurable.
- [x] **4.3.** Add periodic checkpoint saving (`checkpoints/cnn_adv.pth`, `checkpoints/vit_adv.pth`, and `checkpoints/training_history.json`) after every epoch, plus an automatic Git commit/push hook every 5 epochs so progress is backed up to GitHub mid-training.
- [x] **4.4. [AUTO-PUSH TRIGGER]:** Update `TASK.md` checkboxes `[x]` and run:
  ```bash
  ./scripts/auto_push.sh "feat(defense): add 10-epoch AMP adversarial fine-tuning pipeline with periodic auto-save [Milestone 4]"
  ```

---

## Milestone 5: Evaluation Pipeline & Visualization Suite (`src/evaluate.py` & `src/visualize.py`)
- [x] **5.1.** Implement `evaluate_epsilon_sweep()` in `src/evaluate.py` across $\epsilon \in \{0, \frac{1}{255}, \frac{2}{255}, \frac{4}{255}, \frac{8}{255}, \frac{16}{255}\}$ for both FGSM and PGD-10 on a stratified subset of $N=1000$ CIFAR-10 test images (`SEED=42`).
- [x] **5.2.** Implement `evaluate_transferability()` between `CNN-Clean`, `ViT-Clean`, `CNN-AdvTrained`, and `ViT-AdvTrained` at $\epsilon = 8/255$.
- [x] **5.3.** Implement `src/visualize.py` to generate all 4 required 300-DPI figures in `outputs/`:
  1. `outputs/acc_vs_epsilon.png` (FGSM & PGD-10 Accuracy vs. $\epsilon$ curves)
  2. `outputs/clean_vs_robust_tradeoff.png` (Bar chart of Clean Accuracy cost vs. Robust Accuracy gain)
  3. `outputs/transferability_heatmap.png` (Cross-architecture transfer matrix)
  4. `outputs/adversarial_examples_grid.png` (Visual grid of Clean, Magnified Noise, and Adversarial images)
- [x] **5.4.** Create `main.py` CLI entry point supporting `--quick-test` (CPU 2-minute smoke test on 64 images) and `--full-run` (Colab GPU 10-epoch run).
- [x] **5.5. [AUTO-PUSH TRIGGER]:** Update `TASK.md` checkboxes `[x]` and run:
  ```bash
  ./scripts/auto_push.sh "feat(eval): add epsilon-sweep evaluation, transferability matrix, and 4 plot generators [Milestone 5]"
  ```

---

## Milestone 6: All-in-One Google Colab Notebook & Smoke Test Verification
- [x] **6.1.** Generate `Adversarial_CNN_vs_ViT_Colab.ipynb` containing the entire self-contained pipeline, including an optional cell at the top to configure a GitHub Personal Access Token (`GITHUB_TOKEN`) so Colab automatically pushes checkpoints, CSVs, and PNG plots back to the GitHub repo after each stage finishes.
- [x] **6.2.** Run `python main.py --quick-test` locally to verify zero syntax/runtime errors and generate initial baseline artifacts in `outputs/`. _(verified in a CPU Docker container, not natively: torch is blocked by Windows Application Control on this laptop. Ran in ~3-4 min; outputs/ now holds QUICK-TEST artifacts, watermarked "not a result", to be overwritten by the Milestone 7 GPU run.)_
- [x] **6.3.** Update `README.md` with embedded links to `outputs/acc_vs_epsilon.png`, `outputs/clean_vs_robust_tradeoff.png`, `outputs/transferability_heatmap.png`, and `outputs/adversarial_examples_grid.png`, along with the research findings summary.
- [x] **6.4. [AUTO-PUSH TRIGGER]:** Update `TASK.md` checkboxes `[x]` and run:
  ```bash
  ./scripts/auto_push.sh "feat(colab): add self-contained Colab notebook, verified smoke test outputs, and README [Milestone 6]"
  ```

---

## Milestone 7: Full GPU Execution & Final Results Auto-Push (Colab / Kaggle Run)
- [x] **7.1.** Execute 10-epoch Adversarial Fine-Tuning for `CNN-AdvTrained` $\to$ auto-push `checkpoints/training_history.json` to GitHub.
- [x] **7.2.** Execute 10-epoch Adversarial Fine-Tuning for `ViT-AdvTrained` $\to$ auto-push `checkpoints/training_history.json` to GitHub. _(done in Colab T4, RS-FGSM for the ViT per spec option; see README: the ViT shows signs of catastrophic overfitting, a PGD-5 re-run is recommended for a fair CNN-vs-ViT comparison. Results were downloaded and committed manually, no GITHUB_TOKEN was used.)_
- [x] **7.3.** Run full $N=1000$ FGSM & PGD-10 $\epsilon$-sweep and export `outputs/results_summary.csv` and `outputs/results_summary.md`.
- [x] **7.4.** Generate final high-resolution plots in `outputs/*.png`.
- [x] **7.5. [FINAL AUTO-PUSH TRIGGER]:** Mark all tasks `[x]` in `TASK.md` and run:
  ```bash
  ./scripts/auto_push.sh "results(final): push complete CIFAR-10 CNN vs ViT robustness metrics and 300-DPI plots [Milestone 7]"
  ```