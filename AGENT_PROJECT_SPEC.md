# AI Agent Execution Spec: Adversarial Attacks & Defences (CNN vs. ViT Robustness)

> **MANDATORY DIRECTIVE FOR THE AI CODING AGENT:**
>
> Follow this specification using **pretrained clean models** and a **10-epoch Fast Adversarial Fine-Tuning loop** designed for Google Colab / Kaggle.
> * **Strict Training Time Budget:** Total runtime MUST finish in **~20 to 45 minutes** on a free Google Colab T4 GPU (and NEVER exceed 4–5 hours under any circumstance). Do NOT train for 30+ hours or run 100-epoch from-scratch training loops.
> * **Train Only What is Strictly Necessary:** Load pre-trained CIFAR-10 weights for the clean baselines (`CNN-Clean` and `ViT-Clean`), and run a **10-epoch Adversarial Fine-Tuning loop** (using Mixed Precision `torch.cuda.amp` and PGD-5 or Random-Start FGSM) on those exact same models to create `CNN-AdvTrained` and `ViT-AdvTrained`. This gives an exact, scientifically valid measurement of the **cost of adversarial training on clean accuracy**.

---

## 1. Project Context & Academic Requirements

* **Project Title:** Adversarial Attacks and Defences: CNN vs Vision Transformer Robustness
* **Domain / Type:** Computer Vision / Image Classification
* **Problem Statement:** Small, invisible pixel changes can fool classifiers. Compare how robust Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) are against gradient-based adversarial perturbations.
* **Dataset:** `CIFAR-10` via `torchvision.datasets.CIFAR10` ($32 \times 32$ RGB images, 10 classes).
* **Required Pipeline (Baseline $\to$ Advanced):**
  1. Load clean baseline **ResNet** and **small ViT** pretrained models on CIFAR-10 (with optional 2–3 epoch clean head warmup if adapting from ImageNet).
  2. Implement **FGSM** (Fast Gradient Sign Method) and **PGD** (Projected Gradient Descent) $\ell_\infty$ attacks from scratch in PyTorch.
  3. Execute a **10-Epoch Adversarial Fine-Tuning Defence** on both models using Mixed Precision (`torch.amp`) and save `.pth` checkpoints after every epoch.
  4. Generate **Accuracy vs. Epsilon ($\epsilon$) curves** and analyze the **Research Angle**: *Robustness comparison across architectures and the cost of adversarial training on clean accuracy.*
* **Core Reference Papers:**
  * `[1]` Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015 — https://arxiv.org/abs/1412.6572
  * `[2]` Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018 — https://arxiv.org/abs/1706.06083

---

## 2. CRITICAL Hardware, Time & Training Constraints (Must Follow)

1. **No Local GPU Available:** The user's laptop **does not have a dedicated GPU**.
2. **Primary Execution Target:** Free **Google Colab (1x NVIDIA T4 GPU)** or **Kaggle Notebooks** (plus a CPU quick-test fallback mode `--quick-test` using 200 images so the user can verify code locally).
3. **Strictly Necessary Training Strategy (~20–40 Minutes Total):**
   * **Phase 1 — Clean Baselines (0 to 5 mins):**
     * **`CNN-Clean`**: Load pretrained CIFAR-10 ResNet directly via `torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)` (or `cifar10_resnet32`). Zero clean training needed!
     * **`ViT-Clean`**: Load a pretrained CIFAR-10 Vision Transformer from HuggingFace (`aaraki/vit-base-patch16-224-in21k-finetuned-cifar10` or `nateraw/vit-base-patch16-224-cifar10`) OR load `vit_tiny_patch16_224` from `timm` and fine-tune cleanly for just **5 epochs** (~4 mins on T4 GPU) on CIFAR-10.
   * **Phase 2 — Adversarial Training Defence (10 Epochs Only, ~20–30 mins total):**
     * Clone `CNN-Clean` $\to$ fine-tune for **10 epochs** using Adversarial Training ($\epsilon = 8/255$) to produce **`CNN-AdvTrained`**.
     * Clone `ViT-Clean` $\to$ fine-tune for **10 epochs** using Adversarial Training ($\epsilon = 8/255$) to produce **`ViT-AdvTrained`**.
     * Use **Automatic Mixed Precision (`torch.cuda.amp.autocast()`)** during both attack generation and weight updates to speed up training by **2.5x**.
     * Default training attack: **5-step PGD (`PGD-5`, $\alpha = 2/255$, $\epsilon = 8/255$)** or **Fast Adversarial Training (Random-Start FGSM, Wong et al., 2020 with $\alpha = 10/255, \epsilon = 8/255$)** so all 10 epochs complete in ~15–25 minutes.
   * **Phase 3 — Checkpoint Persistence:**
     * Automatically save `cnn_clean.pth`, `vit_clean.pth`, `cnn_adv.pth`, and `vit_adv.pth` inside a `checkpoints/` folder (and support Google Drive mounting in Colab) so the user never has to re-train if the session disconnects.
4. **Fast Evaluation Subset (`N = 1000`):**
   * Evaluate the full $\epsilon$-sweep on a fixed stratified subset of **`N = 1000` test images** (`SEED = 42`) so PGD-10/PGD-20 evaluation across 6 $\epsilon$ values finishes in **under 3 minutes**.

---

## 3. Architecture & Normalization Specifications

### 3.1. The Golden Rule of Input Normalization

* **NEVER** normalize images inside `torchvision.transforms` before passing them to the attack functions.
* Keep all DataLoader image tensors strictly in the pixel range $x \in [0.0, 1.0]$ using only `transforms.ToTensor()`.
* Wrap every model in a custom `NormalizedModelWrapper(nn.Module)` that performs $(x - \mu) / \sigma$ (and bilinear interpolation if the ViT expects $224 \times 224$) **inside the `forward()` pass**.
* **Why:** This guarantees that $\epsilon \in \{0, \frac{1}{255}, \frac{2}{255}, \frac{4}{255}, \frac{8}{255}, \frac{16}{255}\}$ and `torch.clamp(x_adv, 0.0, 1.0)` are mathematically exact in $[0, 1]$ pixel space across both CNN and ViT models.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedModelWrapper(nn.Module):
    def __init__(self, model, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616), img_size=None):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        if self.img_size is not None and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x_norm = (x - self.mean) / self.std
        out = self.model(x_norm)
        return out.logits if hasattr(out, 'logits') else out
```

### 3.2. Summary of the 4 Models to Compare

| Model ID | Base Architecture | How Weights Are Obtained | Est. Time on Colab T4 |
| :--- | :--- | :--- | :--- |
| **`CNN-Clean`** | ResNet-20 / ResNet-18 (CIFAR-10) | Loaded directly from pretrained CIFAR-10 TorchHub checkpoint | **< 1 min** |
| **`ViT-Clean`** | ViT-Tiny / ViT-Base (CIFAR-10) | Loaded from HuggingFace CIFAR-10 pretrained checkpoint (or 5-epoch `timm` fine-tune) | **1 – 4 mins** |
| **`CNN-AdvTrained`** | Same as `CNN-Clean` | **10-epoch Adversarial Fine-Tuning** ($\epsilon=8/255$, mixed precision) starting from `CNN-Clean` | **~8 – 12 mins** |
| **`ViT-AdvTrained`** | Same as `ViT-Clean` | **10-epoch Adversarial Fine-Tuning** ($\epsilon=8/255$, mixed precision) starting from `ViT-Clean` | **~12 – 18 mins** |

---

## 4. Mathematical Specifications for Attacks & Defences

### 4.1. Fast Gradient Sign Method (FGSM — Goodfellow et al., 2015)

Implement `fgsm_attack(model, x, y, eps)`:

$$
x_{\text{adv}} = \text{clip}_{[0, 1]}\Big(x + \epsilon \cdot \text{sign}\big(\nabla_x \mathcal{L}(\theta, x, y)\big)\Big)
$$

* When $\epsilon = 0$, return $x$ unchanged without computing gradients.

### 4.2. Projected Gradient Descent (PGD — Madry et al., 2018)

Implement `pgd_attack(model, x, y, eps, alpha, steps=10, random_start=True)`:

1. Initialize $x^0 = \text{clip}_{[0, 1]}(x + \mathcal{U}(-\epsilon, \epsilon))$ if `random_start=True`, else $x^0 = x$.
2. For $t = 0 \dots K-1$:

$$
x^{t+1} = \text{clip}_{[0, 1]}\Big( x + \text{clip}_{[-\epsilon, \epsilon]}\big( x^t + \alpha \cdot \text{sign}(\nabla_{x^t} \mathcal{L}(\theta, x^t, y)) - x \big) \Big)
$$

* Default step size rule: $\alpha = \max(\epsilon / 4, 1/255)$ and $K = 10$ for evaluation.

### 4.3. 10-Epoch Fast Adversarial Fine-Tuning Loop

Implement `adversarial_finetune(model, train_loader, epochs=10, eps=8/255, method="pgd5" | "rs_fgsm", lr=1e-4)`:

* Use `torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)` and `CosineAnnealingLR(optimizer, T_max=epochs)`.
* Use `scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())`.
* For each batch $(x, y)$:
  1. Set `model.eval()` and generate $x_{\text{adv}}$ inside `torch.cuda.amp.autocast()` using either:
     * **RS-FGSM (Fastest, 1 step):** Random start in $[-\epsilon, \epsilon]$ + single step $\alpha = 1.25\epsilon$, projected back to $[-\epsilon, \epsilon]$ and $[0, 1]$.
     * **PGD-5 (Strongest, 5 steps):** 5-step PGD with $\alpha = 2/255, \epsilon = 8/255$.
  2. Set `model.train()`, zero gradients, compute combined trade-off loss:
     $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{CE}}(\theta, x_{\text{clean}}, y) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\theta, x_{\text{adv}}, y)$$
     *(Using a 50/50 clean + adversarial loss prevents catastrophic forgetting on pretrained weights during short 10-epoch fine-tuning while achieving strong robustness).*
  3. Update weights with `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, and `scaler.update()`.
  4. Print progress bar (`tqdm`) with Epoch, Train Loss, Clean Acc, and Robust Acc, and save the model checkpoint.

---

## 5. Required Experiments, Plots & Outputs

The script/notebook must automatically save all plots as high-resolution PNGs (`dpi=300`) in an `outputs/` directory and print a Markdown table of results:

### Deliverable 1: Accuracy vs. Epsilon ($\epsilon$) Curves (`outputs/acc_vs_epsilon.png`)

* Evaluate all 4 models (`CNN-Clean`, `ViT-Clean`, `CNN-AdvTrained`, `ViT-AdvTrained`) across:

$$
\epsilon \in \left\{0, \frac{1}{255}, \frac{2}{255}, \frac{4}{255}, \frac{8}{255}, \frac{16}{255}\right\}
$$

* Create a **1x2 subplot figure**:
  * **Left Subplot:** Accuracy (%) vs. $\epsilon$ (in `/255` units) under **FGSM Attack**.
  * **Right Subplot:** Accuracy (%) vs. $\epsilon$ (in `/255` units) under **PGD-10 Attack**.
* Use distinct colors for CNN vs. ViT and solid vs. dashed lines for Clean vs. AdvTrained.

### Deliverable 2: Cost of Adversarial Training on Clean Accuracy (`outputs/clean_vs_robust_tradeoff.png`)

* A grouped bar chart comparing each architecture Before vs. After 10-Epoch Adversarial Fine-Tuning on:
  1. **Clean Accuracy ($\epsilon = 0$)**
  2. **FGSM Robust Accuracy ($\epsilon = 8/255$)**
  3. **PGD-10 Robust Accuracy ($\epsilon = 8/255$)**
* Annotate the exact percentage drop in clean accuracy ($\Delta \text{Acc}_{\text{clean}}$) and gain in robust accuracy ($\Delta \text{Acc}_{\text{robust}}$) on the bars to directly answer the project's **Research Angle**.

### Deliverable 3: Cross-Architecture Transferability Matrix (`outputs/transferability_heatmap.png`)

* Generate PGD ($\epsilon = 8/255$) adversarial examples using `CNN-Clean` as the source model and evaluate on `ViT-Clean`, and vice-versa.
* Plot a $2 \times 2$ or $4 \times 4$ heatmap showing how well CNN perturbations transfer to ViTs and how well ViT perturbations transfer to CNNs.

### Deliverable 4: Qualitative Visualization of Perturbations (`outputs/adversarial_examples_grid.png`)

* For 4 sample CIFAR-10 test images, display:
  * Column 1: Clean Image ($x$) + True Label
  * Column 2: Magnified FGSM/PGD Perturbation ($\frac{x_{\text{adv}} - x + \epsilon}{2\epsilon}$)
  * Column 3: Adversarial Image ($x_{\text{adv}}$ at $\epsilon = 8/255$) + `CNN-Clean` vs. `ViT-Clean` and `CNN-AdvTrained` vs. `ViT-AdvTrained` predictions.

### Deliverable 5: Quantitative Summary Table (`outputs/results_summary.csv` & `.md`)

* Export a clean CSV and Markdown table containing all accuracy values across $\epsilon$ levels so the user can paste it directly into their final report and presentation.

---

## 6. Required Project Structure

Generate the project cleanly into modular Python files **AND** provide a single all-in-one `.ipynb` (Google Colab Notebook) so the user can upload one file to Google Colab and click **"Run All"**:

```text
adversarial-robustness-cnn-vs-vit/
├── Adversarial_CNN_vs_ViT_Colab.ipynb   # Self-contained 1-click Google Colab / Kaggle notebook
├── requirements.txt                     # torch, torchvision, timm, transformers, matplotlib, seaborn, tqdm, pandas
├── checkpoints/                         # Auto-saved .pth weights after 10-epoch adversarial fine-tuning
├── outputs/                             # Auto-generated 300-DPI plots and CSV/Markdown tables
├── src/
│   ├── models.py                        # NormalizedModelWrapper + Pretrained CIFAR-10 CNN & ViT loaders
│   ├── attacks.py                       # Native PyTorch FGSM and PGD implementations
│   ├── defense.py                       # 10-Epoch Fast Adversarial Fine-Tuning loop with AMP
│   ├── evaluate.py                      # Epsilon sweep + Cross-architecture transferability
│   └── visualize.py                     # Generates the 4 publication-quality plots
├── main.py                              # CLI entry point (--epochs 10 --eval-samples 1000)
└── README.md                            # Setup instructions, Colab guide, and research findings summary
```

---

## 7. Acceptance Criteria Checklist for the AI Agent

Before finishing your task, verify:

* [ ] Does the code use **pretrained CIFAR-10 clean models** and run **exactly 10 epochs of Fast Adversarial Fine-Tuning** (finishing in ~20–35 minutes on a Colab T4 GPU, well below the user's 4–5 hour cap)?
* [ ] Are checkpoints saved automatically so if `checkpoints/*.pth` already exist, the script loads them instead of retraining from scratch?
* [ ] Are `fgsm_attack` and `pgd_attack` written transparently in native PyTorch (satisfying academic grading for `[1] Goodfellow et al. (2015)` and `[2] Madry et al. (2018)`)?
* [ ] Are image tensors guaranteed to stay in $[0, 1]$ during attacks with normalization wrapped inside the model's `forward` pass?
* [ ] Are all 4 plots and the summary CSV/Markdown table automatically saved and displayed?