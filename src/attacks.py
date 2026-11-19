"""Native PyTorch L-infinity attacks: FGSM, PGD and RS-FGSM.

References
  [1] Goodfellow et al., Explaining and Harnessing Adversarial Examples, ICLR 2015.
  [2] Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks, ICLR 2018.
  [3] Wong et al., Fast is better than free, ICLR 2020 (RS-FGSM).

Conventions
  * x is always in [0, 1]; normalisation happens inside the model's forward().
  * Attacks never change model.train()/eval() mode. The caller decides (use model.eval()).
  * Gradients are taken with torch.autograd.grad, so no .grad is accumulated on model parameters.
  * amp=True runs the forward/backward under autocast (CUDA only). The loss is summed and computed in
    fp32, which keeps fp16 input-gradients from underflowing to 0 (sign() is scale invariant).
"""
# ---------------------------------------------------------------------------------------------------
# BIG PICTURE
# An adversarial example is an image x_adv that looks almost identical to a real image x but makes the
# model predict the wrong class. "L-infinity" means we limit the LARGEST change to any single pixel:
#     max |x_adv - x| <= eps        (eps = 8/255 means no pixel may move by more than 8 brightness levels)
# Normal training changes the WEIGHTS to DECREASE the loss.
# An attack keeps the weights fixed and changes the INPUT IMAGE to INCREASE the loss.
# ---------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F


def _input_grad(model, x, y, amp=False):
    """Gradient of the summed cross-entropy loss w.r.t. the input x."""
    # This helper answers: "in which direction should each pixel move to make the loss bigger?"
    # It is shared by all three attacks below.

    # Fresh leaf tensor so we get d(loss)/d(input) and not d(loss)/d(weights).
    # clone() copies the data, detach() cuts it off from any earlier computation graph,
    # and requires_grad_(True) tells PyTorch to track gradients for the IMAGE.
    x = x.clone().detach().requires_grad_(True)

    # Mixed precision (fp16) is only used on a GPU, where it makes training faster.
    use_amp = bool(amp) and x.is_cuda
    with torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=use_amp):
        logits = model(x)    # raw class scores, shape (batch, 10)

    # Cast logits to fp32 before the loss so fp16 gradients cannot underflow to 0.
    # "sum" (not "mean") keeps each image's gradient independent of the batch size.
    loss = F.cross_entropy(logits.float(), y, reduction="sum")

    # torch.autograd.grad returns d(loss)/d(x) directly. Unlike loss.backward(), it does NOT store
    # gradients in the model's parameters (.grad), so the attack never interferes with training.
    grad = torch.autograd.grad(loss, x)[0]

    # nan_to_num: a NaN gradient would make sign() NaN and silently corrupt the image.
    return torch.nan_to_num(grad.detach())


def fgsm_attack(model, x, y, eps, amp=False):
    """x_adv = clip_[0,1]( x + eps * sign( grad_x L(theta, x, y) ) )   [Goodfellow et al., 2015]"""
    # FGSM = Fast Gradient Sign Method. A single, cheap, one-step attack.

    # eps = 0 means "no attack": return an unchanged copy (this gives the clean accuracy).
    if eps == 0:
        return x.clone().detach()
    grad = _input_grad(model, x, y, amp)

    # One step of size eps in the direction that INCREASES the loss (sign = L-inf steepest ascent).
    # sign() turns every gradient value into -1, 0 or +1, so every pixel moves by exactly eps.
    # That is the biggest change allowed under the L-infinity limit.
    x_adv = x.detach() + eps * grad.sign()

    # Pixel values must stay valid (between 0 and 1), so clip them.
    return torch.clamp(x_adv, 0.0, 1.0).detach()


def pgd_attack(model, x, y, eps, alpha=None, steps=10, random_start=True, amp=False):
    """L-inf PGD with projection onto the eps-ball around x and the [0, 1] box   [Madry et al., 2018]

    x^0     = clip_[0,1]( x + U(-eps, eps) )   (or x if random_start=False)
    x^{t+1} = clip_[0,1]( x + clip_[-eps,eps]( x^t + alpha * sign(grad) - x ) )
    Default step size: alpha = max(eps / 4, 1/255).
    """
    # PGD = Projected Gradient Descent. It is "FGSM repeated many times with small steps".
    # It is much stronger than FGSM because it can follow the loss surface around curves,
    # so it is the standard attack for measuring real robustness.

    if eps == 0:
        return x.clone().detach()

    # alpha = size of ONE small step. It is smaller than eps so that several steps are needed.
    if alpha is None:
        alpha = max(eps / 4.0, 1.0 / 255.0)
    x = x.detach()    # the original clean image; we never modify it

    # Random start: begin at a random point inside the allowed box around x, not at x itself.
    # uniform_(-eps, eps) gives random noise in [-eps, eps] for every pixel.
    if random_start:
        x_adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)
    else:
        x_adv = x.clone()

    # Iterate: small gradient-sign step, then project back into the allowed set (the two clamps below).
    for _ in range(steps):
        grad = _input_grad(model, x_adv, y, amp)    # gradient at the CURRENT adversarial image
        x_adv = x_adv + alpha * grad.sign()         # take a small step uphill on the loss
        # "Projection" = pull the image back if it went outside the allowed region.
        delta = torch.clamp(x_adv - x, -eps, eps)      # project onto the L-inf ball
        x_adv = torch.clamp(x + delta, 0.0, 1.0)       # project onto the valid image box
    return x_adv.detach()


def rs_fgsm_attack(model, x, y, eps, alpha=None, amp=False):
    """Random-start FGSM ("Fast AT", Wong et al., 2020): U(-eps, eps) init + one step of size alpha.

    Default alpha = 1.25 * eps, as recommended in the paper.
    """
    # RS-FGSM is used for TRAINING only (not for evaluation). It costs one gradient like FGSM,
    # but the random start makes the training much more effective. It is about 2.5x cheaper than PGD-5.

    if eps == 0:
        return x.clone().detach()
    if alpha is None:
        alpha = 1.25 * eps    # step slightly bigger than eps (paper's recommendation)
    x = x.detach()

    # Random start avoids the "catastrophic overfitting" of plain FGSM training: the model cannot
    # just learn to mask the gradient at exactly x.
    x0 = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)
    grad = _input_grad(model, x0, y, amp)

    # One step from the random point, then keep the total change inside [-eps, eps].
    delta = torch.clamp(x0 + alpha * grad.sign() - x, -eps, eps)
    return torch.clamp(x + delta, 0.0, 1.0).detach()


def check_perturbation(x, x_adv, eps, tol=1e-6):
    """Assert the L-inf budget and the [0, 1] pixel range (task 3.4)."""
    # Safety check used in tests: proves an attack did not "cheat" by changing pixels too much.
    linf = torch.max(torch.abs(x_adv - x)).item()    # the largest change of any single pixel
    # tol allows tiny floating-point rounding errors.
    assert linf <= eps + tol, f"L-inf violated: {linf:.8f} > eps {eps:.8f}"
    assert 0.0 <= x_adv.min().item() and x_adv.max().item() <= 1.0, "x_adv left the [0, 1] range"
    return linf


def self_test(device=None, n=8):
    """Unit assertions on a tiny random CNN. Run with: python -m src.attacks"""
    # Quick automatic test that needs no dataset and no GPU.
    import torch.nn as nn
    device = device or torch.device("cpu")
    torch.manual_seed(0)    # fixed seed -> same random numbers every run

    # A tiny untrained CNN: conv -> ReLU -> average pool -> flatten -> linear layer with 10 outputs.
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(8, 10)).to(device).eval()

    # n random "images" of size 3x32x32 (like CIFAR-10) and random labels 0..9.
    x = torch.rand(n, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (n,), device=device)

    # Try every attack at several budgets and check that the L-inf and [0, 1] rules hold.
    for eps in (0.0, 1 / 255, 4 / 255, 8 / 255, 16 / 255):
        for name, fn in (("fgsm", lambda: fgsm_attack(model, x, y, eps)),
                         ("pgd10", lambda: pgd_attack(model, x, y, eps, steps=10)),
                         ("pgd-no-rs", lambda: pgd_attack(model, x, y, eps, steps=3, random_start=False)),
                         ("rs_fgsm", lambda: rs_fgsm_attack(model, x, y, eps))):
            x_adv = fn()
            linf = check_perturbation(x, x_adv, eps)
            if eps == 0:
                assert linf == 0.0, f"{name}: eps=0 must return x unchanged"
        print(f"[attacks] eps={eps * 255:.0f}/255: fgsm / pgd / rs_fgsm all satisfy L-inf <= eps and [0, 1]")

    # The attacks use autograd.grad, so the model's weights must have no stored gradients.
    assert all(p.grad is None for p in model.parameters()), "attacks must not accumulate parameter grads"
    print("[attacks] self-test passed")


# This block only runs when the file is executed directly (python -m src.attacks), not when imported.
if __name__ == "__main__":
    self_test()
