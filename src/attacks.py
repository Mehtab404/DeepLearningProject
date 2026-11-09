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
import torch
import torch.nn.functional as F


def _input_grad(model, x, y, amp=False):
    """Gradient of the summed cross-entropy loss w.r.t. the input x."""
    x = x.clone().detach().requires_grad_(True)
    use_amp = bool(amp) and x.is_cuda
    with torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=use_amp):
        logits = model(x)
    loss = F.cross_entropy(logits.float(), y, reduction="sum")
    grad = torch.autograd.grad(loss, x)[0]
    return torch.nan_to_num(grad.detach())


def fgsm_attack(model, x, y, eps, amp=False):
    """x_adv = clip_[0,1]( x + eps * sign( grad_x L(theta, x, y) ) )   [Goodfellow et al., 2015]"""
    if eps == 0:
        return x.clone().detach()
    grad = _input_grad(model, x, y, amp)
    x_adv = x.detach() + eps * grad.sign()
    return torch.clamp(x_adv, 0.0, 1.0).detach()


def pgd_attack(model, x, y, eps, alpha=None, steps=10, random_start=True, amp=False):
    """L-inf PGD with projection onto the eps-ball around x and the [0, 1] box   [Madry et al., 2018]

    x^0     = clip_[0,1]( x + U(-eps, eps) )   (or x if random_start=False)
    x^{t+1} = clip_[0,1]( x + clip_[-eps,eps]( x^t + alpha * sign(grad) - x ) )
    Default step size: alpha = max(eps / 4, 1/255).
    """
    if eps == 0:
        return x.clone().detach()
    if alpha is None:
        alpha = max(eps / 4.0, 1.0 / 255.0)
    x = x.detach()
    if random_start:
        x_adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)
    else:
        x_adv = x.clone()
    for _ in range(steps):
        grad = _input_grad(model, x_adv, y, amp)
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)      # project onto the L-inf ball
        x_adv = torch.clamp(x + delta, 0.0, 1.0)       # project onto the valid image box
    return x_adv.detach()


def rs_fgsm_attack(model, x, y, eps, alpha=None, amp=False):
    """Random-start FGSM ("Fast AT", Wong et al., 2020): U(-eps, eps) init + one step of size alpha.

    Default alpha = 1.25 * eps, as recommended in the paper.
    """
    if eps == 0:
        return x.clone().detach()
    if alpha is None:
        alpha = 1.25 * eps
    x = x.detach()
    x0 = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)
    grad = _input_grad(model, x0, y, amp)
    delta = torch.clamp(x0 + alpha * grad.sign() - x, -eps, eps)
    return torch.clamp(x + delta, 0.0, 1.0).detach()


def check_perturbation(x, x_adv, eps, tol=1e-6):
    """Assert the L-inf budget and the [0, 1] pixel range (task 3.4)."""
    linf = torch.max(torch.abs(x_adv - x)).item()
    assert linf <= eps + tol, f"L-inf violated: {linf:.8f} > eps {eps:.8f}"
    assert 0.0 <= x_adv.min().item() and x_adv.max().item() <= 1.0, "x_adv left the [0, 1] range"
    return linf


def self_test(device=None, n=8):
    """Unit assertions on a tiny random CNN. Run with: python -m src.attacks"""
    import torch.nn as nn
    device = device or torch.device("cpu")
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(8, 10)).to(device).eval()
    x = torch.rand(n, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (n,), device=device)
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
    assert all(p.grad is None for p in model.parameters()), "attacks must not accumulate parameter grads"
    print("[attacks] self-test passed")


if __name__ == "__main__":
    self_test()
