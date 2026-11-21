"""Attacks: small tricks that change an image a tiny bit so the model gets it wrong."""
import torch  # main library for deep learning
import torch.nn.functional as F  # ready-made maths functions (we use it for the loss)


def _input_grad(model, x, y, amp=False):  # finds which way to change each pixel so the model becomes MORE wrong
    x = x.clone().detach().requires_grad_(True)  # copy the image and tell PyTorch we want the push direction for the IMAGE (not the weights)
    use_amp = bool(amp) and x.is_cuda  # faster half-precision maths only if asked AND a GPU is used
    with torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=use_amp):  # switch on the faster maths when allowed
        logits = model(x)  # model gives 10 scores per image (one score for each class)
    loss = F.cross_entropy(logits.float(), y, reduction="sum")  # loss = how wrong the model is; "sum" adds it over all images
    grad = torch.autograd.grad(loss, x)[0]  # for every pixel: does raising it make the loss go up or down?
    return torch.nan_to_num(grad.detach())  # replace any broken (NaN) numbers with 0 so they cannot ruin the image


def fgsm_attack(model, x, y, eps, amp=False):  # FGSM attack: ONE quick step that pushes every pixel to make the model wrong
    if eps == 0:  # eps = biggest change allowed per pixel; 0 means "do not attack"
        return x.clone().detach()  # give back an untouched copy of the image
    grad = _input_grad(model, x, y, amp)  # get the direction that makes the model more wrong
    x_adv = x.detach() + eps * grad.sign()  # move every pixel by exactly +eps or -eps (sign keeps only the + or - direction)
    return torch.clamp(x_adv, 0.0, 1.0).detach()  # keep pixel values between 0 (black) and 1 (white)


def pgd_attack(model, x, y, eps, alpha=None, steps=10, random_start=True, amp=False):  # PGD attack: many small steps, so much stronger than FGSM
    if eps == 0:  # no change allowed
        return x.clone().detach()  # return the image as it is
    if alpha is None:  # alpha = size of ONE small step
        alpha = max(eps / 4.0, 1.0 / 255.0)  # default step = a quarter of eps (but never below 1/255)
    x = x.detach()  # the original image; it is never changed
    if random_start:  # start from a slightly shaken image instead of the exact original
        x_adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)  # add random noise between -eps and +eps, keep pixels in 0..1
    else:  # no random start
        x_adv = x.clone()  # start from an exact copy of the original
    for _ in range(steps):  # repeat the small step several times
        grad = _input_grad(model, x_adv, y, amp)  # direction that makes the model more wrong, for the current tricked image
        x_adv = x_adv + alpha * grad.sign()  # take one small step in that direction
        delta = torch.clamp(x_adv - x, -eps, eps)  # total change so far, cut off so no pixel moved more than eps
        x_adv = torch.clamp(x + delta, 0.0, 1.0)  # original + allowed change, pixels kept between 0 and 1
    return x_adv.detach()  # the final tricked image


def rs_fgsm_attack(model, x, y, eps, alpha=None, amp=False):  # RS-FGSM: random start + ONE step; fast, used while TRAINING the defence
    if eps == 0:  # no change allowed
        return x.clone().detach()  # return the image as it is
    if alpha is None:  # alpha = size of the one step
        alpha = 1.25 * eps  # default step is a bit bigger than eps (the paper's advice)
    x = x.detach()  # the original image; it is never changed
    x0 = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)  # random start: add random noise, keep pixels in 0..1
    grad = _input_grad(model, x0, y, amp)  # direction that makes the model more wrong, from the random start
    delta = torch.clamp(x0 + alpha * grad.sign() - x, -eps, eps)  # one step, then cut the total change to within +-eps
    return torch.clamp(x + delta, 0.0, 1.0).detach()  # original + allowed change, pixels kept between 0 and 1


def check_perturbation(x, x_adv, eps, tol=1e-6):  # safety check: did the attack stay within the rules?
    linf = torch.max(torch.abs(x_adv - x)).item()  # the biggest change made to any single pixel
    assert linf <= eps + tol, f"L-inf violated: {linf:.8f} > eps {eps:.8f}"  # stop with an error if some pixel changed more than eps (tol = tiny rounding allowance)
    assert 0.0 <= x_adv.min().item() and x_adv.max().item() <= 1.0, "x_adv left the [0, 1] range"  # stop with an error if any pixel is outside 0..1
    return linf  # report the biggest change


def self_test(device=None, n=8):  # quick built-in test with fake images; run it with: python -m src.attacks
    import torch.nn as nn  # building blocks for a tiny model
    device = device or torch.device("cpu")  # use the CPU if no device was given
    torch.manual_seed(0)  # fixed starting number so the random numbers are the same every time
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),  # tiny test model: look at the image, keep positive signals, average them
                          nn.Flatten(), nn.Linear(8, 10)).to(device).eval()  # then flatten and give 10 class scores; eval = test mode
    x = torch.rand(n, 3, 32, 32, device=device)  # n fake colour images of size 32x32
    y = torch.randint(0, 10, (n,), device=device)  # n fake labels between 0 and 9
    for eps in (0.0, 1 / 255, 4 / 255, 8 / 255, 16 / 255):  # try several attack strengths
        for name, fn in (("fgsm", lambda: fgsm_attack(model, x, y, eps)),  # test the FGSM attack
                         ("pgd10", lambda: pgd_attack(model, x, y, eps, steps=10)),  # test PGD with 10 steps
                         ("pgd-no-rs", lambda: pgd_attack(model, x, y, eps, steps=3, random_start=False)),  # test PGD with 3 steps and no random start
                         ("rs_fgsm", lambda: rs_fgsm_attack(model, x, y, eps))):  # test RS-FGSM
            x_adv = fn()  # run the attack
            linf = check_perturbation(x, x_adv, eps)  # make sure it followed the rules
            if eps == 0:  # with eps = 0 nothing should change
                assert linf == 0.0, f"{name}: eps=0 must return x unchanged"  # stop with an error if the image changed
        print(f"[attacks] eps={eps * 255:.0f}/255: fgsm / pgd / rs_fgsm all satisfy L-inf <= eps and [0, 1]")  # tell the user this strength passed
    assert all(p.grad is None for p in model.parameters()), "attacks must not accumulate parameter grads"  # attacks must not leave any training information inside the model
    print("[attacks] self-test passed")  # all checks passed


if __name__ == "__main__":  # true only when this file is run directly
    self_test()  # run the quick test
