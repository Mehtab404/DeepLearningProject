"""Attacks: small tricks that change an image a tiny bit so the model gets it wrong."""
# WHAT: Imports PyTorch (tensors, autograd) and its functional API F, which every attack in this file needs.
# HOW: torch holds images as tensors and tracks operations so gradients can be computed; F.cross_entropy turns
#      model scores and true labels into one loss number.
# WHY: A gradient attack needs d(loss)/d(pixel). Only an autograd library can compute that for a deep network.
# PROS & CONS: + Standard, GPU-ready, no extra dependency. - Ties the attacks to PyTorch models (not
#              framework-agnostic).
# EXAMPLE: F.cross_entropy(logits of shape (4, 10), y of shape (4,)) returns one scalar loss.
import torch  # main library for deep learning
import torch.nn.functional as F  # ready-made maths functions (we use it for the loss)


# WHAT: Returns d(loss)/d(input image) for a whole batch: the per-pixel direction that makes the model MORE
#       wrong.
# HOW: Copy x, mark it as differentiable, run the model, compute cross-entropy summed over the batch, then
#      torch.autograd.grad(loss, x) returns a tensor shaped exactly like x. NaN values are replaced by 0.
# WHY: FGSM, PGD and RS-FGSM are all built from this one gradient, so a single tested routine keeps them
#      consistent.
# PROS & CONS: + One shared routine; autograd.grad leaves the weights' .grad buffers untouched. - nan_to_num
#              hides numerical problems instead of fixing them.
# EXAMPLE: x of shape (100, 3, 32, 32) gives a gradient of shape (100, 3, 32, 32); its .sign() contains only -1,
#          0 or +1.
def _input_grad(model, x, y, amp=False):  # finds which way to change each pixel so the model becomes MORE wrong
    # WHAT: Makes a private copy of the image batch that autograd will track.
    # HOW: clone() copies the data, detach() cuts any old graph, requires_grad_(True) tells autograd to record
    #      gradients for these pixels.
    # WHY: The original x must stay unchanged, and we want gradients for the IMAGE, not for the model weights.
    # PROS & CONS: + Cannot corrupt the caller's tensor. - One extra copy of the batch in memory.
    # EXAMPLE: x.requires_grad is False before this line and True after it (on the copy).
    x = x.clone().detach().requires_grad_(True)  # copy the image and tell PyTorch we want the push direction for the IMAGE (not the weights)
    # WHAT: Decides whether to use fast half-precision (fp16) maths for this forward pass.
    # HOW: fp16 is used only when amp=True AND the tensor lives on a CUDA GPU (x.is_cuda).
    # WHY: Training attacks run in fp16 for speed on the T4; evaluation attacks use full fp32 so the reported
    #      accuracy is not affected by fp16 gradient noise.
    # PROS & CONS: + Roughly 2x faster and half the memory in training. - fp16 gradients are slightly noisier,
    #              which is why evaluation passes amp=False.
    # EXAMPLE: Evaluation: _input_grad(model, x, y, amp=False) runs in fp32. Training: amp=True on a GPU.
    use_amp = bool(amp) and x.is_cuda  # faster half-precision maths only if asked AND a GPU is used
    # WHAT: Runs the model forward pass, optionally under automatic mixed precision.
    # HOW: torch.autocast switches eligible operations (convolutions, matmuls) to fp16 inside the block;
    #      enabled=False makes it a no-op.
    # WHY: Keeps one code path for CPU, fp32 GPU and fp16 GPU.
    # PROS & CONS: + No duplicated code. - The autocast context adds a small fixed overhead even when disabled.
    # EXAMPLE: logits = model(x) gives a (batch, 10) score table, one score per CIFAR-10 class.
    with torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=use_amp):  # switch on the faster maths when allowed
        logits = model(x)  # model gives 10 scores per image (one score for each class)
    # WHAT: Measures how wrong the model is on the batch.
    # HOW: Cross-entropy of the float32 scores against the true labels, reduction='sum' adds the per-image
    #      losses instead of averaging them.
    # WHY: With a mean, each image's gradient would be divided by the batch size, so the gradient scale would
    #      change with the batch size. Only the sign matters for the attacks, but a sum keeps magnitudes
    #      consistent.
    # PROS & CONS: + Per-image gradients independent of batch size. - Large raw loss values (not meant to be
    #              logged).
    # EXAMPLE: A batch of 100 images with average loss 2.3 gives loss of about 230.
    loss = F.cross_entropy(logits.float(), y, reduction="sum")  # loss = how wrong the model is; "sum" adds it over all images
    # WHAT: Computes the gradient of the loss with respect to every input pixel.
    # HOW: Backpropagation from the loss to x; returns a tuple, [0] picks the gradient for x.
    # WHY: autograd.grad (instead of loss.backward()) returns the gradient directly and does not write into
    #      model.parameters().grad.
    # PROS & CONS: + No side effects on the model. - Only one gradient is returned per call (fine here, we need
    #              just d/dx).
    # EXAMPLE: A pixel with gradient +0.8 would increase the loss if it became brighter.
    grad = torch.autograd.grad(loss, x)[0]  # for every pixel: does raising it make the loss go up or down?
    # WHAT: Returns the gradient with any NaN/Inf cleaned up and detached from the graph.
    # HOW: nan_to_num replaces NaN by 0 (and Inf by large finite numbers); detach() removes autograd history.
    # WHY: A single NaN would otherwise spread through sign() and ruin the adversarial image; fp16 can produce
    #      them.
    # PROS & CONS: + Robust against rare overflow. - A zero gradient means that pixel simply is not moved in
    #              that step.
    # EXAMPLE: tensor([nan, 0.3]) becomes tensor([0.0, 0.3]).
    return torch.nan_to_num(grad.detach())  # replace any broken (NaN) numbers with 0 so they cannot ruin the image


# WHAT: FGSM: the Fast Gradient Sign Method (Goodfellow et al., 2015), a ONE-step attack.
# HOW: x_adv = clip_[0,1]( x + eps * sign( grad_x L ) ). Every pixel moves exactly +eps or -eps in the direction
#      that raises the loss.
# WHY: Cheapest white-box attack; also the baseline against which the stronger PGD is compared.
# PROS & CONS: + Very fast (one forward and one backward pass). - Weak on adversarially trained models: the loss
#              surface is not linear over a large eps, so one big step can overshoot.
# EXAMPLE: eps = 8/255 = 0.0314: a pixel at 0.50 becomes 0.5314 or 0.4686.
def fgsm_attack(model, x, y, eps, amp=False):  # FGSM attack: ONE quick step that pushes every pixel to make the model wrong
    # WHAT: Shortcut for 'no attack allowed'.
    # HOW: With eps = 0 the L-infinity ball contains only the original image, so return an untouched copy.
    # WHY: Makes the eps = 0 point of the accuracy-vs-epsilon curve equal to ordinary clean accuracy.
    # PROS & CONS: + Exact and free. - None.
    # EXAMPLE: fgsm_attack(model, x, y, 0.0) returns a copy equal to x.
    if eps == 0:  # eps = biggest change allowed per pixel; 0 means "do not attack"
        return x.clone().detach()  # give back an untouched copy of the image
    # WHAT: Gets the direction that increases the loss.
    # HOW: One forward and one backward pass through the model.
    # WHY: FGSM needs only the SIGN of this gradient.
    # PROS & CONS: + Single gradient evaluation. - No iteration, so no refinement.
    # EXAMPLE: grad.shape == x.shape == (100, 3, 32, 32).
    grad = _input_grad(model, x, y, amp)  # get the direction that makes the model more wrong
    # WHAT: Takes the single FGSM step.
    # HOW: Add eps times the sign of the gradient to every pixel.
    # WHY: The sign makes every pixel change by the maximum allowed amount, which is the strongest single step
    #      inside the L-infinity ball.
    # PROS & CONS: + Uses the full budget. - Ignores gradient magnitude, so it can waste budget on pixels that
    #              barely matter.
    # EXAMPLE: grad = [-0.3, 0.0, 2.0] -> sign = [-1, 0, 1] -> change = [-eps, 0, +eps].
    x_adv = x.detach() + eps * grad.sign()  # move every pixel by exactly +eps or -eps (sign keeps only the + or - direction)
    # WHAT: Keeps the adversarial image a valid image.
    # HOW: clamp limits every pixel value to the range [0, 1].
    # WHY: Pixel values outside 0..1 are not real images and would break the claim that eps is measured in true
    #      pixel space.
    # PROS & CONS: + Guarantees valid pixels. - Pixels already at 0 or 1 cannot move one way, so the real change
    #              can be smaller than eps.
    # EXAMPLE: 0.99 + 0.0314 = 1.0214 is clamped to 1.0.
    return torch.clamp(x_adv, 0.0, 1.0).detach()  # keep pixel values between 0 (black) and 1 (white)


# WHAT: PGD: Projected Gradient Descent (Madry et al., 2018), a MULTI-step attack and the standard strong
#       L-infinity attack.
# HOW: Start at (optionally randomised) x, repeat `steps` times: x_adv += alpha * sign(grad); project the total
#      change back into [-eps, +eps]; clamp to [0, 1].
# WHY: Many small steps find much worse inputs than one big FGSM step; this is the main robustness test and the
#      attack used inside adversarial training.
# PROS & CONS: + Much stronger than FGSM; still simple and white-box. - Cost grows with `steps` (each step is a
#              forward plus backward pass); still not a guarantee (AutoAttack is stronger).
# EXAMPLE: PGD-10 at eps = 8/255 on 1000 images runs 10 forward+backward passes per image.
def pgd_attack(model, x, y, eps, alpha=None, steps=10, random_start=True, amp=False):  # PGD attack: many small steps, so much stronger than FGSM
    # WHAT: Same 'no attack' shortcut as in FGSM.
    # HOW: Return an unchanged copy when eps is 0.
    # WHY: Keeps the eps = 0 point equal to clean accuracy.
    # PROS & CONS: + Exact and free. - None.
    # EXAMPLE: pgd_attack(..., eps=0.0) returns x unchanged.
    if eps == 0:  # no change allowed
        return x.clone().detach()  # return the image as it is
    # WHAT: Chooses a default step size.
    # HOW: alpha = max(eps / 4, 1/255).
    # WHY: Steps must be small enough to follow the loss surface but large enough to reach the boundary in
    #      `steps` moves.
    # PROS & CONS: + Sensible default for 10 steps. - A fixed rule; with 50 steps a smaller alpha (eps / 10) is
    #              better, which strong_attack_check.py passes explicitly.
    # EXAMPLE: eps = 8/255 gives alpha = 2/255.
    if alpha is None:  # alpha = size of ONE small step
        alpha = max(eps / 4.0, 1.0 / 255.0)  # default step = a quarter of eps (but never below 1/255)
    # WHAT: Freezes the original image as the reference point.
    # HOW: detach() removes gradient history; x is never modified afterwards.
    # WHY: The projection step measures the total change relative to this original.
    # PROS & CONS: + Prevents accidental in-place edits. - None.
    # EXAMPLE: x_adv may move, x stays fixed.
    x = x.detach()  # the original image; it is never changed
    # WHAT: Chooses the starting point of the search.
    # HOW: With random_start, begin at x + uniform noise in [-eps, +eps], clamped to [0, 1]; otherwise begin
    #      exactly at x.
    # WHY: A random start avoids a deterministic path and finds stronger examples; it is part of Madry et al.'s
    #      PGD.
    # PROS & CONS: + Stronger, and restarts become possible (different seed gives a different path). - Results
    #              vary slightly with the random seed (42 vs 42.6 in the report).
    # EXAMPLE: A pixel at 0.50 with eps = 0.0314 may start anywhere in [0.4686, 0.5314].
    if random_start:  # start from a slightly shaken image instead of the exact original
        x_adv = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)  # add random noise between -eps and +eps, keep pixels in 0..1
    else:  # no random start
        x_adv = x.clone()  # start from an exact copy of the original
    # WHAT: The PGD iteration loop.
    # HOW: Each pass: gradient at the current x_adv, one signed step of size alpha, then the projection below.
    # WHY: More iterations let the attack climb the loss surface further.
    # PROS & CONS: + Adaptive: each step uses the gradient at the new point. - Linear cost in `steps`.
    # EXAMPLE: steps = 10 means 10 gradient evaluations.
    for _ in range(steps):  # repeat the small step several times
        # WHAT: Gradient at the CURRENT adversarial image (not at the original).
        # HOW: One forward + backward pass through the model for the batch.
        # WHY: The loss surface is curved; re-evaluating the gradient after each step is what makes PGD stronger
        #      than FGSM.
        # PROS & CONS: + Follows the curve. - One full model pass per step.
        # EXAMPLE: After step 1, x_adv differs from x, so the gradient also differs.
        grad = _input_grad(model, x_adv, y, amp)  # direction that makes the model more wrong, for the current tricked image
        # WHAT: Takes one small signed step that increases the loss.
        # HOW: Add alpha times the sign of the gradient to every pixel.
        # WHY: The sign makes the step an L-infinity step: each pixel moves by exactly alpha.
        # PROS & CONS: + Uses the budget evenly. - Ignores magnitude.
        # EXAMPLE: alpha = 2/255: a pixel moves by 0.00784.
        x_adv = x_adv + alpha * grad.sign()  # take one small step in that direction
        # WHAT: PROJECTION: forces the image back inside the allowed region.
        # HOW: delta = x_adv - x is the total change so far; clamp it to [-eps, +eps] per pixel, then x_adv =
        #      clamp(x + delta, 0, 1).
        # WHY: After several steps the total change could exceed eps. The attack is only valid if every pixel
        #      stays within eps of the original and within the valid pixel range.
        # PROS & CONS: + Guarantees ||x_adv - x||_inf <= eps and 0 <= x_adv <= 1 (checked by
        #              check_perturbation). - Pixels pushed against the boundary stop moving there.
        # EXAMPLE: A pixel at 0.5 with eps = 8/255 can never leave [0.4686, 0.5314], however many steps are
        #          taken.
        delta = torch.clamp(x_adv - x, -eps, eps)  # total change so far, cut off so no pixel moved more than eps
        x_adv = torch.clamp(x + delta, 0.0, 1.0)  # original + allowed change, pixels kept between 0 and 1
    # WHAT: Hands back the final adversarial images.
    # HOW: detach() removes autograd history so later code does not keep the graph alive.
    # WHY: Saves memory and avoids accidental gradient flow into the attack.
    # PROS & CONS: + Lower memory. - None.
    # EXAMPLE: The returned tensor has shape (batch, 3, 32, 32).
    return x_adv.detach()  # the final tricked image


# WHAT: RS-FGSM: random start plus one large FGSM step (Wong et al., 2020), used only to train defences quickly.
# HOW: x0 = x + U(-eps, eps); x_adv = clip( x + clamp( x0 + alpha * sign(grad at x0) - x, -eps, eps ) ), with
#      alpha = 1.25 * eps by default.
# WHY: About 2.5x cheaper than PGD-5 for training (30.1 vs 74.7 minutes for the ViT).
# PROS & CONS: + Fast. - Can cause 'catastrophic overfitting': robust to FGSM but not to PGD, which is exactly
#              what the report found (79.8% under FGSM, 4.9% under PGD-10).
# EXAMPLE: Report result: ViT trained with RS-FGSM scored 4.9% under PGD-10 at 8/255, PGD-5 training gave 42.2%.
def rs_fgsm_attack(model, x, y, eps, alpha=None, amp=False):  # RS-FGSM: random start + ONE step; fast, used while TRAINING the defence
    # WHAT: Same 'no attack' shortcut.
    # HOW: Return an unchanged copy when eps is 0.
    # WHY: Consistent behaviour across all three attacks.
    # PROS & CONS: + Exact and free. - None.
    # EXAMPLE: rs_fgsm_attack(..., eps=0.0) returns x unchanged.
    if eps == 0:  # no change allowed
        return x.clone().detach()  # return the image as it is
    # WHAT: Default step size for RS-FGSM.
    # HOW: alpha = 1.25 * eps.
    # WHY: Wong et al. recommend a step slightly larger than eps to compensate for the random start.
    # PROS & CONS: + Follows the published recipe. - Larger steps are what makes catastrophic overfitting more
    #              likely.
    # EXAMPLE: eps = 8/255 gives alpha = 10/255.
    if alpha is None:  # alpha = size of the one step
        alpha = 1.25 * eps  # default step is a bit bigger than eps (the paper's advice)
    # WHAT: Freezes the original image as the reference point.
    # HOW: detach() removes gradient history.
    # WHY: The final projection is measured against this original.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: x stays unchanged.
    x = x.detach()  # the original image; it is never changed
    # WHAT: Random starting point.
    # HOW: Add uniform noise in [-eps, +eps] and clamp to [0, 1].
    # WHY: The gradient is taken at a random nearby point, not exactly at x, so the model cannot exploit one
    #      fixed gradient direction.
    # PROS & CONS: + Cheap diversification. - Randomness makes training slightly noisy.
    # EXAMPLE: x0 differs from x by at most 8/255 per pixel.
    x0 = torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0.0, 1.0)  # random start: add random noise, keep pixels in 0..1
    # WHAT: Gradient at the random start.
    # HOW: One forward + backward pass.
    # WHY: RS-FGSM uses exactly ONE gradient (that is why it is cheap).
    # PROS & CONS: + One pass only. - No refinement.
    # EXAMPLE: grad.shape == (batch, 3, 32, 32).
    grad = _input_grad(model, x0, y, amp)  # direction that makes the model more wrong, from the random start
    # WHAT: The single large step plus projection.
    # HOW: Move from x0 by alpha along sign(grad), subtract x to get the total change, clamp it to [-eps, +eps].
    # WHY: Keeps the result inside the same L-infinity ball as PGD, so the attacks are comparable.
    # PROS & CONS: + Valid perturbation guaranteed. - Clipping can remove part of the big step.
    # EXAMPLE: |delta| <= 8/255 for every pixel.
    delta = torch.clamp(x0 + alpha * grad.sign() - x, -eps, eps)  # one step, then cut the total change to within +-eps
    # WHAT: Final adversarial image for RS-FGSM.
    # HOW: x + delta, clamped to [0, 1], detached.
    # WHY: Valid image, no gradient history.
    # PROS & CONS: + Valid and light. - None.
    # EXAMPLE: Same shape as x.
    return torch.clamp(x + delta, 0.0, 1.0).detach()  # original + allowed change, pixels kept between 0 and 1


# WHAT: Safety check that an attack followed its rules.
# HOW: Compute the largest absolute pixel change (L-infinity norm) and assert it is <= eps + tol; assert all
#      pixels are in [0, 1].
# WHY: A bug here would make every robustness number meaningless (an attack that cheats with a larger eps looks
#      stronger than it is).
# PROS & CONS: + Cheap, catches silent errors. - Only checks the rules, not whether the attack is effective.
# EXAMPLE: check_perturbation(x, x_adv, 8/255) returns about 0.03137 for a full-strength PGD.
def check_perturbation(x, x_adv, eps, tol=1e-6):  # safety check: did the attack stay within the rules?
    # WHAT: Largest per-pixel change in the whole batch.
    # HOW: max over |x_adv - x| of all pixels.
    # WHY: The L-infinity norm is exactly the quantity eps bounds.
    # PROS & CONS: + One number summarises the whole batch. - Does not say which image broke the rule.
    # EXAMPLE: linf = 0.0314 when some pixel moved the full 8/255.
    linf = torch.max(torch.abs(x_adv - x)).item()  # the biggest change made to any single pixel
    # WHAT: Fails loudly if any pixel moved more than eps.
    # HOW: tol = 1e-6 allows float rounding.
    # WHY: Rounding makes 8/255 + tiny error possible.
    # PROS & CONS: + Tolerant to float error. - A tolerance could hide an error smaller than 1e-6 (irrelevant
    #              for images).
    # EXAMPLE: linf = 0.03137256 <= 0.03137255 + 1e-6 passes.
    assert linf <= eps + tol, f"L-inf violated: {linf:.8f} > eps {eps:.8f}"  # stop with an error if some pixel changed more than eps (tol = tiny rounding allowance)
    # WHAT: Fails loudly if any pixel left the valid range.
    # HOW: Check the minimum is >= 0 and the maximum is <= 1.
    # WHY: Out-of-range pixels are not real images.
    # PROS & CONS: + Guarantees valid images. - Asserts can be disabled with python -O.
    # EXAMPLE: x_adv.min() == 0.0 and x_adv.max() == 1.0 pass.
    assert 0.0 <= x_adv.min().item() and x_adv.max().item() <= 1.0, "x_adv left the [0, 1] range"  # stop with an error if any pixel is outside 0..1
    # WHAT: Returns the measured largest change for logging or tests.
    # HOW: Plain float.
    # WHY: Lets callers print or compare it.
    # PROS & CONS: + Useful in tests. - None.
    # EXAMPLE: 0.03137254901960784 for eps = 8/255.
    return linf  # report the biggest change


# WHAT: Built-in unit test for all attacks, run with: python -m src.attacks.
# HOW: Builds a tiny model and random images, then checks every attack at several eps values.
# WHY: Lets you verify the attacks anywhere (even on CPU) without downloading models or data.
# PROS & CONS: + Fast (seconds). - Uses a toy model, so it checks the rules, not the strength.
# EXAMPLE: Prints '[attacks] self-test passed' when everything holds.
def self_test(device=None, n=8):  # quick built-in test with fake images; run it with: python -m src.attacks
    # WHAT: Imports PyTorch layers for the toy model, only inside the test.
    # HOW: Local import keeps normal imports light.
    # WHY: Only the test needs layers.
    # PROS & CONS: + No cost in production. - Unusual import position.
    # EXAMPLE: nn.Linear(8, 10) is a layer with 8 inputs and 10 outputs.
    import torch.nn as nn  # building blocks for a tiny model
    # WHAT: Picks the device for the test.
    # HOW: Use the one given, otherwise the CPU.
    # WHY: The test must run anywhere.
    # PROS & CONS: + Portable. - Slow on large inputs (not an issue here).
    # EXAMPLE: self_test() uses the CPU.
    device = device or torch.device("cpu")  # use the CPU if no device was given
    # WHAT: Makes the random numbers repeatable.
    # HOW: Fixes PyTorch's random generator.
    # WHY: A test that changes between runs is hard to debug.
    # PROS & CONS: + Reproducible. - Tests only one random draw.
    # EXAMPLE: The same fake images appear every run.
    torch.manual_seed(0)  # fixed starting number so the random numbers are the same every time
    # WHAT: A toy classifier: one conv layer, ReLU, average pooling, one linear layer.
    # HOW: Turns a 3x32x32 image into 10 scores; .eval() means test mode.
    # WHY: A tiny model is enough to check attack rules.
    # PROS & CONS: + Fast and dependency-free. - Not representative of ResNet or ViT.
    # EXAMPLE: Output shape (8, 10) for 8 images.
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),  # tiny test model: look at the image, keep positive signals, average them
                          nn.Flatten(), nn.Linear(8, 10)).to(device).eval()  # then flatten and give 10 class scores; eval = test mode
    # WHAT: Creates fake images and labels.
    # HOW: rand gives values in [0, 1); randint gives labels 0..9.
    # WHY: No dataset is needed to test the attacks.
    # PROS & CONS: + Fully offline. - Random images are not natural images.
    # EXAMPLE: x.shape == (8, 3, 32, 32).
    x = torch.rand(n, 3, 32, 32, device=device)  # n fake colour images of size 32x32
    y = torch.randint(0, 10, (n,), device=device)  # n fake labels between 0 and 9
    # WHAT: Tests the attacks at several budgets.
    # HOW: Includes 0 (no attack) up to 16/255, the same budgets as the report.
    # WHY: Edge cases (0) and large budgets are where bugs hide.
    # PROS & CONS: + Covers the whole range. - Not exhaustive.
    # EXAMPLE: eps = 8/255 is the training budget.
    for eps in (0.0, 1 / 255, 4 / 255, 8 / 255, 16 / 255):  # try several attack strengths
        # WHAT: Lists every attack as a small function to test.
        # HOW: FGSM, PGD-10, PGD without random start (3 steps), RS-FGSM.
        # WHY: All attack variants share the same rules.
        # PROS & CONS: + One loop tests all. - Slight duplication of attack settings.
        # EXAMPLE: ('pgd10', lambda: pgd_attack(model, x, y, eps, steps=10)).
        for name, fn in (("fgsm", lambda: fgsm_attack(model, x, y, eps)),  # test the FGSM attack
                         ("pgd10", lambda: pgd_attack(model, x, y, eps, steps=10)),  # test PGD with 10 steps
                         ("pgd-no-rs", lambda: pgd_attack(model, x, y, eps, steps=3, random_start=False)),  # test PGD with 3 steps and no random start
                         ("rs_fgsm", lambda: rs_fgsm_attack(model, x, y, eps))):  # test RS-FGSM
            # WHAT: Runs one attack and checks it.
            # HOW: Call the attack, then check_perturbation verifies the L-infinity and range rules.
            # WHY: Each attack must obey the same rules.
            # PROS & CONS: + Fails at the first violation. - Stops at the first failure only.
            # EXAMPLE: Raises AssertionError if a pixel moved more than eps.
            x_adv = fn()  # run the attack
            linf = check_perturbation(x, x_adv, eps)  # make sure it followed the rules
            # WHAT: Extra test for the eps = 0 case.
            # HOW: Requires the largest change to be exactly 0.
            # WHY: With eps = 0 the image must be returned unchanged.
            # PROS & CONS: + Strict. - None.
            # EXAMPLE: linf == 0.0.
            if eps == 0:  # with eps = 0 nothing should change
                assert linf == 0.0, f"{name}: eps=0 must return x unchanged"  # stop with an error if the image changed
        # WHAT: Progress message per budget.
        # HOW: Prints eps converted back to /255 units.
        # WHY: Shows which budgets passed.
        # PROS & CONS: + Visible progress. - Console noise.
        # EXAMPLE: [attacks] eps=8/255: fgsm / pgd / rs_fgsm all satisfy L-inf <= eps and [0, 1]
        print(f"[attacks] eps={eps * 255:.0f}/255: fgsm / pgd / rs_fgsm all satisfy L-inf <= eps and [0, 1]")  # tell the user this strength passed
    # WHAT: Checks the attacks did not touch the model's weight gradients.
    # HOW: Every parameter's .grad must still be None.
    # WHY: If attacks left gradients on the weights, adversarial training would silently accumulate wrong
    #      updates.
    # PROS & CONS: + Catches a subtle bug. - Only checks this toy model.
    # EXAMPLE: p.grad is None for all 8*3*3*3+8 + 8*10+10 parameters.
    assert all(p.grad is None for p in model.parameters()), "attacks must not accumulate parameter grads"  # attacks must not leave any training information inside the model
    # WHAT: Final success message.
    # HOW: Printed only if no assertion failed.
    # WHY: Clear signal to the user.
    # PROS & CONS: + Obvious. - None.
    # EXAMPLE: [attacks] self-test passed
    print("[attacks] self-test passed")  # all checks passed


# WHAT: Entry point when the file is run directly.
# HOW: Python sets __name__ to '__main__' only for direct runs.
# WHY: Lets 'python -m src.attacks' run the test while imports stay silent.
# PROS & CONS: + Standard pattern. - None.
# EXAMPLE: python -m src.attacks
if __name__ == "__main__":  # true only when this file is run directly
    self_test()  # run the quick test
