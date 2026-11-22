"""Defence: train the model on tricked images too, so it learns not to be fooled."""
# WHAT: Imports helpers for files and timing, PyTorch, a progress bar, the training attacks and the project
#       utilities.
# HOW: tqdm shows a progress bar, F.cross_entropy is the loss, pgd_attack and rs_fgsm_attack create the
#      adversarial images.
# WHY: Adversarial training needs an attack inside the training loop.
# PROS & CONS: + Reuses the evaluation attacks. - None.
# EXAMPLE: from .attacks import pgd_attack, rs_fgsm_attack.
import os  # work with file and folder paths
import time  # measure how long things take

import torch  # main library for deep learning
import torch.nn.functional as F  # ready-made maths functions (we use it for the loss)
from tqdm import tqdm  # shows a progress bar while training

from .attacks import pgd_attack, rs_fgsm_attack  # the two attacks we use to make tricked images during training
from .models import save_checkpoint  # function that saves the model's learned numbers to a file
from .utils import git_auto_push, load_json, save_json, sync_to_drive  # helper functions from utils.py


# WHAT: Switches on mixed-precision maths only when a GPU is used.
# HOW: torch.autocast(device_type, enabled=...) with a CUDA check.
# WHY: fp16 is about twice as fast on a T4 and halves memory; on CPU it gives no benefit.
# PROS & CONS: + One helper for all loops. - fp16 can lose small gradients (GradScaler compensates).
# EXAMPLE: _autocast(device, True) on CUDA runs convolutions in fp16.
def _autocast(device, enabled):  # switches on faster half-precision maths (only on a GPU)
    return torch.autocast(device_type=device.type, enabled=bool(enabled) and device.type == "cuda")  # on only if asked AND a GPU is used


# WHAT: Chooses the attack used to create adversarial images during training.
# HOW: 'pgd5': 5 steps of size 2/255 with a random start. 'rs_fgsm': one step of size 1.25*eps with a random
#      start.
# WHY: PGD-5 is the stronger and safer defence; RS-FGSM is faster but risked catastrophic overfitting.
# PROS & CONS: + Two interchangeable options. - PGD-5 costs about 2.5x more time for the ViT.
# EXAMPLE: make_train_attack('pgd5', 8/255).
def make_train_attack(method, eps):  # picks which attack makes the tricked images during training
    if method == "pgd5":  # option 1: PGD with 5 small steps (stronger, slower)
        return lambda m, x, y: pgd_attack(m, x, y, eps=eps, alpha=2 / 255, steps=5, random_start=True, amp=True)  # 5 steps of size 2/255 with a random start
    if method == "rs_fgsm":  # option 2: RS-FGSM, one step (faster, a bit less stable)
        return lambda m, x, y: rs_fgsm_attack(m, x, y, eps=eps, alpha=1.25 * eps, amp=True)  # one step of size 1.25 * eps with a random start
    raise ValueError(f"method must be 'pgd5' or 'rs_fgsm', got {method!r}")  # any other name is a mistake, so show an error


# WHAT: Keeps a log of training progress in a JSON file.
# HOW: Loads the old log, adds settings and the new epoch entry (replacing an entry of the same epoch), saves
#      it.
# WHY: Enables resuming and later plotting of training curves.
# PROS & CONS: + Human-readable. - Rewrites the whole file each epoch (small).
# EXAMPLE: checkpoints/training_history.json with train_loss, train_clean_acc, train_robust_acc.
def update_history(history_path, tag, entry=None, **meta):  # keeps a log file of how training went (loss and accuracy per epoch)
    hist = load_json(history_path, {}) or {}  # read the old log (or start an empty one)
    rec = hist.setdefault(tag, {"epochs": []})  # find this model's part of the log (or create it)
    rec.update(meta)  # store extra settings such as method, eps, lr
    if entry is not None:  # an entry = the result of one finished epoch
        rec["epochs"] = [e for e in rec["epochs"] if e["epoch"] != entry["epoch"]] + [entry]  # replace an old entry of the same epoch, then add the new one
    save_json(hist, history_path)  # write the log back to the file


# WHAT: True if a model has already finished all epochs.
# HOW: Reads the log: 'completed' flag set and at least `epochs` entries.
# WHY: Prevents retraining when a finished checkpoint exists.
# PROS & CONS: + Saves hours. - Trusts the log.
# EXAMPLE: is_training_complete(path, 'vit_adv', 10).
def is_training_complete(history_path, tag, epochs):  # True if this model has already finished all its epochs
    rec = (load_json(history_path, {}) or {}).get(tag, {})  # read this model's part of the log
    return bool(rec.get("completed")) and len(rec.get("epochs", [])) >= epochs  # marked finished AND enough epochs are recorded


# WHAT: Adversarial training: trains an already trained model on clean AND adversarial images.
# HOW: For each batch, craft x_adv with the chosen attack (model in eval mode), then take one optimiser step on
#      0.5*CE(clean) + 0.5*CE(adv). Uses AdamW, cosine learning-rate decay, mixed precision and a resumable
#      checkpoint.
# WHY: It implements Madry et al.'s defence: min over weights of the worst-case loss inside the eps-ball. Mixing
#      clean loss limits the clean-accuracy drop (about 3 points).
# PROS & CONS: + Large robustness gain (PGD-10 at 8/255: 0% to 18.3% CNN, 42.2% ViT). - About 6 forward/backward
#              passes per batch; ViT took 74.7 minutes for 10 epochs; clean accuracy drops.
# EXAMPLE: adversarial_finetune(model, loader, epochs=10, eps=8/255, method='pgd5').
def adversarial_finetune(model, train_loader, epochs=10, eps=8 / 255, method="pgd5", lr=1e-4,  # train an already-trained model more, using tricked images too
                         device=None, tag="cnn_adv", ckpt_dir="checkpoints", drive_dir=None,  # device = GPU/CPU, tag = name, ckpt_dir = folder for saved files
                         auto_push_every=0, push_fn=git_auto_push, max_batches=0):  # optional GitHub upload; max_batches only limits the quick test
    # WHAT: Picks the device.
    # HOW: Uses the model's own device if none is given.
    # WHY: Avoids mismatches between model and data.
    # PROS & CONS: + Safe default. - None.
    # EXAMPLE: device(type='cuda').
    device = device or next(model.parameters()).device  # if no device was given, use the one the model is already on
    # WHAT: Moves the model to the chosen device.
    # HOW: Copies weights to GPU memory.
    # WHY: Training on GPU is far faster.
    # PROS & CONS: + Fast. - GPU memory limit.
    # EXAMPLE: T4 has 15 GB.
    model.to(device)  # move the model to the GPU/CPU
    # WHAT: Selects the attack function.
    # HOW: Returns a function (model, x, y) -> x_adv.
    # WHY: Training code stays independent of the attack type.
    # PROS & CONS: + Flexible. - None.
    # EXAMPLE: PGD-5.
    attack = make_train_attack(method, eps)  # choose the attack used to make tricked images
    # WHAT: Mixed precision flag.
    # HOW: True on CUDA.
    # WHY: Speed and memory.
    # PROS & CONS: + 2x faster. - Needs GradScaler.
    # EXAMPLE: True on Colab.
    use_amp = device.type == "cuda"  # use faster half-precision maths only on a GPU
    # WHAT: File names for the final weights, the resume state and the log.
    # HOW: Built from the tag (cnn_adv or vit_adv).
    # WHY: Separate files let a crashed run resume.
    # PROS & CONS: + Resumable. - Extra disk (optimizer state is 2x model size).
    # EXAMPLE: checkpoints/vit_adv.pth, vit_adv_state.pt.
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")  # file for the final learned numbers (weights)
    state_path = os.path.join(ckpt_dir, f"{tag}_state.pt")  # file with everything needed to continue after a crash
    hist_path = os.path.join(ckpt_dir, "training_history.json")  # the log file of loss and accuracy
    # WHAT: The weight-update rule.
    # HOW: AdamW with learning rate 1e-4 and weight decay 1e-4.
    # WHY: Small learning rate fine-tunes pretrained weights gently; decay stops weights growing.
    # PROS & CONS: + Robust default for CNN and ViT. - Needs tuning for training from scratch.
    # EXAMPLE: lr = 0.0001.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # the tool that updates the weights; lr = how big each update is; weight_decay stops weights growing too large
    # WHAT: Learning-rate plan.
    # HOW: Cosine annealing from the initial value to 0 over the epochs.
    # WHY: Large steps early, small steps late give a smoother end.
    # PROS & CONS: + No manual schedule. - Assumes the planned number of epochs.
    # EXAMPLE: Epoch 1: 9.8e-5, epoch 10: 0.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # slowly lowers the learning rate towards 0 as epochs pass
    # WHAT: Protects small fp16 gradients.
    # HOW: Multiplies the loss by a large factor before backward and divides afterwards.
    # WHY: fp16 cannot represent very small numbers.
    # PROS & CONS: + Stable mixed precision. - Rare skipped steps when overflow occurs.
    # EXAMPLE: scale = 65536 initially.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # stops very small numbers becoming 0 in half-precision maths
    # WHAT: Resume logic.
    # HOW: If a state file exists, restore model, optimiser, scheduler, scaler and the epoch counter.
    # WHY: Colab may disconnect during a 75-minute run.
    # PROS & CONS: + Lose at most one epoch. - State file is large.
    # EXAMPLE: '[vit_adv] Resuming from epoch 4/10'.
    start_epoch = 0  # begin from the first epoch unless we find a saved state
    if os.path.exists(state_path):  # a saved state means an earlier run stopped half-way
        st = torch.load(state_path, map_location=device)  # load that saved state
        model.load_state_dict(st["model"])  # put the saved weights back into the model
        optimizer.load_state_dict(st["optimizer"])  # restore the weight-updating tool
        scheduler.load_state_dict(st["scheduler"])  # restore the learning-rate plan
        scaler.load_state_dict(st["scaler"])  # restore the small-number protector
        start_epoch = st["epoch"]  # continue from the epoch where it stopped
        print(f"[{tag}] Resuming from epoch {start_epoch}/{epochs}")  # tell the user training is continuing
    # WHAT: Writes the run settings into the log and marks it unfinished.
    # HOW: Stores method, eps, learning rate and device.
    # WHY: Documents exactly how a result was produced.
    # PROS & CONS: + Reproducibility. - None.
    # EXAMPLE: method='pgd5', eps=0.0314.
    update_history(hist_path, tag, completed=False, method=method, eps=eps, lr=lr, target_epochs=epochs,  # write the settings into the log and mark "not finished yet"
                   amp=use_amp, device=str(device))  # also store whether half-precision maths and which device are used
    # WHAT: The epoch loop: one pass over all 50,000 training images.
    # HOW: Repeats batch training, logging and checkpointing.
    # WHY: Standard training structure.
    # PROS & CONS: + Resumable via start_epoch. - None.
    # EXAMPLE: 10 epochs.
    for epoch in range(start_epoch, epochs):  # one loop = one epoch = one full pass over all training images
        # WHAT: Starts the stopwatch and the running totals for this epoch.
        # HOW: Totals of loss, clean-correct, adversarial-correct and image count.
        # WHY: Needed for the log and the progress bar.
        # PROS & CONS: + Cheap. - None.
        # EXAMPLE: seconds = 447 for the ViT.
        t0 = time.time()  # note the start time of this epoch
        tot_loss = tot_clean = tot_adv = n = 0  # running totals: loss, correct normal guesses, correct tricked guesses, images seen
        # WHAT: Progress bar.
        # HOW: Wraps the loader.
        # WHY: Shows remaining time during long runs.
        # PROS & CONS: + Visible progress. - Clutters logs.
        # EXAMPLE: [vit_adv] Epoch 3/10 ... loss 0.72
        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch + 1}/{epochs}", leave=False)  # progress bar over the batches
        # WHAT: Loop over batches.
        # HOW: x = images, y = labels.
        # WHY: One optimiser step per batch.
        # PROS & CONS: + Standard. - None.
        # EXAMPLE: 390 batches of 128.
        for b, (x, y) in enumerate(pbar):  # take one batch (a small group) of images x and their labels y
            # WHAT: Stops early in quick tests.
            # HOW: Only active when max_batches > 0.
            # WHY: Smoke tests finish in minutes.
            # PROS & CONS: + Fast tests. - Must be 0 in real runs.
            # EXAMPLE: max_batches=2 in --quick-test.
            if max_batches and b >= max_batches:  # only the quick test sets a limit
                break  # stop this epoch early
            # WHAT: Moves the batch to the GPU.
            # HOW: non_blocking overlaps copying and computing.
            # WHY: Data starts on the CPU.
            # PROS & CONS: + Faster. - Needs pinned memory.
            # EXAMPLE: x is a tensor in GPU memory after this line.
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # move the batch to the GPU/CPU
            # WHAT: Test mode while crafting adversarial images.
            # HOW: BatchNorm uses running statistics, dropout is off.
            # WHY: The attack must target the model as it will be used at test time and must not change
            #      BatchNorm statistics.
            # PROS & CONS: + Deterministic attack. - Slight mismatch with train-mode statistics.
            # EXAMPLE: model.training is False.
            model.eval()  # test mode while making tricked images (so the model does not change)
            # WHAT: Crafts the adversarial batch.
            # HOW: x_adv = attack(model, x, y) runs 5 forward/backward passes (PGD-5) in fp16.
            # WHY: This is the inner maximisation of adversarial training.
            # PROS & CONS: + Cheap in fp16. - Largest cost of training (about 5 of the 6 passes).
            # EXAMPLE: x_adv.shape == x.shape; |x_adv - x| <= 8/255.
            with _autocast(device, use_amp):  # faster maths if a GPU is used
                x_adv = attack(model, x, y)  # make the tricked version of every image in the batch
            # WHAT: Back to training mode.
            # HOW: BatchNorm updates statistics, dropout is on.
            # WHY: Weights are updated now.
            # PROS & CONS: + Standard. - None.
            # EXAMPLE: model.training is True.
            model.train()  # back to learning mode
            # WHAT: Clears old gradients.
            # HOW: Sets them to None (cheaper than zeros).
            # WHY: Gradients would otherwise accumulate across batches.
            # PROS & CONS: + Saves memory. - None.
            # EXAMPLE: None.
            optimizer.zero_grad(set_to_none=True)  # erase the update directions left from the previous batch
            # WHAT: Forward pass on clean and adversarial images together.
            # HOW: torch.cat stacks them into one batch of 256, giving one forward pass.
            # WHY: One pass is faster than two.
            # PROS & CONS: + Faster. - BatchNorm statistics mix clean and adversarial images.
            # EXAMPLE: logits.shape == (256, 10).
            with _autocast(device, use_amp):  # faster maths if a GPU is used
                logits = model(torch.cat([x, x_adv], dim=0))  # give normal and tricked images together in one go; get 10 scores per image
            # WHAT: Loss in full precision.
            # HOW: Converts fp16 scores to fp32.
            # WHY: Cross-entropy is numerically sensitive.
            # PROS & CONS: + Accurate. - Tiny cost.
            # EXAMPLE: float32.
            logits = logits.float()  # use normal-precision numbers for the loss
            # WHAT: Remembers the batch size.
            # HOW: Used to split clean (first half) from adversarial (second half).
            # WHY: Needed for the loss.
            # PROS & CONS: + Simple. - None.
            # EXAMPLE: 128.
            bs = x.size(0)  # bs = number of images in the batch
            # WHAT: THE training objective: 0.5 * CE(clean) + 0.5 * CE(adversarial).
            # HOW: Average of the loss on clean images and on their adversarial versions.
            # WHY: A purely adversarial loss lowers clean accuracy a lot; mixing 50/50 keeps the clean drop near
            #      3 points.
            # PROS & CONS: + Controlled accuracy-robustness trade-off. - Less robust than 100% adversarial loss.
            # EXAMPLE: Epoch 10 CNN: 0.99; ViT: 0.52.
            loss = 0.5 * F.cross_entropy(logits[:bs], y) + 0.5 * F.cross_entropy(logits[bs:], y)  # loss = half "mistakes on normal images" + half "mistakes on tricked images"
            # WHAT: Computes how every weight should change.
            # HOW: Backpropagation through the scaled loss.
            # WHY: Standard learning step.
            # PROS & CONS: + Stable. - None.
            # EXAMPLE: Gradients exist for all weights.
            scaler.scale(loss).backward()  # work out how every weight should change (made bigger first so small numbers survive)
            # WHAT: Applies the weight update.
            # HOW: AdamW update using the unscaled gradients.
            # WHY: Learning.
            # PROS & CONS: + Skips steps with overflow. - Occasionally skips one batch.
            # EXAMPLE: Weights change by about lr.
            scaler.step(optimizer)  # actually change the weights
            # WHAT: Adjusts the loss scale.
            # HOW: Increases the scale slowly, halves it after an overflow.
            # WHY: Keeps fp16 stable.
            # PROS & CONS: + Automatic. - None.
            # EXAMPLE: 65536 to 32768 after overflow.
            scaler.update()  # adjust the protector for the next batch
            # WHAT: Updates the running totals.
            # HOW: Loss times batch size, plus the number of correctly classified clean and adversarial images.
            # WHY: Gives epoch averages for the log.
            # PROS & CONS: + Cheap. - .item() forces a GPU sync.
            # EXAMPLE: train_clean_acc = 100 * tot_clean / n.
            tot_loss += loss.item() * bs  # add this batch's loss to the running total
            tot_clean += (logits[:bs].argmax(1) == y).sum().item()  # count normal images guessed right (highest score = the guess)
            tot_adv += (logits[bs:].argmax(1) == y).sum().item()  # count tricked images guessed right
            n += bs  # count the images seen so far
            # WHAT: Shows live numbers on the progress bar.
            # HOW: Loss, clean accuracy and adversarial accuracy.
            # WHY: Watch training without waiting for the epoch end.
            # PROS & CONS: + Feedback. - None.
            # EXAMPLE: loss=0.720 clean=97.3% robust=47.6%
            pbar.set_postfix(loss=f"{tot_loss / n:.3f}", clean=f"{100 * tot_clean / n:.1f}%",  # show the loss and the normal accuracy on the progress bar
                             robust=f"{100 * tot_adv / n:.1f}%")  # and the accuracy on tricked images
        # WHAT: Closes the progress bar.
        # HOW: Frees the terminal line.
        # WHY: Cleaner logs.
        # PROS & CONS: + Tidy. - None.
        # EXAMPLE: With leave=False the finished bar disappears and only the printed epoch line remains.
        pbar.close()  # close the progress bar
        # WHAT: Lowers the learning rate once per epoch.
        # HOW: Moves one step along the cosine curve.
        # WHY: Smooth convergence.
        # PROS & CONS: + Standard. - None.
        # EXAMPLE: lr 9.76e-5 after epoch 1.
        scheduler.step()  # lower the learning rate a little, once per epoch
        # WHAT: Switches to test mode before saving.
        # HOW: Disables training behaviour.
        # WHY: Saved models are used for evaluation.
        # PROS & CONS: + Safe. - None.
        # EXAMPLE: model.training is False.
        model.eval()  # test mode again
        # WHAT: Summary of the epoch.
        # HOW: Dictionary with epoch number, average loss, clean and robust training accuracy, learning rate and
        #      seconds.
        # WHY: Written to the log and used for the report's robust-generalisation numbers.
        # PROS & CONS: + Complete record. - Training accuracy is measured on augmented batches.
        # EXAMPLE: Epoch 10 ViT: robust 59.11%, 446 s.
        entry = {"epoch": epoch + 1, "train_loss": tot_loss / max(n, 1),  # summary of this epoch: epoch number and average loss (max(n, 1) avoids dividing by 0)
                 "train_clean_acc": 100 * tot_clean / max(n, 1), "train_robust_acc": 100 * tot_adv / max(n, 1),  # accuracy on normal images and on tricked images
                 "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}  # learning rate used and time taken
        # WHAT: Prints the epoch result.
        # HOW: One line per epoch.
        # WHY: Keeps a readable log in Colab.
        # PROS & CONS: + Visible. - None.
        # EXAMPLE: [vit_adv] Epoch 10/10 | loss 0.52 | clean 99.85% | robust 59.11% | 446s
        print(f"[{tag}] Epoch {entry['epoch']}/{epochs} | loss {entry['train_loss']:.4f} | "  # print the epoch result: epoch number and loss
              f"clean {entry['train_clean_acc']:.2f}% | robust {entry['train_robust_acc']:.2f}% | "  # normal accuracy and tricked-image accuracy
              f"{entry['seconds']:.0f}s")  # seconds taken
        # WHAT: Saves the weights after every epoch.
        # HOW: Overwrites the file.
        # WHY: A disconnect loses at most one epoch.
        # PROS & CONS: + Safe. - Always the latest, not the best, epoch.
        # EXAMPLE: checkpoints/vit_adv.pth.
        save_checkpoint(model, ckpt_path)  # save the learned weights after every epoch
        # WHAT: Saves everything needed to resume.
        # HOW: Weights, optimiser, scheduler, scaler and epoch number.
        # WHY: Exact continuation.
        # PROS & CONS: + Resumable. - Large file.
        # EXAMPLE: vit_adv_state.pt.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),  # also save everything needed to continue later: weights, weight-updating tool
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),  # learning-rate plan, small-number protector
                    "epoch": epoch + 1}, state_path)  # and the epoch number
        # WHAT: Adds the epoch entry to the log.
        # HOW: Rewrites the JSON.
        # WHY: Training curves can be plotted later.
        # PROS & CONS: + Persistent. - None.
        # EXAMPLE: entries list grows to 10.
        update_history(hist_path, tag, entry=entry)  # add this epoch's result to the log
        # WHAT: Backs up to Google Drive.
        # HOW: Copies files if a Drive folder is given.
        # WHY: Colab local files vanish.
        # PROS & CONS: + Safe. - Slower epoch end.
        # EXAMPLE: Copies cnn_adv.pth.
        sync_to_drive(ckpt_dir, drive_dir)  # copy the saved files to Google Drive as a backup (only if a Drive folder is given)
        # WHAT: Optional GitHub backup every few epochs.
        # HOW: Pushes after epochs 5 (not the last).
        # WHY: Protects intermediate progress.
        # PROS & CONS: + Remote backup. - Needs a token.
        # EXAMPLE: epoch 5.
        if auto_push_every and (epoch + 1) % auto_push_every == 0 and (epoch + 1) < epochs:  # every few epochs, if upload is switched on
            push_fn(f"chore(train): {tag} epoch {epoch + 1}/{epochs} checkpoint")  # upload progress to GitHub
    # WHAT: Marks training as finished.
    # HOW: Sets completed=True.
    # WHY: Next run will skip retraining.
    # PROS & CONS: + Idempotent. - None.
    # EXAMPLE: completed: true.
    update_history(hist_path, tag, completed=True)  # mark in the log that training is finished
    # WHAT: Deletes the resume state.
    # HOW: The finished model is in the .pth file.
    # WHY: Frees disk space and prevents wrong resumes.
    # PROS & CONS: + Clean. - None.
    # EXAMPLE: vit_adv_state.pt removed.
    if os.path.exists(state_path):  # the "continue later" file is not needed any more
        os.remove(state_path)  # delete it
    # WHAT: Final backup.
    # HOW: Copies the finished files.
    # WHY: Secure result.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: cnn_adv.pth and training_history.json are copied to the Drive folder.
    sync_to_drive(ckpt_dir, drive_dir)  # final backup to Google Drive
    # WHAT: Leaves the model in test mode.
    # HOW: eval().
    # WHY: Callers evaluate it next.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: model.training is False when adversarial_finetune returns.
    model.eval()  # leave the model in test mode
    # WHAT: Returns the list of epoch summaries.
    # HOW: Reads them back from the log.
    # WHY: Lets scripts plot or inspect them.
    # PROS & CONS: + Convenient. - None.
    # EXAMPLE: 10 entries.
    return load_json(hist_path)[tag]["epochs"]  # give back the list of epoch results


# WHAT: Normal (non-adversarial) fine-tuning, used to prepare the ViT and the ResNet-18 for CIFAR-10.
# HOW: Standard training with cross-entropy, AdamW (lr 2e-4), cosine decay and mixed precision.
# WHY: A freshly attached 10-class head must learn CIFAR-10 before adversarial training starts.
# PROS & CONS: + Fast: no attack passes, so far cheaper than adversarial training. - No robustness at all.
# EXAMPLE: 5 epochs bring ViT-Tiny to 97.1% test accuracy.
def clean_finetune(model, train_loader, epochs=5, lr=2e-4, device=None, max_batches=0, desc="ViT-clean"):  # normal training (no tricked images); used once to prepare the ViT for CIFAR-10
    # WHAT: Picks the device for clean training.
    # HOW: Uses the model's own device when none is passed.
    # WHY: Prevents a mismatch between model and data devices.
    # PROS & CONS: + Safe default. - None.
    # EXAMPLE: clean_finetune(model, loader) trains on the model's current device.
    device = device or next(model.parameters()).device  # if no device was given, use the one the model is already on
    use_amp = device.type == "cuda"  # use faster half-precision maths only on a GPU
    # WHAT: Optimiser, scheduler and scaler for clean training.
    # HOW: AdamW lr 2e-4, cosine, GradScaler.
    # WHY: Same tools as adversarial training; larger learning rate because the head is new.
    # PROS & CONS: + Consistent. - None.
    # EXAMPLE: lr 0.0002.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # the tool that updates the weights
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # slowly lowers the learning rate towards 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # stops very small numbers becoming 0 in half-precision maths
    # WHAT: Epoch loop of clean training.
    # HOW: Forward, loss, backward, step per batch.
    # WHY: Learn the 10 classes.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: 5 epochs.
    for epoch in range(epochs):  # one loop = one full pass over all training images
        model.train()  # learning mode
        tot = correct = n = 0  # running totals: loss, correct guesses, images seen
        pbar = tqdm(train_loader, desc=f"[{desc}] Epoch {epoch + 1}/{epochs}", leave=False)  # progress bar over the batches
        for b, (x, y) in enumerate(pbar):  # take one batch of images x and labels y
            if max_batches and b >= max_batches:  # only the quick test sets a limit
                break  # stop this epoch early
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # move the batch to the GPU/CPU
            optimizer.zero_grad(set_to_none=True)  # erase the update directions from the previous batch
            with _autocast(device, use_amp):  # faster maths if a GPU is used
                logits = model(x)  # 10 scores per image, normal images only
            # WHAT: Clean loss.
            # HOW: Cross-entropy only on normal images.
            # WHY: No attack here.
            # PROS & CONS: + Standard. - No robustness.
            # EXAMPLE: The loss falls over the 5 epochs as the new head learns.
            loss = F.cross_entropy(logits.float(), y)  # loss = how wrong the guesses are
            scaler.scale(loss).backward()  # work out how every weight should change
            scaler.step(optimizer)  # change the weights
            scaler.update()  # adjust the small-number protector
            tot += loss.item() * x.size(0)  # add this batch's loss to the running total
            correct += (logits.argmax(1) == y).sum().item()  # count right guesses (highest score = the guess)
            n += x.size(0)  # count the images seen so far
            pbar.set_postfix(loss=f"{tot / n:.3f}", acc=f"{100 * correct / n:.1f}%")  # show loss and accuracy on the progress bar
        pbar.close()  # close the progress bar
        scheduler.step()  # lower the learning rate a little
        # WHAT: Prints loss and accuracy after each epoch.
        # HOW: One log line.
        # WHY: Monitoring.
        # PROS & CONS: + Visible. - None.
        # EXAMPLE: [ViT-clean] Epoch 5/5 | loss <average loss> | acc <training accuracy>%
        print(f"[{desc}] Epoch {epoch + 1}/{epochs} | loss {tot / max(n, 1):.4f} | acc {100 * correct / max(n, 1):.2f}%")  # print this epoch's loss and accuracy
    # WHAT: Returns the trained model in test mode.
    # HOW: eval().
    # WHY: Ready for evaluation or copying.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: vit_clean = clean_finetune(vit_clean, loader, epochs=5, device=device).
    return model.eval()  # give back the trained model in test mode
