"""Defence: train the model on tricked images too, so it learns not to be fooled."""
import os  # work with file and folder paths
import time  # measure how long things take

import torch  # main library for deep learning
import torch.nn.functional as F  # ready-made maths functions (we use it for the loss)
from tqdm import tqdm  # shows a progress bar while training

from .attacks import pgd_attack, rs_fgsm_attack  # the two attacks we use to make tricked images during training
from .models import save_checkpoint  # function that saves the model's learned numbers to a file
from .utils import git_auto_push, load_json, save_json, sync_to_drive  # helper functions from utils.py


def _autocast(device, enabled):  # switches on faster half-precision maths (only on a GPU)
    return torch.autocast(device_type=device.type, enabled=bool(enabled) and device.type == "cuda")  # on only if asked AND a GPU is used


def make_train_attack(method, eps):  # picks which attack makes the tricked images during training
    if method == "pgd5":  # option 1: PGD with 5 small steps (stronger, slower)
        return lambda m, x, y: pgd_attack(m, x, y, eps=eps, alpha=2 / 255, steps=5, random_start=True, amp=True)  # 5 steps of size 2/255 with a random start
    if method == "rs_fgsm":  # option 2: RS-FGSM, one step (faster, a bit less stable)
        return lambda m, x, y: rs_fgsm_attack(m, x, y, eps=eps, alpha=1.25 * eps, amp=True)  # one step of size 1.25 * eps with a random start
    raise ValueError(f"method must be 'pgd5' or 'rs_fgsm', got {method!r}")  # any other name is a mistake, so show an error


def update_history(history_path, tag, entry=None, **meta):  # keeps a log file of how training went (loss and accuracy per epoch)
    hist = load_json(history_path, {}) or {}  # read the old log (or start an empty one)
    rec = hist.setdefault(tag, {"epochs": []})  # find this model's part of the log (or create it)
    rec.update(meta)  # store extra settings such as method, eps, lr
    if entry is not None:  # an entry = the result of one finished epoch
        rec["epochs"] = [e for e in rec["epochs"] if e["epoch"] != entry["epoch"]] + [entry]  # replace an old entry of the same epoch, then add the new one
    save_json(hist, history_path)  # write the log back to the file


def is_training_complete(history_path, tag, epochs):  # True if this model has already finished all its epochs
    rec = (load_json(history_path, {}) or {}).get(tag, {})  # read this model's part of the log
    return bool(rec.get("completed")) and len(rec.get("epochs", [])) >= epochs  # marked finished AND enough epochs are recorded


def adversarial_finetune(model, train_loader, epochs=10, eps=8 / 255, method="pgd5", lr=1e-4,  # train an already-trained model more, using tricked images too
                         device=None, tag="cnn_adv", ckpt_dir="checkpoints", drive_dir=None,  # device = GPU/CPU, tag = name, ckpt_dir = folder for saved files
                         auto_push_every=0, push_fn=git_auto_push, max_batches=0):  # optional GitHub upload; max_batches only limits the quick test
    device = device or next(model.parameters()).device  # if no device was given, use the one the model is already on
    model.to(device)  # move the model to the GPU/CPU
    attack = make_train_attack(method, eps)  # choose the attack used to make tricked images
    use_amp = device.type == "cuda"  # use faster half-precision maths only on a GPU
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")  # file for the final learned numbers (weights)
    state_path = os.path.join(ckpt_dir, f"{tag}_state.pt")  # file with everything needed to continue after a crash
    hist_path = os.path.join(ckpt_dir, "training_history.json")  # the log file of loss and accuracy
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # the tool that updates the weights; lr = how big each update is; weight_decay stops weights growing too large
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # slowly lowers the learning rate towards 0 as epochs pass
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # stops very small numbers becoming 0 in half-precision maths
    start_epoch = 0  # begin from the first epoch unless we find a saved state
    if os.path.exists(state_path):  # a saved state means an earlier run stopped half-way
        st = torch.load(state_path, map_location=device)  # load that saved state
        model.load_state_dict(st["model"])  # put the saved weights back into the model
        optimizer.load_state_dict(st["optimizer"])  # restore the weight-updating tool
        scheduler.load_state_dict(st["scheduler"])  # restore the learning-rate plan
        scaler.load_state_dict(st["scaler"])  # restore the small-number protector
        start_epoch = st["epoch"]  # continue from the epoch where it stopped
        print(f"[{tag}] Resuming from epoch {start_epoch}/{epochs}")  # tell the user training is continuing
    update_history(hist_path, tag, completed=False, method=method, eps=eps, lr=lr, target_epochs=epochs,  # write the settings into the log and mark "not finished yet"
                   amp=use_amp, device=str(device))  # also store whether half-precision maths and which device are used
    for epoch in range(start_epoch, epochs):  # one loop = one epoch = one full pass over all training images
        t0 = time.time()  # note the start time of this epoch
        tot_loss = tot_clean = tot_adv = n = 0  # running totals: loss, correct normal guesses, correct tricked guesses, images seen
        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch + 1}/{epochs}", leave=False)  # progress bar over the batches
        for b, (x, y) in enumerate(pbar):  # take one batch (a small group) of images x and their labels y
            if max_batches and b >= max_batches:  # only the quick test sets a limit
                break  # stop this epoch early
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # move the batch to the GPU/CPU
            model.eval()  # test mode while making tricked images (so the model does not change)
            with _autocast(device, use_amp):  # faster maths if a GPU is used
                x_adv = attack(model, x, y)  # make the tricked version of every image in the batch
            model.train()  # back to learning mode
            optimizer.zero_grad(set_to_none=True)  # erase the update directions left from the previous batch
            with _autocast(device, use_amp):  # faster maths if a GPU is used
                logits = model(torch.cat([x, x_adv], dim=0))  # give normal and tricked images together in one go; get 10 scores per image
            logits = logits.float()  # use normal-precision numbers for the loss
            bs = x.size(0)  # bs = number of images in the batch
            loss = 0.5 * F.cross_entropy(logits[:bs], y) + 0.5 * F.cross_entropy(logits[bs:], y)  # loss = half "mistakes on normal images" + half "mistakes on tricked images"
            scaler.scale(loss).backward()  # work out how every weight should change (made bigger first so small numbers survive)
            scaler.step(optimizer)  # actually change the weights
            scaler.update()  # adjust the protector for the next batch
            tot_loss += loss.item() * bs  # add this batch's loss to the running total
            tot_clean += (logits[:bs].argmax(1) == y).sum().item()  # count normal images guessed right (highest score = the guess)
            tot_adv += (logits[bs:].argmax(1) == y).sum().item()  # count tricked images guessed right
            n += bs  # count the images seen so far
            pbar.set_postfix(loss=f"{tot_loss / n:.3f}", clean=f"{100 * tot_clean / n:.1f}%",  # show the loss and the normal accuracy on the progress bar
                             robust=f"{100 * tot_adv / n:.1f}%")  # and the accuracy on tricked images
        pbar.close()  # close the progress bar
        scheduler.step()  # lower the learning rate a little, once per epoch
        model.eval()  # test mode again
        entry = {"epoch": epoch + 1, "train_loss": tot_loss / max(n, 1),  # summary of this epoch: epoch number and average loss (max(n, 1) avoids dividing by 0)
                 "train_clean_acc": 100 * tot_clean / max(n, 1), "train_robust_acc": 100 * tot_adv / max(n, 1),  # accuracy on normal images and on tricked images
                 "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}  # learning rate used and time taken
        print(f"[{tag}] Epoch {entry['epoch']}/{epochs} | loss {entry['train_loss']:.4f} | "  # print the epoch result: epoch number and loss
              f"clean {entry['train_clean_acc']:.2f}% | robust {entry['train_robust_acc']:.2f}% | "  # normal accuracy and tricked-image accuracy
              f"{entry['seconds']:.0f}s")  # seconds taken
        save_checkpoint(model, ckpt_path)  # save the learned weights after every epoch
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),  # also save everything needed to continue later: weights, weight-updating tool
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),  # learning-rate plan, small-number protector
                    "epoch": epoch + 1}, state_path)  # and the epoch number
        update_history(hist_path, tag, entry=entry)  # add this epoch's result to the log
        sync_to_drive(ckpt_dir, drive_dir)  # copy the saved files to Google Drive as a backup (only if a Drive folder is given)
        if auto_push_every and (epoch + 1) % auto_push_every == 0 and (epoch + 1) < epochs:  # every few epochs, if upload is switched on
            push_fn(f"chore(train): {tag} epoch {epoch + 1}/{epochs} checkpoint")  # upload progress to GitHub
    update_history(hist_path, tag, completed=True)  # mark in the log that training is finished
    if os.path.exists(state_path):  # the "continue later" file is not needed any more
        os.remove(state_path)  # delete it
    sync_to_drive(ckpt_dir, drive_dir)  # final backup to Google Drive
    model.eval()  # leave the model in test mode
    return load_json(hist_path)[tag]["epochs"]  # give back the list of epoch results


def clean_finetune(model, train_loader, epochs=5, lr=2e-4, device=None, max_batches=0, desc="ViT-clean"):  # normal training (no tricked images); used once to prepare the ViT for CIFAR-10
    device = device or next(model.parameters()).device  # if no device was given, use the one the model is already on
    use_amp = device.type == "cuda"  # use faster half-precision maths only on a GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # the tool that updates the weights
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # slowly lowers the learning rate towards 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # stops very small numbers becoming 0 in half-precision maths
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
        print(f"[{desc}] Epoch {epoch + 1}/{epochs} | loss {tot / max(n, 1):.4f} | acc {100 * correct / max(n, 1):.2f}%")  # print this epoch's loss and accuracy
    return model.eval()  # give back the trained model in test mode
