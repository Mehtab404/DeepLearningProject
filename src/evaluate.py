"""Evaluation: measure how accurate each model is, with and without attacks."""
import os  # work with file and folder paths

import pandas as pd  # tables (like Excel) in Python
import torch  # main library for deep learning

from .attacks import fgsm_attack, pgd_attack  # the two attacks used for testing

EPS_LIST = [0, 1, 2, 4, 8, 16]  # attack strengths to test, in 1/255 units (0 = no attack = normal accuracy)
MODEL_ORDER = ["CNN-Clean", "ViT-Clean", "CNN-AdvTrained", "ViT-AdvTrained"]  # the 4 models, in the order we show them


@torch.no_grad()  # do not keep learning information here (faster, uses less memory)
def _correct(model, x, y):  # counts how many images the model guesses right
    return (model(x).argmax(1) == y).sum().item()  # the highest of the 10 scores is the guess; count guesses equal to the true label


def accuracy(model, X, Y, device, batch_size=100):  # accuracy in % of the model on images X with true labels Y
    model.eval()  # test mode
    c = 0  # number of right guesses so far
    for i in range(0, len(X), batch_size):  # go through the images in groups of 100 (so memory does not run out)
        x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)  # take one group and move it to the GPU/CPU
        c += _correct(model, x, y)  # add the right guesses of this group
    return 100.0 * c / len(X)  # right guesses divided by total images, as a percentage


def _attack_fn(name, eps, pgd_steps):  # gives back the attack function with the test settings filled in
    if name == "FGSM":  # the one-step attack
        return lambda m, x, y: fgsm_attack(m, x, y, eps)  # attack with strength eps
    if name == "PGD-10":  # the strong many-step attack
        return lambda m, x, y: pgd_attack(m, x, y, eps, alpha=max(eps / 4, 1 / 255), steps=pgd_steps,  # small steps of size eps/4 (at least 1/255), repeated pgd_steps times
                                          random_start=True)  # begin from a randomly shaken image
    raise ValueError(name)  # any other name is a mistake, so show an error


def evaluate_epsilon_sweep(models, X, Y, device, eps_list=EPS_LIST, pgd_steps=10, batch_size=100, seed=42):  # test every model against every attack at every strength
    rows = []  # one row of results for each combination
    for name, model in models.items():  # go through the 4 models
        model.eval()  # test mode
        for attack in ("FGSM", "PGD-10"):  # go through both attacks
            for e in eps_list:  # go through every attack strength
                torch.manual_seed(seed)  # fixed starting number so the random parts are the same every run
                eps = e / 255.0  # change e.g. 8 into 8/255, because pixel values are between 0 and 1
                fn = _attack_fn(attack, eps, pgd_steps)  # get the attack function for this attack and strength
                correct = 0  # right guesses so far
                for i in range(0, len(X), batch_size):  # go through the test images in groups
                    x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)  # take one group and move it to the GPU/CPU
                    x_adv = fn(model, x, y)  # step 1: trick the images (the attacker uses this same model)
                    correct += _correct(model, x_adv, y)  # step 2: count how many tricked images are still guessed right
                acc = 100.0 * correct / len(X)  # accuracy in percent
                rows.append({"model": name, "attack": attack, "eps_255": e, "accuracy": acc})  # save this result as one row
                print(f"[eval] {name:15s} {attack:6s} eps={e:2d}/255 -> {acc:5.1f}%")  # show the result on screen
    return pd.DataFrame(rows)  # turn all rows into one table


def evaluate_transferability(models, X, Y, device, eps=8 / 255, pgd_steps=10, batch_size=100, seed=42):  # does a trick made on one model also fool another model?
    names = list(models.keys())  # the 4 model names
    mat = pd.DataFrame(index=names, columns=names, dtype=float)  # empty 4x4 table: rows = model that made the trick, columns = model being tested
    for src in names:  # take each model in turn as the one that makes the tricked images
        torch.manual_seed(seed)  # fixed starting number for repeatable results
        models[src].eval()  # test mode
        adv_batches = []  # will hold the tricked images
        for i in range(0, len(X), batch_size):  # go through the test images in groups
            x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)  # take one group and move it to the GPU/CPU
            adv_batches.append(pgd_attack(models[src], x, y, eps, alpha=max(eps / 4, 1 / 255),  # trick the group using the PGD attack on the source model
                                          steps=pgd_steps, random_start=True).cpu())  # many small steps, random start; move result to the CPU to save GPU memory
        X_adv = torch.cat(adv_batches)  # join all groups into one set of tricked images
        for tgt in names:  # test the same tricked images on every model
            mat.loc[src, tgt] = accuracy(models[tgt], X_adv, Y, device, batch_size)  # store that model's accuracy in the table
        print(f"[transfer] source={src:15s} -> " + ", ".join(f"{t}: {mat.loc[src, t]:.1f}%" for t in names))  # show this row of the table on screen
    mat.index.name = "source \\ target"  # label the table corner: rows = source, columns = target
    return mat  # give back the finished table


def to_markdown(df, floatfmt="{:.1f}"):  # turns a table into text that GitHub can show as a table
    cols = [df.index.name or ""] + [str(c) for c in df.columns]  # the column titles
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]  # title line plus the "---" divider line
    for idx, row in df.iterrows():  # go through each row of the table
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row.values]  # row name, then each number with 1 decimal place
        lines.append("| " + " | ".join(cells) + " |")  # add this row as a text line
    return "\n".join(lines)  # join all lines into one block of text


def export_results(sweep_df, transfer_df, out_dir, note=""):  # saves all results as files (CSV and Markdown)
    os.makedirs(out_dir, exist_ok=True)  # create the output folder if it is missing
    sweep_df.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)  # save the long results list as a CSV file
    wide = sweep_df.pivot_table(index=["attack", "model"], columns="eps_255", values="accuracy", sort=False)  # reshape: one row per attack+model, one column per strength
    wide.columns = [f"eps={c}/255" for c in wide.columns]  # give the columns readable names
    wide.index = [f"{a}, {m}" for a, m in wide.index]  # join attack and model into one row name
    wide.index.name = "attack, model"  # name of the row-title column
    md = ["# Results summary (accuracy %, white-box)", ""]  # first lines of the Markdown report
    if note:  # if a note was given
        md += [f"> {note}", ""]  # add it near the top
    md += [to_markdown(wide), "", "## Transferability: PGD-10 at eps=8/255 (rows = source, cols = target; accuracy %)", "",  # add the main table, then the heading of the second table
           to_markdown(transfer_df), ""]  # add the second table (which model's tricks fool which model)
    with open(os.path.join(out_dir, "results_summary.md"), "w", encoding="utf-8") as f:  # open the Markdown file for writing
        f.write("\n".join(md))  # write the report into it
    transfer_df.to_csv(os.path.join(out_dir, "transferability.csv"))  # save the second table as a CSV file
    print("\n".join(md))  # also show the report on screen
