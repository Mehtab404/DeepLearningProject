"""Evaluation: measure how accurate each model is, with and without attacks."""
# WHAT: Imports file helpers, pandas tables, PyTorch and the two evaluation attacks.
# HOW: pandas builds the result tables that become CSV and Markdown.
# WHY: Evaluation produces tables.
# PROS & CONS: + Standard. - None.
# EXAMPLE: pd.DataFrame(rows).
import os  # work with file and folder paths

import pandas as pd  # tables (like Excel) in Python
import torch  # main library for deep learning

from .attacks import fgsm_attack, pgd_attack  # the two attacks used for testing

# WHAT: Attack strengths and model order used everywhere.
# HOW: eps values in 1/255 units; 0 means no attack.
# WHY: Same grid for every model, covering clean to twice the training budget.
# PROS & CONS: + Comparable curves. - Only 6 points per curve.
# EXAMPLE: [0, 1, 2, 4, 8, 16].
EPS_LIST = [0, 1, 2, 4, 8, 16]  # attack strengths to test, in 1/255 units (0 = no attack = normal accuracy)
MODEL_ORDER = ["CNN-Clean", "ViT-Clean", "CNN-AdvTrained", "ViT-AdvTrained"]  # the 4 models, in the order we show them


# WHAT: Counts correct predictions without tracking gradients.
# HOW: argmax of the 10 scores is the prediction.
# WHY: Fast and memory-light.
# PROS & CONS: + Efficient. - None.
# EXAMPLE: _correct(model, x, y) returns an integer.
@torch.no_grad()  # do not keep learning information here (faster, uses less memory)
def _correct(model, x, y):  # counts how many images the model guesses right
    return (model(x).argmax(1) == y).sum().item()  # the highest of the 10 scores is the guess; count guesses equal to the true label


# WHAT: Accuracy in percent on a set of images.
# HOW: Processes in batches of 100, sums correct predictions.
# WHY: Batches keep memory low.
# PROS & CONS: + Works for any size. - None.
# EXAMPLE: accuracy(model, X, Y, device) = 91.5.
def accuracy(model, X, Y, device, batch_size=100):  # accuracy in % of the model on images X with true labels Y
    model.eval()  # test mode
    c = 0  # number of right guesses so far
    for i in range(0, len(X), batch_size):  # go through the images in groups of 100 (so memory does not run out)
        x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)  # take one group and move it to the GPU/CPU
        c += _correct(model, x, y)  # add the right guesses of this group
    return 100.0 * c / len(X)  # right guesses divided by total images, as a percentage


# WHAT: Returns the attack function for a name and eps.
# HOW: FGSM or PGD-10 (alpha = max(eps/4, 1/255), random start).
# WHY: One place defines the evaluation attacks.
# PROS & CONS: + Consistent settings. - Only two attacks.
# EXAMPLE: _attack_fn('PGD-10', 8/255, 10).
def _attack_fn(name, eps, pgd_steps):  # gives back the attack function with the test settings filled in
    if name == "FGSM":  # the one-step attack
        return lambda m, x, y: fgsm_attack(m, x, y, eps)  # attack with strength eps
    if name == "PGD-10":  # the strong many-step attack
        return lambda m, x, y: pgd_attack(m, x, y, eps, alpha=max(eps / 4, 1 / 255), steps=pgd_steps,  # small steps of size eps/4 (at least 1/255), repeated pgd_steps times
                                          random_start=True)  # begin from a randomly shaken image
    raise ValueError(name)  # any other name is a mistake, so show an error


# WHAT: Accuracy of every model under every attack at every eps.
# HOW: For each model, attack and eps: fix the seed, attack each batch with the model itself (white-box), count
#      correct predictions.
# WHY: Produces the accuracy-versus-epsilon curves requested by the mentor.
# PROS & CONS: + Complete and repeatable. - Cost grows with models, eps values and PGD steps.
# EXAMPLE: Result: ViT-AdvTrained, PGD-10, eps = 8 -> 42.2%.
def evaluate_epsilon_sweep(models, X, Y, device, eps_list=EPS_LIST, pgd_steps=10, batch_size=100, seed=42):  # test every model against every attack at every strength
    # WHAT: Result list.
    # HOW: One dictionary per combination.
    # WHY: Becomes the table.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: []
    rows = []  # one row of results for each combination
    # WHAT: Loops over the 4 models.
    # HOW: Sets test mode.
    # WHY: Attack and evaluate each.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: CNN-Clean first.
    for name, model in models.items():  # go through the 4 models
        model.eval()  # test mode
        # WHAT: Loops over attacks and budgets.
        # HOW: Nested loops.
        # WHY: Full grid.
        # PROS & CONS: + Complete. - 48 evaluations.
        # EXAMPLE: FGSM then PGD-10.
        for attack in ("FGSM", "PGD-10"):  # go through both attacks
            for e in eps_list:  # go through every attack strength
                # WHAT: Fixes the random numbers before each cell of the grid.
                # HOW: Seed 42 each time.
                # WHY: Random starts become repeatable.
                # PROS & CONS: + Reproducible. - Same noise pattern for every cell.
                # EXAMPLE: Seed 42 is set before every (model, attack, eps) cell, so a re-run gives the same
                #          numbers.
                torch.manual_seed(seed)  # fixed starting number so the random parts are the same every run
                # WHAT: Converts eps to pixel units.
                # HOW: 8 becomes 0.0314.
                # WHY: Pixels live in [0, 1].
                # PROS & CONS: + Correct scale. - None.
                # EXAMPLE: 8 / 255 = 0.03137.
                eps = e / 255.0  # change e.g. 8 into 8/255, because pixel values are between 0 and 1
                # WHAT: Gets the attack function for this attack name and eps.
                # HOW: Calls _attack_fn, which returns a function (model, x, y) -> x_adv.
                # WHY: The batch loop can then call fn(model, x, y) without knowing which attack it is.
                # PROS & CONS: + One loop serves both attacks. - None.
                # EXAMPLE: fn = _attack_fn('PGD-10', 8/255, 10).
                fn = _attack_fn(attack, eps, pgd_steps)  # get the attack function for this attack and strength
                correct = 0  # right guesses so far
                # WHAT: Attacks the 1000 images in batches of 100.
                # HOW: x_adv = attack(model, x, y), then count the model's correct answers on x_adv.
                # WHY: Memory limits.
                # PROS & CONS: + Works on small GPUs. - None.
                # EXAMPLE: 10 batches.
                for i in range(0, len(X), batch_size):  # go through the test images in groups
                    x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)  # take one group and move it to the GPU/CPU
                    x_adv = fn(model, x, y)  # step 1: trick the images (the attacker uses this same model)
                    correct += _correct(model, x_adv, y)  # step 2: count how many tricked images are still guessed right
                # WHAT: Accuracy in percent.
                # HOW: correct / N.
                # WHY: Report format.
                # PROS & CONS: + Direct. - Standard error up to 1.6 points for N = 1000.
                # EXAMPLE: 422 correct out of 1000 = 42.2%.
                acc = 100.0 * correct / len(X)  # accuracy in percent
                # WHAT: Saves and prints the result.
                # HOW: Dictionary row plus a console line.
                # WHY: Feeds the CSV and plots.
                # PROS & CONS: + Visible. - None.
                # EXAMPLE: [eval] ViT-AdvTrained PGD-10 eps=8/255 -> 42.2%
                rows.append({"model": name, "attack": attack, "eps_255": e, "accuracy": acc})  # save this result as one row
                print(f"[eval] {name:15s} {attack:6s} eps={e:2d}/255 -> {acc:5.1f}%")  # show the result on screen
    # WHAT: Returns all results as a table.
    # HOW: pandas DataFrame.
    # WHY: Easy to save and plot.
    # PROS & CONS: + Convenient. - None.
    # EXAMPLE: 48 rows.
    return pd.DataFrame(rows)  # turn all rows into one table


# WHAT: Transfer test: craft PGD-10 examples on a source model, test them on all target models.
# HOW: 4x4 matrix: row = source, column = target; the diagonal is white-box.
# WHY: Shows whether attacks transfer across architectures and from clean to defended models.
# PROS & CONS: + Black-box view of robustness. - Only eps = 8/255 and PGD-10.
# EXAMPLE: CNN-Clean to ViT-Clean: 85.4% remains.
def evaluate_transferability(models, X, Y, device, eps=8 / 255, pgd_steps=10, batch_size=100, seed=42):  # does a trick made on one model also fool another model?
    # WHAT: Empty result matrix.
    # HOW: pandas DataFrame with model names on both axes.
    # WHY: Filled in the loop.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: 4x4.
    names = list(models.keys())  # the 4 model names
    mat = pd.DataFrame(index=names, columns=names, dtype=float)  # empty 4x4 table: rows = model that made the trick, columns = model being tested
    # WHAT: Generates adversarial images from each source model.
    # HOW: PGD-10 on every batch; moved to CPU to save GPU memory.
    # WHY: These images are reused for all targets.
    # PROS & CONS: + One attack per source. - Stores 1000 adversarial images on the CPU.
    # EXAMPLE: (1000, 3, 32, 32).
    for src in names:  # take each model in turn as the one that makes the tricked images
        torch.manual_seed(seed)  # fixed starting number for repeatable results
        models[src].eval()  # test mode
        adv_batches = []  # will hold the tricked images
        for i in range(0, len(X), batch_size):  # go through the test images in groups
            x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)  # take one group and move it to the GPU/CPU
            adv_batches.append(pgd_attack(models[src], x, y, eps, alpha=max(eps / 4, 1 / 255),  # trick the group using the PGD attack on the source model
                                          steps=pgd_steps, random_start=True).cpu())  # many small steps, random start; move result to the CPU to save GPU memory
        X_adv = torch.cat(adv_batches)  # join all groups into one set of tricked images
        # WHAT: Tests every target model on the same adversarial images.
        # HOW: accuracy(target, X_adv, Y).
        # WHY: Fills one matrix row.
        # PROS & CONS: + Fair comparison. - None.
        # EXAMPLE: Row CNN-Clean: 0.0, 85.4, 69.2, 87.1.
        for tgt in names:  # test the same tricked images on every model
            mat.loc[src, tgt] = accuracy(models[tgt], X_adv, Y, device, batch_size)  # store that model's accuracy in the table
        print(f"[transfer] source={src:15s} -> " + ", ".join(f"{t}: {mat.loc[src, t]:.1f}%" for t in names))  # show this row of the table on screen
    # WHAT: Label and result.
    # HOW: Names the table corner and returns it.
    # WHY: Readable CSV.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: 'source \ target'.
    mat.index.name = "source \\ target"  # label the table corner: rows = source, columns = target
    return mat  # give back the finished table


# WHAT: Converts a table to Markdown text.
# HOW: Builds header, divider and one line per row with 1-decimal numbers.
# WHY: GitHub renders Markdown tables.
# PROS & CONS: + No extra library. - Very small formatter.
# EXAMPLE: | CNN-Clean | 91.5 | ...
def to_markdown(df, floatfmt="{:.1f}"):  # turns a table into text that GitHub can show as a table
    cols = [df.index.name or ""] + [str(c) for c in df.columns]  # the column titles
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]  # title line plus the "---" divider line
    for idx, row in df.iterrows():  # go through each row of the table
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row.values]  # row name, then each number with 1 decimal place
        lines.append("| " + " | ".join(cells) + " |")  # add this row as a text line
    return "\n".join(lines)  # join all lines into one block of text


# WHAT: Writes all result files.
# HOW: results_summary.csv, results_summary.md and transferability.csv.
# WHY: Tracked in git so the tables are visible on GitHub.
# PROS & CONS: + Reproducible outputs. - Overwrites previous files.
# EXAMPLE: outputs/results_summary.md.
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
