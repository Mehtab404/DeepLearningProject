"""Pictures: draws the 4 result graphs and saves them as image files."""
# WHAT: Imports the path helper used to create the output folder.
# HOW: os.makedirs and os.path.join build the PNG file paths.
# WHY: The plots are saved as files, never shown in a window.
# PROS & CONS: + Standard library. - None.
# EXAMPLE: os.path.join('outputs', 'acc_vs_epsilon.png').
import os  # work with file and folder paths

# WHAT: Imports matplotlib and selects the 'Agg' backend.
# HOW: Agg renders straight into image files with no display window.
# WHY: Colab, Kaggle and servers have no screen; a GUI backend would crash.
# PROS & CONS: + Works anywhere. - No interactive zoom.
# EXAMPLE: fig.savefig('x.png') works on a headless machine.
import matplotlib  # drawing library
matplotlib.use("Agg")  # draw into files only (no pop-up window), so it also works on a server
# WHAT: Imports plotting, array, colour-table and PyTorch libraries.
# HOW: pyplot draws figures, numpy positions bars, seaborn draws the heat map, torch runs the attack for the
#      example figure.
# WHY: Each library covers one part of the figures.
# PROS & CONS: + Mature tools. - seaborn is only needed for one figure.
# EXAMPLE: sns.heatmap(matrix, annot=True).
import matplotlib.pyplot as plt  # the drawing commands
import numpy as np  # fast number lists
import seaborn as sns  # nicer-looking graphs (we use it for the colour table)
import torch  # main library for deep learning

# WHAT: Imports the PGD attack and the class names.
# HOW: Relative imports from the same package.
# WHY: The example figure creates fresh adversarial images; the class names label them.
# PROS & CONS: + Reuses tested code. - None.
# EXAMPLE: CIFAR10_CLASSES[3] == 'cat'.
from .attacks import pgd_attack  # the strong attack, used to make the example pictures
from .models import CIFAR10_CLASSES  # names of the 10 picture types

# WHAT: Picture sharpness and model colours.
# HOW: 300 dots per inch; blue for the CNN, red for the ViT.
# WHY: 300 DPI is print quality for the report; fixed colours make the figures easy to read.
# PROS & CONS: + Consistent look. - Larger files (200-350 KB per PNG).
# EXAMPLE: COLORS['ViT'] == '#d62728'.
DPI = 300  # picture sharpness (300 = print quality)
COLORS = {"CNN": "#1f77b4", "ViT": "#d62728"}  # blue for CNN, red for ViT


# WHAT: Chooses colour, line style and marker for a model name.
# HOW: The part before '-' gives the architecture; 'Adv' in the name gives a dashed line with squares.
# WHY: Solid circles = clean model, dashed squares = adversarially trained model.
# PROS & CONS: + Visual meaning is consistent in all plots. - Relies on the naming convention.
# EXAMPLE: _style('ViT-AdvTrained') returns ('#d62728', '--', 's').
def _style(name):  # picks line colour, line style and marker shape for a model name
    # WHAT: Extracts the architecture from the model name.
    # HOW: split('-')[0].
    # WHY: Used as a key into COLORS.
    # PROS & CONS: + Simple. - Breaks if names change.
    # EXAMPLE: 'CNN-Clean' becomes 'CNN'.
    arch = name.split("-")[0]  # first part of the name: "CNN" or "ViT"
    # WHAT: Returns the style triple.
    # HOW: colour, linestyle, marker.
    # WHY: Consumed by ax.plot.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: ('#1f77b4', '-', 'o').
    return COLORS[arch], ("--" if "Adv" in name else "-"), ("s" if "Adv" in name else "o")  # colour by type; dashed line + square for defended models, solid line + circle for normal ones


# WHAT: Adds a small grey note at the bottom of a figure.
# HOW: fig.text at the bottom centre when a tag is given.
# WHY: Marks quick-test figures as 'not a result' so they cannot be mistaken for real ones.
# PROS & CONS: + Prevents misuse. - None.
# EXAMPLE: 'QUICK-TEST smoke run ... not a result'.
def _tag(fig, tag):  # writes a small grey note at the bottom of a graph
    if tag:  # only if a note was given (used to mark quick-test graphs as "not a real result")
        fig.text(0.5, 0.005, tag, ha="center", va="bottom", fontsize=8, color="gray")  # place the note in the bottom middle


# WHAT: Figure 1: accuracy versus attack strength, FGSM on the left and PGD-10 on the right.
# HOW: For every model draw one line (eps on x, accuracy on y). Solid = clean, dashed = adversarially trained.
# WHY: This is the accuracy-versus-epsilon curve the mentor asked for.
# PROS & CONS: + Shows all four models in one view. - No error bars (standard error up to 1.6 points).
# EXAMPLE: outputs/acc_vs_epsilon.png.
def plot_acc_vs_epsilon(df, path, tag=""):  # graph 1: accuracy as the attack gets stronger (FGSM on the left, PGD-10 on the right)
    # WHAT: Creates two side-by-side panels with a shared y axis.
    # HOW: 1 row, 2 columns, 12 by 4.6 inches.
    # WHY: Sharing the axis makes FGSM and PGD directly comparable.
    # PROS & CONS: + Easy comparison. - Wide figure.
    # EXAMPLE: axes[0] = FGSM, axes[1] = PGD-10.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)  # a canvas with two graphs side by side that share the up-down scale
    # WHAT: Loops over the two attacks.
    # HOW: One panel per attack.
    # WHY: Same drawing code for both.
    # PROS & CONS: + No duplication. - None.
    # EXAMPLE: ('FGSM', 'PGD-10').
    for ax, attack in zip(axes, ("FGSM", "PGD-10")):  # one graph for each attack
        # WHAT: Loops over the four models.
        # HOW: One line per model.
        # WHY: All models in each panel.
        # PROS & CONS: + Complete. - Four lines can overlap at eps = 0.
        # EXAMPLE: 'CNN-Clean', 'ViT-Clean', ...
        for name in df["model"].unique():  # one line for each model
            # WHAT: Selects the rows for this model and attack, sorted by eps.
            # HOW: Boolean filter, then sort_values('eps_255').
            # WHY: Lines must be drawn from small to large eps.
            # PROS & CONS: + Correct curve order. - None.
            # EXAMPLE: 6 rows (eps 0, 1, 2, 4, 8, 16).
            d = df[(df.model == name) & (df.attack == attack)].sort_values("eps_255")  # keep this model's rows for this attack, ordered by strength
            # WHAT: Gets the style for this model.
            # HOW: Calls _style.
            # WHY: Consistent colours.
            # PROS & CONS: + Consistent. - None.
            # EXAMPLE: ('#1f77b4', '-', 'o').
            c, ls, mk = _style(name)  # colour, line style and marker for this model
            # WHAT: Draws the line.
            # HOW: x = eps in /255 units, y = accuracy in percent, with markers.
            # WHY: Shows how accuracy collapses as the attack gets stronger.
            # PROS & CONS: + Clear trend. - Markers hide values at eps = 0 where all models start high.
            # EXAMPLE: (8, 42.2) is one point of the ViT-AdvTrained PGD-10 curve.
            ax.plot(d.eps_255, d.accuracy, color=c, linestyle=ls, marker=mk, linewidth=2, label=name)  # draw the line: attack strength across, accuracy up
        # WHAT: Titles, axis label, tick marks and grid of each panel.
        # HOW: x ticks sit exactly at the tested eps values.
        # WHY: Readers can find each tested budget.
        # PROS & CONS: + Readable. - Uneven spacing near 0.
        # EXAMPLE: 'FGSM attack'.
        ax.set_title(f"{attack} attack")  # graph title
        ax.set_xlabel(r"$\epsilon$ (in /255 units)")  # label under the graph: attack strength
        ax.set_xticks(sorted(df.eps_255.unique()))  # show a mark for each strength we tested
        ax.grid(alpha=0.3)  # faint grid lines
    # WHAT: Y axis label and range on the left panel.
    # HOW: Range fixed to 0..100 percent.
    # WHY: Honest, comparable scale.
    # PROS & CONS: + No exaggeration. - Small differences are hard to see.
    # EXAMPLE: 'Accuracy (%)'.
    axes[0].set_ylabel("Accuracy (%)")  # label on the left side
    axes[0].set_ylim(0, 100)  # the up-down scale goes from 0 to 100 percent
    # WHAT: Legend with the four model names.
    # HOW: Placed upper right of the PGD panel.
    # WHY: Explains the colours and line styles.
    # PROS & CONS: + Needed. - May cover points.
    # EXAMPLE: 'CNN-Clean', 'ViT-Clean', ...
    axes[1].legend(loc="upper right")  # box that says which line is which model
    # WHAT: Overall title.
    # HOW: States that solid = clean and dashed = adversarially fine-tuned.
    # WHY: Self-explanatory figure.
    # PROS & CONS: + Standalone. - Long title.
    # EXAMPLE: Accuracy vs. perturbation budget on CIFAR-10 ...
    fig.suptitle("Accuracy vs. perturbation budget on CIFAR-10 (solid = clean, dashed = adversarially fine-tuned)")  # main title
    # WHAT: Finishes and saves the figure.
    # HOW: Fit the layout, add the optional note, write the PNG at 300 DPI, close the figure.
    # WHY: close() frees memory; tight_layout avoids clipped labels.
    # PROS & CONS: + Clean output. - None.
    # EXAMPLE: outputs/acc_vs_epsilon.png (3600x1380 pixels).
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # fit everything neatly, leaving room at the bottom
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


# WHAT: Looks up one accuracy value.
# HOW: Filters the result table by model, attack and eps and returns the first value as a float.
# WHY: Used by the trade-off figure.
# PROS & CONS: + Simple. - Raises an error if the row is missing.
# EXAMPLE: _acc(df, 'ViT-Clean', 'PGD-10', 8) == 0.0.
def _acc(df, model, attack, eps):  # looks up one accuracy value in the results table
    # WHAT: The filter itself.
    # HOW: Three equality conditions combined with &.
    # WHY: Exactly one row matches.
    # PROS & CONS: + Precise. - None.
    # EXAMPLE: 0.0
    return float(df[(df.model == model) & (df.attack == attack) & (df.eps_255 == eps)].accuracy.iloc[0])  # find the row for this model, attack and strength; take its accuracy


# WHAT: Figure 2: what adversarial training costs and gains.
# HOW: Grouped bars (before / after) for clean, FGSM (8/255) and PGD-10 (8/255) accuracy, with the change in
#      percentage points written above.
# WHY: Answers RQ2: cost of adversarial training on clean accuracy versus robustness gain.
# PROS & CONS: + Cost and gain visible together. - Only the 8/255 budget.
# EXAMPLE: CNN: -3.1 pp clean, +18.3 pp PGD-10.
def plot_tradeoff(df, path, tag=""):  # graph 2: what defence training costs (normal accuracy) and gains (accuracy under attack)
    # WHAT: The three measurements shown.
    # HOW: (label, attack, eps). FGSM at eps 0 equals clean accuracy.
    # WHY: A clean number needs no extra evaluation.
    # PROS & CONS: + Reuses existing data. - Relies on FGSM(0) == clean.
    # EXAMPLE: ('Clean (eps=0)', 'FGSM', 0).
    metrics = [("Clean (eps=0)", "FGSM", 0), ("FGSM (eps=8/255)", "FGSM", 8), ("PGD-10 (eps=8/255)", "PGD-10", 8)]  # three tests: no attack, FGSM, PGD-10 (FGSM with eps=0 is just normal accuracy)
    # WHAT: Two panels: CNN and ViT.
    # HOW: Shared y axis.
    # WHY: Same scale for both architectures.
    # PROS & CONS: + Comparable. - None.
    # EXAMPLE: axes[0] = CNN.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)  # a canvas with two graphs side by side: CNN and ViT
    # WHAT: Loops over the two architectures.
    # HOW: One panel each.
    # WHY: Same drawing code.
    # PROS & CONS: + No duplication. - None.
    # EXAMPLE: ('CNN', 'ViT').
    for ax, arch in zip(axes, ("CNN", "ViT")):  # one graph for each model type
        # WHAT: Collects the accuracies before and after defence training.
        # HOW: Lookups of the Clean and AdvTrained models.
        # WHY: Bars compare the two.
        # PROS & CONS: + Clear. - None.
        # EXAMPLE: CNN before: [91.5, 11.9, 0.0].
        before = [_acc(df, f"{arch}-Clean", a, e) for _, a, e in metrics]  # accuracies of the normal model
        after = [_acc(df, f"{arch}-AdvTrained", a, e) for _, a, e in metrics]  # accuracies of the defended model
        # WHAT: Bar positions and width.
        # HOW: Positions 0, 1, 2; each bar 0.36 wide.
        # WHY: Two bars side by side per group.
        # PROS & CONS: + Neat. - None.
        # EXAMPLE: [0, 1, 2].
        xs = np.arange(len(metrics))  # positions of the bar groups: 0, 1, 2
        w = 0.36  # width of one bar
        # WHAT: Draws the 'before' bars.
        # HOW: Light colour, shifted left.
        # WHY: Light = clean model.
        # PROS & CONS: + Clear. - None.
        # EXAMPLE: Light blue for the CNN.
        ax.bar(xs - w / 2, before, w, label="Before (clean model)", color="#9ecae1" if arch == "CNN" else "#fcae91")  # light bars = before defence training
        # WHAT: Draws the 'after' bars.
        # HOW: Strong colour, shifted right.
        # WHY: Strong colour = defended model.
        # PROS & CONS: + Clear. - None.
        # EXAMPLE: Dark blue for the CNN.
        ax.bar(xs + w / 2, after, w, label="After 10-epoch adv. fine-tuning", color=COLORS[arch])  # strong-colour bars = after defence training
        # WHAT: Annotates each group.
        # HOW: Writes both values and the change.
        # WHY: Numbers are easier to read than bar heights.
        # PROS & CONS: + Precise. - Text can overlap in a crowded plot.
        # EXAMPLE: 'Delta Acc = +42.2 pp' style label.
        for i in range(len(metrics)):  # go through the three tests
            # WHAT: Change in percentage points.
            # HOW: Subtract before from after.
            # WHY: Quantifies the trade-off.
            # PROS & CONS: + Clear. - Absolute points, not relative change.
            # EXAMPLE: 94.1 - 97.1 = -3.0.
            delta = after[i] - before[i]  # change in accuracy (in percentage points)
            # WHAT: Writes the two values above the bars.
            # HOW: One decimal place.
            # WHY: Reading exact numbers.
            # PROS & CONS: + Precise. - None.
            # EXAMPLE: '97.1' and '94.1'.
            ax.text(xs[i] - w / 2, before[i] + 1, f"{before[i]:.1f}", ha="center", fontsize=8)  # write the "before" number above its bar
            ax.text(xs[i] + w / 2, after[i] + 1, f"{after[i]:.1f}", ha="center", fontsize=8)  # write the "after" number above its bar
            # WHAT: Chooses the label for the change.
            # HOW: Clean change for the first group, robust change for the others.
            # WHY: Different meaning.
            # PROS & CONS: + Clear. - None.
            # EXAMPLE: 'Delta Acc_clean' vs 'Delta Acc_robust'.
            label = r"$\Delta$Acc$_{clean}$" if i == 0 else r"$\Delta$Acc$_{robust}$"  # name of the change: normal-accuracy change or attacked-accuracy change
            # WHAT: Writes the change above each group.
            # HOW: Green if accuracy rose, red if it fell.
            # WHY: Colour tells the direction at a glance.
            # PROS & CONS: + Quick reading. - None.
            # EXAMPLE: Green '+42.2 pp'.
            ax.text(xs[i], max(before[i], after[i]) + 8, f"{label} = {delta:+.1f} pp", ha="center", fontsize=9,  # write the change above the bars
                    fontweight="bold", color="darkgreen" if delta >= 0 else "darkred")  # bold; green if accuracy went up, red if it went down
        # WHAT: Group names on the x axis.
        # HOW: One label per group.
        # WHY: Shows what is measured.
        # PROS & CONS: + Clear. - None.
        # EXAMPLE: 'PGD-10 (eps=8/255)'.
        ax.set_xticks(xs)  # one mark per bar group
        ax.set_xticklabels([m[0] for m in metrics])  # name each group
        # WHAT: Title and y range of each panel.
        # HOW: Range 0 to 115 leaves room for text.
        # WHY: Prevents text clipping.
        # PROS & CONS: + Readable. - Axis above 100.
        # EXAMPLE: 'ViT: cost of adversarial training vs. robustness gain'.
        ax.set_title(f"{arch}: cost of adversarial training vs. robustness gain")  # graph title
        ax.set_ylim(0, 115)  # the up-down scale (a bit above 100 to leave room for the text)
        ax.grid(axis="y", alpha=0.3)  # faint horizontal grid lines
        ax.legend(loc="upper right", fontsize=8)  # box that says which bar is which
    # WHAT: Y label and finishing steps.
    # HOW: Finishes and saves the figure.
    # WHY: tight_layout fits the labels, _tag writes the optional note, savefig writes the PNG at 300 DPI, close
    #      frees the figure.
    # PROS & CONS: + Clean output. - None.
    # EXAMPLE: outputs/clean_vs_robust_tradeoff.png.
    axes[0].set_ylabel("Accuracy (%)")  # label on the left side
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # fit everything neatly
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


# WHAT: Figure 3: heat map of the 4x4 transferability matrix.
# HOW: Colour from red (0%, strong attack) to green (100%, attack fails), with the number in each cell.
# WHY: Shows at a glance which source model fools which target model.
# PROS & CONS: + Easy pattern spotting. - Cell colours exaggerate small differences.
# EXAMPLE: Cell (CNN-Clean, ViT-Clean) = 85.4.
def plot_transferability(matrix, path, tag=""):  # graph 3: colour table showing which model's tricks fool which model
    # WHAT: One canvas.
    # HOW: 7.5 by 6 inches.
    # WHY: One heat map.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: One 7.5 x 6 inch canvas holds the whole 4x4 heat map.
    fig, ax = plt.subplots(figsize=(7.5, 6))  # a canvas for one graph
    # WHAT: Draws the table as coloured squares.
    # HOW: annot=True writes values; colour range fixed to 0..100.
    # WHY: Fixed range keeps colours comparable.
    # PROS & CONS: + Quick to read. - Needs seaborn.
    # EXAMPLE: 0.0 is dark red.
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax,  # coloured squares with the number written inside; red = low accuracy, green = high
                cbar_kws={"label": "Target accuracy (%) - lower = stronger attack"})  # label of the colour bar
    # WHAT: Axis labels and title.
    # HOW: Rows = source (crafted on), columns = target (evaluated on).
    # WHY: A reader must know the direction.
    # PROS & CONS: + Clear. - Long label.
    # EXAMPLE: 'Source model (adversarial examples crafted on)'.
    ax.set_xlabel("Target model (evaluated on)")  # label under the table: the model being tested
    ax.set_ylabel("Source model (adversarial examples crafted on)")  # label on the left: the model that made the tricked images
    ax.set_title(r"Transferability of PGD-10 examples ($\epsilon$ = 8/255); diagonal = white-box")  # title; the diagonal is where the same model is both attacked and tested
    # WHAT: Tilts the x labels.
    # HOW: 20 degrees, right aligned.
    # WHY: Long names would overlap.
    # PROS & CONS: + Readable. - Slight crowding of the y label.
    # EXAMPLE: 'CNN-AdvTrained' is written at a 20-degree angle.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")  # tilt the names under the table so they do not overlap
    # WHAT: Layout, optional note, save, close.
    # HOW: tight_layout fits the labels, _tag writes the optional note, savefig writes the PNG at 300 DPI, close
    #      frees the figure.
    # WHY: Clipped labels and leaked figures are the usual matplotlib problems in long runs; the note stops
    #      quick-test images being mistaken for results.
    # PROS & CONS: + Clean output. - None.
    # EXAMPLE: outputs/transferability_heatmap.png.
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # fit everything neatly
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


# WHAT: Figure 4: real pictures, the (magnified) noise and the adversarial pictures.
# HOW: Picks one correctly classified test image per class, attacks them with PGD-10 on the source model, and
#      shows what all four models predict.
# WHY: Shows that the attack is invisible but effective.
# PROS & CONS: + Intuitive evidence. - Only 4 examples; an illustration, not evidence.
# EXAMPLE: frog classified as dog by CNN-Clean.
def plot_adversarial_examples(models, X, Y, device, path, eps=8 / 255, source="CNN-Clean", n=4,  # graph 4: shows real pictures, the added noise, and the tricked pictures
                              pgd_steps=10, tag=""):  # pgd_steps = how many small steps the attack takes
    # WHAT: Chooses the model that creates the adversarial images.
    # HOW: Default CNN-Clean, in test mode.
    # WHY: A single source keeps the figure simple.
    # PROS & CONS: + Simple. - Examples only show attacks on the CNN.
    # EXAMPLE: source='CNN-Clean'.
    src = models[source].eval()  # the model that makes the tricked images, in test mode
    # WHAT: Bookkeeping for picking pictures.
    # HOW: chosen = indices; seen = classes already picked.
    # WHY: Pick one picture per class.
    # PROS & CONS: + Variety. - None.
    # EXAMPLE: [] and set().
    chosen, seen = [], set()  # chosen = picked picture positions; seen = classes already picked
    # WHAT: Searches for suitable pictures without gradients.
    # HOW: Loop over the test images.
    # WHY: Only predictions needed.
    # PROS & CONS: + Fast. - None.
    # EXAMPLE: Predicting 1000 images without gradients builds no computation graph.
    with torch.no_grad():  # no learning information needed here
        # WHAT: Goes through the test images.
        # HOW: Stops when n pictures are chosen.
        # WHY: Finds n different classes.
        # PROS & CONS: + Cheap. - None.
        # EXAMPLE: Stops after 4 pictures.
        for i in range(len(X)):  # go through the test pictures
            # WHAT: True label of this picture.
            # HOW: Convert tensor to int.
            # WHY: Needed for the class check.
            # PROS & CONS: + Simple. - None.
            # EXAMPLE: 6 for frog.
            y = int(Y[i])  # the true label of this picture
            # WHAT: Skips classes already picked.
            # HOW: Set membership.
            # WHY: One picture per class.
            # PROS & CONS: + Variety. - None.
            # EXAMPLE: continue
            if y in seen:  # we already have a picture of this class
                continue  # skip it
            # WHAT: Keeps only pictures the model classifies correctly.
            # HOW: argmax equals the true label.
            # WHY: An attack on an already wrong picture shows nothing.
            # PROS & CONS: + Meaningful examples. - Cherry-picked on purpose; the text says so.
            # EXAMPLE: append(i).
            if src(X[i:i + 1].to(device)).argmax(1).item() == y:  # the model guesses this picture right (an attack only makes sense then)
                chosen.append(i)  # pick it
                seen.add(y)  # remember this class is done
            if len(chosen) == n:  # we have enough pictures
                break  # stop looking
    # WHAT: Fallback when no picture qualifies.
    # HOW: Take the first few pictures.
    # WHY: Never crash.
    # PROS & CONS: + Robust. - Less meaningful.
    # EXAMPLE: Only possible in tiny tests.
    if not chosen:  # nothing suitable was found
        chosen = list(range(min(n, len(X))))  # just take the first few pictures
    # WHAT: Moves the chosen pictures and labels to the device.
    # HOW: Index and copy.
    # WHY: The attack runs on the GPU.
    # PROS & CONS: + Fast. - None.
    # EXAMPLE: (4, 3, 32, 32).
    x = X[chosen].to(device)  # the chosen pictures, on the GPU/CPU
    y = Y[chosen].to(device)  # their true labels
    # WHAT: Fixes the random start.
    # HOW: Seed 42.
    # WHY: Same figure every run.
    # PROS & CONS: + Repeatable. - None.
    # EXAMPLE: The random start is identical each run, so the noise picture does not change.
    torch.manual_seed(42)  # fixed starting number so the random start is the same every time
    # WHAT: Creates the adversarial pictures.
    # HOW: PGD-10, eps = 8/255, random start.
    # WHY: Same attack as the evaluation.
    # PROS & CONS: + Consistent. - None.
    # EXAMPLE: (4, 3, 32, 32).
    x_adv = pgd_attack(src, x, y, eps, alpha=max(eps / 4, 1 / 255), steps=pgd_steps, random_start=True)  # make the tricked version of each picture
    # WHAT: Makes the noise visible.
    # HOW: (x_adv - x + eps) / (2 eps) maps [-eps, +eps] to [0, 1].
    # WHY: The real noise is only 8/255 and would look invisible.
    # PROS & CONS: + Visible. - Exaggerated (magnified).
    # EXAMPLE: 0 change maps to grey 0.5.
    pert = ((x_adv - x + eps) / (2 * eps)).clamp(0, 1)  # the real noise is too small to see, so stretch it to the range 0..1 (grey = no change)
    # WHAT: Collects every model's prediction.
    # HOW: Dictionary name to predictions.
    # WHY: Printed under each picture.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: {}
    preds = {}  # what each model guesses for the tricked pictures
    # WHAT: Predicts the adversarial pictures with all four models.
    # HOW: argmax of the 10 scores.
    # WHY: Shows who is fooled.
    # PROS & CONS: + Direct. - None.
    # EXAMPLE: CNN-Clean: dog [WRONG].
    with torch.no_grad():  # no learning information needed here
        for name, m in models.items():  # go through all the models
            preds[name] = m.eval()(x_adv).argmax(1).cpu().tolist()  # the guess of this model for each tricked picture
    # WHAT: One row of the figure per picture.
    # HOW: Count of chosen pictures.
    # WHY: Layout.
    # PROS & CONS: + Simple. - None.
    # EXAMPLE: 4
    rows = len(chosen)  # one row of the graph per picture
    # WHAT: Canvas with 3 columns.
    # HOW: Original, noise, adversarial.
    # WHY: Story: original, noise, result.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: (4, 3) axes.
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, 3.1 * rows))  # a canvas with 3 columns: original, noise, tricked picture
    # WHAT: Makes axes 2-D even for one row.
    # HOW: numpy.
    # WHY: Same indexing code.
    # PROS & CONS: + Robust. - None.
    # EXAMPLE: axes[0, 0].
    axes = np.atleast_2d(axes)  # makes sure indexing works even when there is only one row
    # WHAT: Fills one row per picture.
    # HOW: Three images and titles.
    # WHY: Presentation.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: 4 rows.
    for r in range(rows):  # go through each picture
        # WHAT: Draws the original picture.
        # HOW: Channel order changed to height x width x colour.
        # WHY: matplotlib expects that order.
        # PROS & CONS: + Correct colours. - None.
        # EXAMPLE: Title 'Clean: frog'.
        axes[r, 0].imshow(x[r].permute(1, 2, 0).cpu().numpy())  # show the original picture (colour layers moved to the order the drawing tool wants)
        axes[r, 0].set_title(f"Clean: {CIFAR10_CLASSES[int(y[r])]}", fontsize=10)  # title: its true class name
        # WHAT: Draws the magnified noise.
        # HOW: Same layout.
        # WHY: Shows what the attack added.
        # PROS & CONS: + Clear. - Magnified.
        # EXAMPLE: Colourful noise.
        axes[r, 1].imshow(pert[r].permute(1, 2, 0).cpu().numpy())  # show the stretched noise
        axes[r, 1].set_title(r"PGD-10 perturbation, magnified $(x_{adv}-x+\epsilon)/2\epsilon$", fontsize=8)  # title: this is the magnified noise
        # WHAT: Draws the adversarial picture.
        # HOW: Detach from autograd first.
        # WHY: Shows it looks unchanged.
        # PROS & CONS: + Clear. - None.
        # EXAMPLE: Same frog.
        axes[r, 2].imshow(x_adv[r].permute(1, 2, 0).detach().cpu().numpy())  # show the tricked picture
        # WHAT: Prediction text for the title.
        # HOW: One line per model with OK or WRONG.
        # WHY: Readers see who is fooled.
        # PROS & CONS: + Informative. - Small font.
        # EXAMPLE: 'CNN-Clean: dog [WRONG]'.
        lines = []  # text lines with each model's guess
        for name in models:  # go through all the models
            p = preds[name][r]  # this model's guess
            ok = "OK" if p == int(y[r]) else "WRONG"  # is the guess right?
            lines.append(f"{name}: {CIFAR10_CLASSES[p]} [{ok}]")  # write: model name, guessed class, right or wrong
        axes[r, 2].set_title(f"x_adv (eps=8/255, crafted on {source})\n" + "\n".join(lines), fontsize=7)  # title: all models' guesses
        # WHAT: Hides the axes.
        # HOW: axis('off').
        # WHY: Cleaner pictures.
        # PROS & CONS: + Clean. - None.
        # EXAMPLE: axes[r, 0].axis('off') removes the tick numbers around the first picture.
        for c in range(3):  # go through the 3 columns
            axes[r, c].axis("off")  # hide the numbers around the picture
    # WHAT: Layout, optional note, save, close.
    # HOW: tight_layout fits the labels, _tag writes the optional note, savefig writes the PNG at 300 DPI, close
    #      frees the figure.
    # WHY: Clipped labels and leaked figures are the usual matplotlib problems in long runs; the note stops
    #      quick-test images being mistaken for results.
    # PROS & CONS: + Clean output. - None.
    # EXAMPLE: outputs/adversarial_examples_grid.png.
    fig.tight_layout(rect=(0, 0.02, 1, 1))  # fit everything neatly
    _tag(fig, tag)  # add the small grey note if there is one
    fig.savefig(path, dpi=DPI)  # save the graph as an image file
    plt.close(fig)  # release the memory


# WHAT: Draws all four figures and returns their paths.
# HOW: Calls the four plot functions in turn.
# WHY: One call from main.py.
# PROS & CONS: + Convenient. - None.
# EXAMPLE: dict of four paths.
def make_all_plots(sweep_df, transfer_df, models, X, Y, device, out_dir, pgd_steps=10, tag=""):  # draws all four graphs and gives back their file paths
    # WHAT: Creates the output folder.
    # HOW: exist_ok.
    # WHY: Avoid errors.
    # PROS & CONS: + Safe. - None.
    # EXAMPLE: outputs/
    os.makedirs(out_dir, exist_ok=True)  # create the output folder if it is missing
    # WHAT: File names for the four figures.
    # HOW: Dictionary.
    # WHY: Same names as in the TASK list.
    # PROS & CONS: + Predictable. - None.
    # EXAMPLE: outputs/acc_vs_epsilon.png
    paths = {  # file name for each graph
        "acc_vs_epsilon": os.path.join(out_dir, "acc_vs_epsilon.png"),  # graph 1
        "tradeoff": os.path.join(out_dir, "clean_vs_robust_tradeoff.png"),  # graph 2
        "transfer": os.path.join(out_dir, "transferability_heatmap.png"),  # graph 3
        "examples": os.path.join(out_dir, "adversarial_examples_grid.png"),  # graph 4
    }
    # WHAT: Draws the four figures.
    # HOW: One function call per figure.
    # WHY: Each saves its own PNG.
    # PROS & CONS: + Clear. - None.
    # EXAMPLE: 4 PNG files.
    plot_acc_vs_epsilon(sweep_df, paths["acc_vs_epsilon"], tag)  # draw graph 1
    plot_tradeoff(sweep_df, paths["tradeoff"], tag)  # draw graph 2
    plot_transferability(transfer_df, paths["transfer"], tag)  # draw graph 3
    plot_adversarial_examples(models, X, Y, device, paths["examples"], pgd_steps=pgd_steps, tag=tag)  # draw graph 4
    # WHAT: Tells the user where the files are.
    # HOW: Print each path.
    # WHY: Convenient.
    # PROS & CONS: + Visible. - None.
    # EXAMPLE: [plots] saved outputs/acc_vs_epsilon.png
    for p in paths.values():  # go through the saved files
        print(f"[plots] saved {p}")  # tell the user where each one was saved
    # WHAT: Returns the paths.
    # HOW: Dictionary.
    # WHY: Callers can open them.
    # PROS & CONS: + Convenient. - None.
    # EXAMPLE: paths['transfer']
    return paths  # give back the file paths
