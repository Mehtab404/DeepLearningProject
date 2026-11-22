# WHAT: Marks the src/ folder as a Python package.
# HOW: The presence of __init__.py lets 'from src.attacks import ...' work.
# WHY: All modules import each other with relative imports (from .utils import ...).
# PROS & CONS: + Standard packaging. - The file has no code.
# EXAMPLE: python -m src.models runs the models module as part of the package.
# Adversarial robustness: CNN vs ViT on CIFAR-10  <- this folder is a Python package (a group of code files)
