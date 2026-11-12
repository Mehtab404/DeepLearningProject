# Results summary (accuracy %, white-box)

> N=1000 stratified CIFAR-10 test images, seed 42, PGD-10.

| attack, model | eps=0/255 | eps=1/255 | eps=2/255 | eps=4/255 | eps=8/255 | eps=16/255 |
| --- | --- | --- | --- | --- | --- | --- |
| FGSM, CNN-Clean | 91.5 | 43.5 | 25.1 | 14.8 | 11.9 | 13.4 |
| FGSM, ViT-Clean | 97.1 | 67.2 | 44.1 | 28.3 | 22.4 | 19.8 |
| FGSM, CNN-AdvTrained | 88.4 | 72.0 | 66.9 | 63.7 | 59.4 | 50.4 |
| FGSM, ViT-AdvTrained | 94.1 | 88.8 | 85.8 | 81.9 | 72.7 | 57.6 |
| PGD-10, CNN-Clean | 91.5 | 21.8 | 0.6 | 0.0 | 0.0 | 0.0 |
| PGD-10, ViT-Clean | 97.1 | 48.4 | 6.8 | 0.3 | 0.0 | 0.0 |
| PGD-10, CNN-AdvTrained | 88.4 | 64.1 | 54.6 | 39.7 | 18.3 | 1.0 |
| PGD-10, ViT-AdvTrained | 94.1 | 85.8 | 80.1 | 68.2 | 42.2 | 5.3 |

## Transferability: PGD-10 at eps=8/255 (rows = source, cols = target; accuracy %)

| source \ target | CNN-Clean | ViT-Clean | CNN-AdvTrained | ViT-AdvTrained |
| --- | --- | --- | --- | --- |
| CNN-Clean | 0.0 | 85.4 | 69.2 | 87.1 |
| ViT-Clean | 65.8 | 0.0 | 79.5 | 86.1 |
| CNN-AdvTrained | 9.1 | 78.0 | 18.3 | 82.1 |
| ViT-AdvTrained | 45.7 | 32.4 | 51.3 | 42.2 |
