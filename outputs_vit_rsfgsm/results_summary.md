# Results summary (accuracy %, white-box)

> N=1000 stratified CIFAR-10 test images, seed 42, PGD-10.

| attack, model | eps=0/255 | eps=1/255 | eps=2/255 | eps=4/255 | eps=8/255 | eps=16/255 |
| --- | --- | --- | --- | --- | --- | --- |
| FGSM, CNN-Clean | 91.5 | 43.5 | 25.1 | 14.8 | 11.9 | 13.4 |
| FGSM, ViT-Clean | 97.4 | 66.7 | 44.6 | 29.7 | 23.3 | 20.9 |
| FGSM, CNN-AdvTrained | 88.4 | 72.0 | 66.9 | 63.7 | 59.4 | 50.4 |
| FGSM, ViT-AdvTrained | 97.4 | 81.2 | 68.4 | 67.7 | 79.8 | 52.6 |
| PGD-10, CNN-Clean | 91.5 | 21.8 | 0.6 | 0.0 | 0.0 | 0.0 |
| PGD-10, ViT-Clean | 97.4 | 49.8 | 7.7 | 0.6 | 0.0 | 0.0 |
| PGD-10, CNN-AdvTrained | 88.4 | 64.1 | 54.6 | 39.7 | 18.3 | 1.0 |
| PGD-10, ViT-AdvTrained | 97.4 | 75.7 | 40.4 | 11.9 | 4.9 | 0.0 |

## Transferability: PGD-10 at eps=8/255 (rows = source, cols = target; accuracy %)

| source \ target | CNN-Clean | ViT-Clean | CNN-AdvTrained | ViT-AdvTrained |
| --- | --- | --- | --- | --- |
| CNN-Clean | 0.0 | 85.6 | 69.2 | 92.8 |
| ViT-Clean | 64.6 | 0.0 | 78.7 | 83.3 |
| CNN-AdvTrained | 9.1 | 77.3 | 18.3 | 86.3 |
| ViT-AdvTrained | 57.9 | 36.5 | 69.8 | 4.9 |
