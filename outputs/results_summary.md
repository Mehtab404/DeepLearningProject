# Results summary (accuracy %, white-box)

> Quick-test smoke run only (N=60, 2 PGD steps): not valid results.

| attack, model | eps=0/255 | eps=1/255 | eps=2/255 | eps=4/255 | eps=8/255 | eps=16/255 |
| --- | --- | --- | --- | --- | --- | --- |
| FGSM, CNN-Clean | 88.3 | 31.7 | 21.7 | 5.0 | 6.7 | 11.7 |
| FGSM, ViT-Clean | 30.0 | 6.7 | 5.0 | 1.7 | 5.0 | 8.3 |
| FGSM, CNN-AdvTrained | 93.3 | 26.7 | 21.7 | 10.0 | 8.3 | 10.0 |
| FGSM, ViT-AdvTrained | 31.7 | 11.7 | 6.7 | 3.3 | 5.0 | 10.0 |
| PGD-10, CNN-Clean | 88.3 | 25.0 | 11.7 | 10.0 | 0.0 | 0.0 |
| PGD-10, ViT-Clean | 30.0 | 6.7 | 1.7 | 0.0 | 0.0 | 0.0 |
| PGD-10, CNN-AdvTrained | 93.3 | 25.0 | 13.3 | 8.3 | 0.0 | 0.0 |
| PGD-10, ViT-AdvTrained | 31.7 | 8.3 | 5.0 | 1.7 | 0.0 | 0.0 |

## Transferability: PGD-10 at eps=8/255 (rows = source, cols = target; accuracy %)

| source \ target | CNN-Clean | ViT-Clean | CNN-AdvTrained | ViT-AdvTrained |
| --- | --- | --- | --- | --- |
| CNN-Clean | 0.0 | 21.7 | 0.0 | 25.0 |
| ViT-Clean | 80.0 | 0.0 | 81.7 | 0.0 |
| CNN-AdvTrained | 0.0 | 21.7 | 0.0 | 21.7 |
| ViT-AdvTrained | 78.3 | 0.0 | 85.0 | 0.0 |
