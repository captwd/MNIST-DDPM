# MNIST Comparison Results

## Protocol

- Device: cuda
- Seed: 42, Epochs: 10, Batch size: 64, Optimizer: Adam(lr=0.001) + StepLR(step=5, gamma=0.5), Loss: CrossEntropy
- Data: official MNIST (60k train / 10k test), Normalize((0.5,), (0.5,)) -> [-1,1], num_workers=0
- Traditional ML inputs: flattened 784-dim vectors scaled to [-1,1]
- DDPM checkpoint: default/ddpm_ckpt.pth, sampling timesteps: 1000, generated: 2000
- Synthetic labels: pseudo-labels from CNN (its real-data best acc 0.9931), confidence >= 0.9, kept 1883/2000 (filtered)

## Trained on: real

| Model | Train size | Params | Time (s) | Best Acc | Final Acc |
|---|---|---|---|---|---|
| ANN | 60000 | 11,935 | 85.0 | 0.9338 | 0.9338 |
| CNN | 60000 | 421,834 | 92.3 | 0.9931 | 0.9923 |
| ViT | 60000 | 1,199,882 | 238.6 | 0.9732 | 0.9696 |
| SVM | 60000 | - | 198.4 | 0.9792 | 0.9792 |
| KNN | 60000 | - | 9.3 | 0.9688 | 0.9688 |
| RandomForest | 60000 | - | 25.4 | 0.9705 | 0.9705 |
| LogisticRegression | 60000 | - | 10.5 | 0.9225 | 0.9225 |

## Trained on: ddpm_synthetic

| Model | Train size | Params | Time (s) | Best Acc | Final Acc |
|---|---|---|---|---|---|
| ANN | 1883 | 11,935 | 12.3 | 0.6658 | 0.6658 |
| CNN | 1883 | 421,834 | 13.8 | 0.9601 | 0.9601 |
| ViT | 1883 | 1,199,882 | 20.4 | 0.8052 | 0.7972 |
