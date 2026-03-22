# Accurate Split Learning on Noisy Signals
This is the official implementation of the paper [Accurate Split Learning on Noisy Signals](https://openreview.net/forum?id=in1T4BlzG9) by [H. Xu](https://scholar.google.com/citations?user=UhUecFUAAAAJ), [S. Maity](https://subhajitmaity.me), [A. Dutta](https://sciences.ucf.edu/math/person/aritra-dutta), [X. Li](https://sciences.ucf.edu/math/person/xin-li), [P. Kalnis](https://kalnis.org/).

---

## Training Scripts

### 1. CIFAR10 Training (`train_cifar10_dp.py`)
Train a ResNet20 model on CIFAR10 dataset with split learning and optional differential privacy.

#### Basic Command
```bash
python train_cifar10_dp.py
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--arch`, `-a` | str | `resnet20` | Model architecture |
| `-j`, `--workers` | int | `2` | Number of data loading workers |
| `--epochs` | int | `200` | Number of total epochs to run |
| `--start-epoch` | int | `0` | Manual epoch number (useful on restarts) |
| `-b`, `--batch-size` | int | `128` | Mini-batch size |
| `--lr`, `--learning-rate` | float | `0.1` | Initial learning rate |
| `--momentum` | float | `0.9` | Momentum |
| `--weight-decay`, `--wd` | float | `1e-4` | Weight decay |
| `--print-freq`, `-p` | int | `50` | Print frequency (batches) |
| `--resume` | str | `''` | Path to latest checkpoint |
| `-e`, `--evaluate` | flag | - | Evaluate model on validation set |
| `--pretrained` | flag | - | Use pre-trained model |
| `--half` | flag | - | Use half-precision (16-bit) |
| `--save-dir` | str | `save_temp` | Directory to save trained models |
| `--save-every` | int | `10` | Save checkpoints every N epochs |
| `--split-layer` | int | `-1` | Layer index to split model |
| `--enable-dp` | flag | - | Add DP Gaussian noise on transmitted tensors |
| `--sigma` | float | `0.7` | Std of Gaussian noise |
| `--enable-denoise` | flag | - | Enable denoising methods (scaling & dropout) |
| `--scaling-factor` | float | `1.0` | Scale value of noise injected tensors |
| `--mask-ratio` | float | `1.0` | Ratio of elements kept after masking |
| `--avg-count` | int | `1` | Averaging counts for dropout |

#### Example Usage

```bash
# Basic training with default settings
python train_cifar10_dp.py

# Split Training with split at the last layer
python train_cifar10_dp.py --split-layer -1

# Split Training with Noise
python train_cifar10_dp.py --enable-dp --sigma 0.7

# Split Training with Denoising (Masking)
python train_cifar10_dp.py --enable-dp --sigma 0.7 --enable-denoise --mask-ratio 0.1

# Split Training with Denoising (Scaling)
python train_cifar10_dp.py --enable-dp --sigma 0.7 --enable-denoise --scaling-factor 0.1

```

---

### 2. ImageNet Training (`train_imagenet_dp.py`)
Train a ResNet50 model on ImageNet dataset with split learning capabilities.

#### Basic Command
```bash
python train_imagenet_dp.py
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--arch`, `-a` | str | `resnet50` | Model architecture |
| `-j`, `--workers` | int | `4` | Number of data loading workers |
| `--epochs` | int | `90` | Number of total epochs to run |
| `--start-epoch` | int | `0` | Manual epoch number |
| `-b`, `--batch-size` | int | `256` | Mini-batch size |
| `--lr`, `--learning-rate` | float | `0.1` | Initial learning rate |
| `--momentum` | float | `0.9` | Momentum |
| `--weight-decay`, `--wd` | float | `1e-4` | Weight decay |
| `--print-freq`, `-p` | int | `50` | Print frequency (batches) |
| `--resume` | str | `''` | Path to checkpoint for resuming |
| `-e`, `--evaluate` | flag | - | Evaluate model on validation set |
| `--pretrained` | flag | - | Use pre-trained model |
| `--half` | flag | - | Use half-precision (16-bit) |
| `--save-dir` | str | `save_temp` | Directory to save models |
| `--save-every` | int | `1` | Save checkpoints every N epochs |
| `--split-layer` | int | `-1` | Layer index to split |
| `--enable-dp` | flag | - | Add DP noise on transmitted tensors |
| `--sigma` | float | `0.7` | Std of Gaussian noise |
| `--enable-denoise` | flag | - | Enable denoising methods |
| `--scaling-factor` | float | `1.0` | Scale of noise injected tensors |
| `--mask-ratio` | float | `1.0` | Ratio of elements kept after masking |
| `--avg-count` | int | `1` | Averaging counts for dropout |
| `--run-name` | str | `None` | Run name for wandb logging |
| `--run-id` | str | `None` | Run ID for wandb (resume run) |

#### Example Usage

```bash
# Basic training with default settings
python train_imagenet_dp.py

# Split Training with split at the last layer
python train_imagenet_dp.py --split-layer -1

# Split Training with Noise
python train_imagenet_dp.py --enable-dp --sigma 0.7

# Split Training with Denoising (Masking)
python train_imagenet_dp.py --enable-dp --sigma 0.7 --enable-denoise --mask-ratio 0.1

# Split Training with Denoising (Scaling)
python train_imagenet_dp.py --enable-dp --sigma 0.7 --enable-denoise --scaling-factor 0.1

```

---

### 3. MNIST Training (`train_mnist_dp.py`)
Train a CNN on MNIST dataset using split learning architecture.

#### Basic Command
```bash
python train_mnist_dp.py
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | int | `64` | Batch size for training |
| `--test-batch-size` | int | `1000` | Batch size for testing |
| `--epochs` | int | `5` | Number of epochs to train |
| `--lr` | float | `0.1` | Learning rate |
| `--gamma` | float | `0.7` | Learning rate step gamma |
| `--no-cuda` | flag | - | Disable CUDA training |
| `--dry-run` | flag | - | Quickly check a single pass |
| `--seed` | int | `1` | Random seed |
| `--log-interval` | int | `10` | Log training status every N batches |
| `--test-interval` | int | `100` | Run test every N batches |
| `--save-model` | flag | - | Save the trained model |
| `--split-layer` | int | `-1` | Layer index to split |
| `--add-noise` | flag | - | Add Gaussian noise on transmitted tensors |
| `--sigma` | float | `0.7` | Std of Gaussian noise |
| `--enable-denoise` | flag | - | Enable denoising methods |
| `--dropout-only` | flag | - | Use dropout instead of masking |
| `--scaling-factor` | float | `1.0` | Scale of noise injected tensors |
| `--mask-ratio` | float | `1.0` | Ratio of elements kept after masking |
| `--weight-decay` | float | `0.0` | Weight decay factor |

#### Example Usage

```bash
# Basic training with default settings
python train_mnist_dp.py

# Split Training with split at the last layer
python train_mnist_dp.py --split-layer -1

# Split Training with Noise
python train_mnist_dp.py --add-noise --sigma 0.7

# Split Training with Denoising (Masking)
python train_mnist_dp.py --add-noise --sigma 0.7 --enable-denoise --mask-ratio 0.1

# Split Training with Denoising (Scaling)
python train_mnist_dp.py --add-noise --sigma 0.7 --enable-denoise --scaling-factor 0.1

```

---

### 4. IMDB Training (`train_imdb_dp.py`)
Train a sentiment prediction model on IMDB movie reviews dataset with split learning.

#### Basic Command
```bash
python train_imdb_dp.py
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-b`, `--batch-size` | int | `64` | Batch size for training |
| `-n`, `--epochs` | int | `10` | Number of epochs to train |
| `--lr` | float | `0.01` | Learning rate |
| `-c`, `--max-per-sample-grad_norm` | float | `1.0` | Clip per-sample gradients to this norm |
| `--delta` | float | `1e-5` | Target delta for privacy |
| `--max-sequence-length` | int | `256` | Max sequence length (longer sequences cut) |
| `--device` | str | `cuda` | GPU ID (cuda or cpu) |
| `--save-model` | flag | - | Save the trained model |
| `--disable-dp` | flag | - | Disable privacy (train with vanilla optimizer) |
| `--secure-rng` | flag | - | Enable secure RNG for trustworthy privacy |
| `--data-root` | str | `../imdb` | Path where IMDB data is/will be stored |
| `-j`, `--workers` | int | `2` | Number of data loading workers |
| `--split-layer` | int | `-1` | Layer index to split |
| `--enable-dp` | flag | - | Add DP noise on transmitted tensors |
| `--sigma` | float | `0.7` | Std of Gaussian noise |
| `--enable-denoise` | flag | - | Enable denoising methods |
| `--scaling-factor` | float | `1.0` | Scale of noise injected tensors |
| `--mask-ratio` | float | `1.0` | Ratio of elements kept after masking |
| `--avg-count` | int | `1` | Averaging counts for dropout |
| `--weight-decay` | float | `0.0` | Weight decay factor |

#### Example Usage

```bash
# Basic training with default settings
python train_imdb_dp.py

# Split Training with split at the last layer
python train_imdb_dp.py --split-layer -1

# Split Training with Noise
python train_imdb_dp.py --enable-dp --sigma 0.7

# Split Training with Denoising (Masking)
python train_imdb_dp.py --enable-dp --sigma 0.7 --enable-denoise --mask-ratio 0.1

# Split Training with Denoising (Scaling)
python train_imdb_dp.py --enable-dp --sigma 0.7 --enable-denoise --scaling-factor 0.1

```

---

### 5. ALBERT Training (`train_albert_dp.py`)
Train an ALBERT classifier on Amazon Reviews dataset with split learning support.

#### Basic Command
```bash
python train_albert_dp.py
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--arch`, `-a` | str | `albert-base-v2` | ALBERT model architecture variant |
| `-j`, `--workers` | int | `4` | Number of data loading workers |
| `--epochs` | int | `10` | Number of epochs to train |
| `--start-epoch` | int | `0` | Manual epoch number |
| `-b`, `--batch-size` | int | `256` | Mini-batch size |
| `--lr`, `--learning-rate` | float | `5e-5` | Initial learning rate |
| `--momentum` | float | `0.9` | Momentum |
| `--weight-decay`, `--wd` | float | `1e-2` | Weight decay |
| `--print-freq`, `-p` | int | `50` | Print frequency (batches) |
| `--resume` | str | `''` | Path to checkpoint |
| `-e`, `--evaluate` | flag | - | Evaluate on validation set |
| `--pretrained` | flag | - | Use pre-trained model |
| `--half` | flag | - | Use half-precision (16-bit) |
| `--save-dir` | str | `save_temp` | Directory to save models |
| `--save-every` | int | `1` | Save checkpoints every N epochs |
| `--split-layer` | int | `-1` | Layer index to split |
| `--enable-dp` | flag | - | Add DP noise on transmitted tensors |
| `--sigma` | float | `0.7` | Std of Gaussian noise |
| `--enable-denoise` | flag | - | Enable denoising methods |
| `--scaling-factor` | float | `1.0` | Scale of noise injected tensors |
| `--mask-ratio` | float | `1.0` | Ratio of elements kept after masking |
| `--avg-count` | int | `1` | Averaging counts for dropout |
| `--run-name` | str | `None` | Run name for wandb |
| `--run-id` | str | `None` | Run ID for wandb (resume run) |

#### Example Usage

```bash
# Basic training with default settings
python train_albert_dp.py

# Split Training with split at the last layer
python train_albert_dp.py --split-layer -1

# Split Training with Noise
python train_albert_dp.py --enable-dp --sigma 0.7

# Split Training with Denoising (Masking)
python train_albert_dp.py --enable-dp --sigma 0.7 --enable-denoise --mask-ratio 0.1

# Split Training with Denoising (Scaling)
python train_albert_dp.py --enable-dp --sigma 0.7 --enable-denoise --scaling-factor 0.1

```

---

## Common Features Across All Training Scripts

### Split Training Options
All training scripts support adding Gaussian noise to sensitive intermediate representations:
- `--enable-dp`: Activates noise injection
- `--sigma`: Controls noise level (higher = more noise = higher privacy)
- `--split-layer`: Layer index where model is split between client and server. Default is -1 that ensures split at the last layer

### Denoising Options
When using DP, enable denoising to improve model performance:
- `--enable-denoise`: Activates denoising methods
- `--mask-ratio`: Proportion of activations to keep, 1 symbolizes no masking
- `--scaling-factor`: Multiplier (0-1) for noise injected IRs, 1 means no scaling
- `--avg-count`: Number of averaging iterations

### Checkpointing
- `--save-dir`: Where to save model checkpoints
- `--save-every`: Frequency of checkpoint saves
- `--resume`: Resume training from checkpoint

---

## Dataset Paths
Ensure datasets are available at expected paths:
- **CIFAR10**: `../cifar10` (relative path)
- **ImageNet**: `/datasets/ImageNet2012nonpub/` (configurable)
- **MNIST**: `./data` (relative path)
- **IMDB**: `../imdb` (configurable via `--data-root`)
- **Amazon Reviews**: `./data/amazon_review_full_csv` (required for ALBERT)

---

## Requirements
- Python 3.14
- torch 2.4.0, torchvision 0.19.0
- transformers
- datasets
- opacus
- wandb
- pandas
