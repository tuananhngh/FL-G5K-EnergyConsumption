# FL-G5K-EnergyConsumption

A Federated Learning (FL) framework for measuring and analysing energy consumption on the [Grid'5000](https://www.grid5000.fr/) testbed, targeting the **estats** cluster (Nvidia Jetson AGX Xavier nodes) at the Toulouse site.

Built on top of [Flower](https://flower.ai/) for FL coordination and [Execo](http://execo.gforge.inria.fr/) for job management and SSH orchestration.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Analysis](#analysis)

---

## Overview

The project automates the full lifecycle of a federated learning experiment on Grid'5000:

1. Reserve nodes via OAR on the estats cluster.
2. Partition a dataset (IID, label-skew, or sample-skew) and distribute it across client nodes.
3. Launch an FL training loop: a central server coordinates multiple Jetson AGX Xavier clients over several rounds using a configurable aggregation strategy.
4. Monitor energy consumption on each node using **jtop** and network traffic using **nethogs**, timestamped alongside training events.
5. Copy results to group storage and summarise hyperparameters in a CSV.
6. Post-process logs to compute per-round and per-experiment energy metrics.

### Supported configurations

| Dimension | Options |
|-----------|---------|
| Models | `Net` (basic CNN), `ResNet18`, `MobileNetV3Small`, `MobileNetV3Large` |
| Strategies | `FedAvg`, `FedAdam`, `FedYogi`, `FedAdaGrad`, `FedMedian` |
| Optimizers | `SGD`, `Adam` |
| Datasets | CIFAR-10, CIFAR-100, MNIST, FashionMNIST, TinyImageNet |
| Data partitioning | IID, label skew (Dirichlet), sample skew (Dirichlet) |

---

## Project Structure

```
.
├── src/
│   ├── server.py                   # FL server entry point (CustomServer + Hydra main)
│   ├── client.py                   # FL client entry point (Flower NumPyClient + Hydra main)
│   ├── reservation.ipynb           # Notebook: OAR job reservation and experiment launch
│   ├── config/
│   │   ├── config_file.yaml        # Top-level Hydra config
│   │   ├── data_config.yaml        # Data partitioning config
│   │   ├── neuralnet/              # Per-model configs
│   │   ├── optimizer/              # Per-optimizer configs
│   │   └── strategy/               # Per-strategy configs
│   ├── energy/
│   │   ├── jetson_monitoring_energy.py   # jtop-based energy logger (runs on each node)
│   │   └── xav_read_power.py             # Alternative: reads /sys hwmon and pushes to kwollect
│   ├── movielens/                  # Experimental MovieLens / FedRecon setup
│   └── utils/
│       ├── experiment.py           # Experiment orchestration via Execo SSH
│       ├── datahandler.py          # Dataset loading, partitioning, and persistence
│       ├── datapartition.py        # CLI script: partition dataset and save to disk
│       ├── models.py               # Model definitions (Net, ResNet18)
│       ├── training.py             # train / test / validation loops + seeding
│       ├── process_results.py      # Result reading and plotting utilities
│       ├── process_energy.py       # Energy aggregation pipeline
│       ├── process_results_for_energy.py  # EnergyResult class for per-experiment analysis
│       ├── readnetwork.py          # Network log parser
│       └── rep_exp.sh              # Shell helper: repeat experiment.py N times
├── analysis/
│   ├── global_analysis.ipynb       # Cross-experiment comparison plots
│   ├── analysis_comm.ipynb         # Communication timing analysis
│   ├── filter_exp.ipynb            # Filter and select experiments
│   ├── processes.ipynb             # Per-process resource usage
│   ├── read_server_results.ipynb   # Server-side result inspection
│   ├── comm_utils.py               # Shared utilities for analysis notebooks
│   └── experiments.json            # Experiment index for analysis
├── images/                         # Kameleon image descriptors for Jetson nodes
├── kameleon_recipes/               # Kameleon recipes for building the custom ARM64 environment
├── legacy/                         # Archived Docker-based deployment scripts (superseded)
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

- A [Grid'5000 account](https://www.grid5000.fr/w/Grid5000:Get_an_account) with access to the **Toulouse** site.
- Access to the **estats** cluster (Nvidia Jetson AGX Xavier nodes).
- A group storage allocation (default: `energyfl` at `storage1.toulouse.grid5000.fr`).
- Python 3.8+ and the dependencies below installed in your environment on the nodes.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tuananhngh/FL-G5K-EnergyConsumption.git
cd FL-G5K-EnergyConsumption
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The core runtime dependencies are:

| Package | Version |
|---------|---------|
| `flwr` | 1.6.0 |
| `hydra-core` | 1.3.2 |
| `omegaconf` | 2.3.0 |
| `tqdm` | 4.66.1 |

Additional packages needed on the nodes (not in `requirements.txt`): `torch`, `torchvision`, `jtop`, `execo`, `python-box`, `pandas`, `pyyaml`.

### 3. Partition the dataset

Before running experiments, partition the dataset and save it to your group storage:

```bash
cd src/utils
python3 datapartition.py
```

This reads `src/config/data_config.yaml` and writes per-client `.pt` files to the configured `partition_dir`.

---

## Configuration

All configuration is managed by [Hydra](https://hydra.cc/). The main entry point is `src/config/config_file.yaml`, which composes sub-configs from:

- `neuralnet/` — model architecture and number of classes
- `optimizer/` — client-side optimizer
- `strategy/` — FL aggregation strategy
- `data` section — dataset name, partition type, alpha (Dirichlet), batch size, number of clients
- `params` section — number of rounds, fraction fit/evaluate, early stopping patience, model saving
- `client` section — local epochs, learning rate schedule
- `comm` section — server host and port (resolved at runtime)

Override any parameter from the command line:

```bash
python3 src/server.py neuralnet=ResNet18 strategy=fedavg data.alpha=0.5 params.num_rounds=100
```

---

## Running Experiments

The recommended workflow is through `src/reservation.ipynb`, which uses the `Experiment` class (`src/utils/experiment.py`) to:

1. Acquire an OAR job on the estats cluster.
2. Resolve the server's IP address.
3. Start energy and network monitors on all nodes.
4. Launch `server.py` on the first node and `client.py` on the remaining nodes via SSH.
5. Wait for training to complete, then copy results to group storage.
6. Record hyperparameters in `experiment_summary.csv`.

Open the notebook from the Toulouse Grid'5000 frontend:

```bash
jupyter notebook src/reservation.ipynb
```

To run programmatically, instantiate `Experiment` directly (see the `if __name__ == "__main__"` block in `experiment.py` for an example).

---

## Analysis

Post-experiment analysis notebooks are in `analysis/`:

| Notebook | Purpose |
|----------|---------|
| `global_analysis.ipynb` | Cross-experiment energy and accuracy comparison |
| `analysis_comm.ipynb` | Per-round communication timing |
| `filter_exp.ipynb` | Filter experiments by hyperparameters |
| `processes.ipynb` | Per-process CPU/GPU/memory usage |
| `read_server_results.ipynb` | Server training curves |

Before running the notebooks, preprocess the raw logs:

```python
from src.utils.process_energy import preprocess
preprocess(["path/to/output/dir"])
```

This computes per-host energy consumption and writes `energy_hosts_summary.csv` and `perf_summary.csv` into each experiment folder.
