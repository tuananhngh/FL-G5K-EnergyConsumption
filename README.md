# FL-G5K-EnergyConsumption

## Overview

This repository contains the source code and documentation for the FL-G5K-EnergyConsumption project, specifically the Toulouse G5K branch. FL-G5K-EnergyConsumption is a Federated Learning (FL) framework designed for energy consumption analysis in the Grid'5000 testbed.

## Table of Contents

- [Introduction](#introduction)
- [Repository organisation](#repository-organisation)
- [Getting Started](#getting-started)

## Introduction

The FL-G5K-EnergyConsumption project aims to provide a comprehensive solution for analyzing energy consumption patterns in Federated Learning frameworks using the Grid'5000 testbed. The Toulouse G5K branch was adapted to be executed on the estats cluster of the Toulouse site within the Grid'5000 infrastructure.
We rely on the [Flower framework](https://flower.dev/) for Federated Learning.

## Repository organisation
The code can be found in the [source repository](./src/).

1. **Configuration:**
   - The `config/` directory provides configuration files (`config.yaml`) for training and evaluation.

2. **Energy monitoring:**
   - The `energy/` directory contains 2 modules for monitoring the energy consumption of the Jetson AGX Xavier ([jetson_monitoring_energy.py](./src/energy/jetson_monitoring_energy.py)).

3. **Utilities:**
   - The `utils/` directory contains utility functions and modules.
        - [experiment.py](./src/utils/experiment.py) provides functions and a class to run experiments on a server and serveral nodes.
        - [models.py](./src/utils/models.py) provides models.
        - [process_results.py](./src/utils/process_results.py) provides functions to read, process and plot logs from experiments.
        - [training.py](./src/utils/training.py) provides functions required to use Flower.

4. **Main files**
    - [reservation.ipynb](./src/reservation.ipynb) is a notebook you can use to make job reservation and start experiments from the frontend.
    - [server.py](./src/server.py) is the code to be executed on the server node.
    - [client.py](./src/client.py) is the code to be executed on the client nodes.

5. **Analysis**
    - Notebooks to analyse results can be found in the [analysis](./analysis/) folder.

## Getting Started

To get started with FL-G5K-EnergyConsumption, you need to have an Grid'5000 account. 
If you do, you can connect to the Toulouse frontend and follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/tuananhngh/FL-G5K-EnergyConsumption.git -b toulouse-g5k
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook [reservation.ipynb](./src/reservation.ipynb) and follow the instructions there.

<!-- ## A more comprehensive documentation
### Federated Learning with Flower
### Job reservation and ssh processes with Execo
### Storage
We relied on a group storage to run experiments.
https://www.grid5000.fr/w/Storage_Manager#Usage -->