# LSTM-Diffusion-GNN-for-SC-OPF

# Group Information
- **Group Number**: 32
- **Track Number**: 1
- **Course**: ECE228
- **Project Type**: Final Project

## Authors
- Daniela Rojas
- Jonathan Cremonesi
- Zhenhua Zhang

# Overview
This repository contains implementations of advanced machine learning models for Security-Constrained Optimal Power Flow (SC-OPF) problems. The project integrates solar PV forecasting techniques with graph-based machine learning models to enhance power grid operation and resilience under varying renewable energy conditions and contingencies.

# Repository Structure
```
.
├── LSTM/                      # LSTM-based solar PV forecasting
│   ├── Data/                  # Training and test datasets
│   ├── LSTM-Point.ipynb       # Point forecasting implementation
│   ├── LSTM-Quantile.ipynb    # Quantile forecasting implementation
│   └── lstm_quantile.py       # Core LSTM implementation
│
├── DDPM/                      # Diffusion-based forecasting
│   ├── data/                  # Training and test datasets
│   ├── results_timegrad/      # Model results and outputs
│   └── timegrad_v3.py         # TimeGrad DDPM implementation
│
└── GNN_DC_OPF/               # Graph Neural Networks for DC-OPF
    └── 3_gnn_dc_opf.ipynb    # GNN implementation and evaluation
```

# Setup and Prerequisites
Before running the code, ensure you have the necessary libraries installed:

```bash
pip install jupyter
pip install numpy scikit-learn matplotlib tqdm
pip install torch 
pip install torch_geometric  # For Graph Neural Networks
pip install cvxpy           # For DC-OPF optimization
pip install networkx        # For graph visualization
```

# Components

## 1. LSTM-based Solar PV Forecasting
Located in the `LSTM/` directory, this component implements two types of LSTM models:
- Point LSTM: Provides single-point forecasts with low RMSE, ideal for routine scheduling
- Quantile LSTM: Offers uncertainty quantification through well-calibrated prediction intervals

Key files:
- `LSTM-Point.ipynb`: Implementation of point forecasting
- `LSTM-Quantile.ipynb`: Implementation of quantile forecasting
- `lstm_quantile.py`: Core LSTM model implementation

## 2. Diffusion-based Forecasting
Located in the `DDPM/` directory, this component implements TimeGrad DDPM (Diffusion Denoising Probabilistic Model) for probabilistic solar PV forecasting.

Key files:
- `timegrad_v3.py`: Implementation of the TimeGrad DDPM model
- Results are stored in `results_timegrad/` directory

## 3. GNN for DC-OPF Solutions
Located in the `GNN_DC_OPF/` directory, this component implements Graph Neural Networks for solving DC-OPF problems.

Key features:
- Implements multiple GNN architectures (SimpleGCN, GCN, GAT)
- Uses physics-informed loss functions
- Includes DC-OPF solver implementation
- Generates synthetic datasets for training

# Usage

## LSTM Models
1. Navigate to the `LSTM/` directory
2. Open either `LSTM-Point.ipynb` or `LSTM-Quantile.ipynb`
3. Run the notebooks sequentially

## Diffusion Model
1. Navigate to the `DDPM/` directory
2. Run the TimeGrad implementation in `timegrad_v3.py`

## GNN for DC-OPF
1. Navigate to the `GNN_DC_OPF/` directory
2. Open `3_gnn_dc_opf.ipynb`
3. Run the notebook sequentially

# Key Findings

## LSTM Models
- Point LSTM achieves lowest RMSE for routine scheduling
- Quantile LSTM provides well-calibrated uncertainty intervals for reserve-margin hedging

## Diffusion Model
- TimeGrad DDPM shows reliable training with tight convergence
- Demonstrates good generalization to unseen solar PV data

## GNN for DC-OPF
- Successfully learns mapping from power system inputs to DC-OPF solutions
- Physics-informed loss functions improve solution feasibility
- Initial results show promise for basic scenarios, with ongoing work on generalization

# Future Work
- Enhance GNN generalization for complex N-1 contingencies
- Improve model robustness for unseen topologies
- Explore integration of different forecasting methods
- Develop more sophisticated physics-informed loss functions

# Contributing
Feel free to submit issues and enhancement requests!
