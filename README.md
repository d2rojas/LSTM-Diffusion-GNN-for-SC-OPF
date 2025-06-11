# LSTM-Diffusion-GNN-for-SC-OPF

# Overview
This repository contains a series of Jupyter notebooks that collectively explore the development of computationally efficient and robust decision-making strategies for the Security-Constrained Optimal Power Flow (SC-OPF) problem. Our work integrates advanced solar PV forecasting techniques with graph-based machine learning models to enhance power grid operation and resilience, particularly under varying renewable energy conditions and contingencies.

# Table of Contents
1. Setup and Prerequisites
2. Notebook 1: Solar PV Forecasting using LSTM [Notebook 1](1_solar_pv_forecasting_lstm.ipynb)
3. Notebook 2: Solar PV Forecasting using Diffusion Models [Notebook 2](2_solar_pv_forecasting_diffusion.ipynb)
4. Notebook 3: GNN for DC OPF Solutions [Notebook 3](3_gnn_dc_opf.ipynb)
5. Overall Project Insights & Future Work

-----

# Setup and Prerequisites
Before running the notebooks, ensure you have the necessary libraries installed.

## Install required libraries
```
pip install jupyter
pip install numpy scikit-learn matplotlib tqdm
pip install torch 
pip install torch_geometric # For Graph Neural Networks
pip install cvxpy # For DC-OPF optimization
pip install networkx # For graph visualization
```

# Notebook 1: Solar PV Forecasting using LSTM
This notebook focuses on the initial task of generating accurate solar PV forecasts, which serve as crucial input for downstream power system optimization problems. It implements and evaluates different deep learning models for forecasting.

Purpose:
- To implement and train various time-series forecasting models for solar PV output.
- To evaluate the performance of these models based on different use cases (point prediction vs. uncertainty quantification).

> [!NOTE]
> How to Run:
> - Open `1_solar_pv_forecasting_lstm.ipynb`.
> - Run all cells sequentially.

Key Takeaways:
- Point LSTM: You'll find that this model yields the lowest RMSE for routine scheduling. It's ideal when a single, most probable forecast value is sufficient for operational planning.
- Quantile LSTM: This model provides well-calibrated uncertainty intervals, albeit with a slightly higher median RMSE compared to the Point LSTM. This makes it invaluable for reserve-margin hedging, allowing operators to account for forecast variability and manage reserve requirements more effectively.

# Notebook 2: Solar PV Forecasting using Diffusion Models

> [!NOTE]
> How to Run:
> - Open `2_solar_pv_forecasting_diffusion.ipynb`.
> - Run all cells sequentially.

Key Takeaways:
- TimeGrad DDPM (Diffusion Denoising Probabilistic Model): Despite an observed optimistic bias in its predictions, the tight convergence of training and validation losses (approximately 0.16) confirms that the - TimeGrad DDPM model trains reliably and generalizes well to unseen solar PV data. This indicates its potential for robust probabilistic forecasting.

# Notebook 3: GNN for DC OPF Solutions
This notebook is designed to create a synthetic dataset of DC-OPF solutions. This dataset will serve as the ground truth for training the GNN models in the next stage. It leverages the `solve_dc_opf` function to obtain optimal power flow solutions under a wide range of simulated load demand scenarios. 

Then, this notebook implements and trains various Graph Neural Network (GNN) models to learn the mapping from power system inputs (e.g., load demands, grid topology) to DC-OPF solutions. Critically, it incorporates physics-informed loss functions to guide the GNN towards physically feasible predictions.

Example Inference: The notebook provides a detailed example inference on a test scenario, allowing you to directly inspect the predicted `Pg` and `theta` values against their true counterparts, along with explicit checks for constraint violations.

Purpose:
- To implement a robust DC-OPF solver (using `cvxpy`).
- To generate a diverse dataset of feasible DC-OPF solutions by varying load demands on individual buses using a `test_dc_opf_100_load_scenarios_per_bus_scale` function.
- To save these input parameters (load demands) and corresponding optimal decision variables (generator outputs, bus angles, line flows) into a structured JSON file for GNN training.
- To implement and train Simple Graph Convolutional Network (SimpleGCN), standard Graph Convolutional Network (GCN), and Graph Attention Network (GAT) models.
- To evaluate the GNNs' performance in predicting OPF solutions and their ability to satisfy physical constraints.

> [!NOTE]
> How to Run:
> - Open `3_gnn_dc_opf.ipynb`.
> - Run all cells sequentially.
> - Examine the `solve_dc_opf` function and the defined IEEE 6-bus system data.
> - Run the `test_dc_opf_100_load_scenarios_per_bus_scale` function. This function will simulate 20k different load scenarios (each bus receiving a unique random load scaling factor) and save the feasible solutions.
> - The output will be saved to `feasible_opf_results_per_bus_scale.json`.
> - Review the `SimpleGNN` model definition and especially the train_one_epoch and evaluate_model functions to understand how the physics-informed loss terms are calculated and applied. Pay attention to the `lambda_gen`, `lambda_flow`, and `lambda_balance` hyperparameters – their tuning is crucial.

Key Takeaways:
- Model Performance: You will observe the training and testing MSE, along with the values of the generator, flow, and balance penalties. The goal is to minimize MSE while keeping penalties low, indicating feasible solutions.
- GNN Generalization (Initial Findings): Based on the current setup (likely on the IEEE 6-bus system and N-0 scenarios):
- Simple GCN models may initially outperform attention-based models in basic accuracy, but this can vary with hyperparameter tuning and dataset size.
- A key observation is that these models, even with physics-informed losses, may still struggle to generalize robustly to unseen topologies or more complex N-1 contingencies.


# Overall Project Insights & Future Work
This collection of notebooks lays the groundwork for leveraging AI in power system optimization.

We have established effective solar PV forecasting methods suitable for different operational needs.
We've demonstrated how to generate structured datasets from conventional OPF solvers for GNN training.
Our initial foray into GNNs for DC-OPF shows promise in learning the complex input-output mappings, and the value of physics-informed loss functions in improving solution feasibility.

This project is an ongoing effort to build data-driven tools that can support faster, more reliable, and more robust power system operations in the era of renewable energy integration.
