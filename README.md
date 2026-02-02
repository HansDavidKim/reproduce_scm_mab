# Structural Causal Bandits (SCB) Simulation Engine

Reproduction study of **"Structural Causal Bandits: Where to Intervene?"** (NeurIPS 2018). This repository contains a simulation engine for evaluating "Where to Intervene" in a causal graph using Multi-Armed Bandit (MAB) algorithms combined with causal inference (POMIS).

## 🚀 Key Features

*   **Causal Inference Engine**: 
    - Implementation of **POMIS** (Possibly Optimal Minimal Intervention Sets) to prune redundant intervention arms.
    - Graph algorithms: `MUCT` (Maximal Unobserved Confounders' Territory) and `IB` (Intervention Boundary).
*   **Multi-Armed Bandit Algorithms**:
    - **UCB** (Upper Confidence Bound) with Hoeffding's Inequality.
    - **KL-UCB**: High-performance variant using Kullback-Leibler divergence.
    - **Thompson Sampling (TS)**: Bayesian approach with Beta-Bernoulli distributions.
*   **Structural Causal Model (SCM) Environment**:
    - Bernoulli-sampling based environment with directed and bidirected edges (latent confounders).
    - **Monte Carlo Oracle**: Accurate identification of the optimal arm through high-precision expectation estimation.
*   **Integrated Benchmark Suite**:
    - Batch experimentation over multiple random seeds.
    - Progress visualization via `tqdm`.
    - Automated result persistence in CSV format and comparative visualization.

## 🛠 Installation

Requirements: Python 3.8+
```bash
pip install -r requirements.txt
```

## 📖 Usage

The engine is controlled via a CLI interface in `main.py`.

### 1. Run a Single Experiment
Evaluate a specific algorithm on a chosen benchmark graph.
```bash
python main.py run-experiment --algorithm kl-ucb --benchmark chain_3.json --T 5000
```

### 2. Run Comparative Benchmarks
Compare all implemented algorithms (TS, UCB, KL-UCB) with and without POMIS filtering across multiple seeds.
```bash
python main.py run-benchmark --benchmark chain_2.json --num-seed 5
```

### 3. Visualize Results
Plot cumulative regret curves from the saved CSV results.
```bash
python main.py visualize-result --benchmark chain_2.json
```

## 📂 Project Structure

- `causal_engine/`: Graph pruning and POMIS logic.
- `environments/`: SCM simulation and Monte Carlo estimation.
- `mab_algorithms/`: Bandit algorithm implementations (UCB, KL-UCB, TS).
- `benchmark/`: JSON-based causal graph definitions.
- `results/`: CSV output directory for benchmark data.
- `main.py`: Main CLI entry point.

---
*Based on the research: "Structural Causal Bandits: Where to Intervene?" (NeurIPS 2018).*