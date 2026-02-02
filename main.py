# You can run experiments here
import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments.environ import Environ
from pathlib import Path

from mab_algorithms.ts import TSBandit
from mab_algorithms.ucb import UCBBandit
from mab_algorithms.kl_ucb import KLUCBBandit
from causal_engine.pomis import POMIS

app = typer.Typer()

@app.command()
def visualize_result(
    benchmark: str = 'chain_2.json'
):
    name = benchmark.split('.')[0]
    csv_path = Path("results") / f"{name}.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run benchmark first.")
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df[column], label=column, linewidth=2)
    
    plt.title(f"Benchmark Results: {name}", fontsize=14)
    plt.xlabel("Time Step (T)", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

@app.command()
def run_benchmark(
    benchmark: str = 'chain_2.json',
    num_seed: int = 5
)->None:
    # Total Five Random Seeds
    seeds = [42 * i for i in range(num_seed)]
    name = benchmark.split('.')[0]

    algorithms = ['ts', 'ucb', 'kl-ucb']
    with_pomis = dict()
    without_pomis = dict()

    # Use tqdm to track progress over seeds as requested
    for algorithm in algorithms:
        for flag in [True, False]:
            average = 0

            for seed in tqdm(seeds, desc=f"Algo: {algorithm.upper()}, POMIS: {flag}", leave=False):
                result = np.array(
                    run_experiment(
                        seed=seed,
                        algorithm=algorithm,
                        use_pomis=flag,
                        benchmark=benchmark,
                        T=10000,
                        show_graph=False
                    )
                )
                average += result
            average /= num_seed
            
            if flag:
                with_pomis[algorithm] = average

            else:
                without_pomis[algorithm] = average
    
    # Aggregate and Save results
    results_dict = {}
    for algo in algorithms:
        if algo in with_pomis:
            results_dict[f"{algo}_pomis"] = with_pomis[algo]
        if algo in without_pomis:
            results_dict[f"{algo}_no_pomis"] = without_pomis[algo]
    
    df = pd.DataFrame(results_dict)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

@app.command()
def run_experiment(
    seed: int = 42, 
    algorithm: str = 'ts',
    use_pomis: bool = True,
    benchmark: str = 'chain_2.json',
    T: int = 1000,
    optimality_estimation: int = 100000,
    show_graph = True
)->list:
    assert algorithm in ['ucb', 'kl-ucb', 'ts'], "Invalid Algorithm"

    env = Environ(seed=seed)
    env.load_graph(benchmark)
    env.allocate_weight()

    expected_optimal_reward = \
        env.get_optimal_expected_reward(n_samples=optimality_estimation)

    if algorithm == 'ucb':
        bandit = UCBBandit()
    elif algorithm == 'ts':
        bandit = TSBandit()
    else:
        bandit = KLUCBBandit()

    # Arm Selection
    if use_pomis:
        pomis_sets = POMIS(env.G)
        arms = env.get_arm_indices(pomis_sets)
    else:
        arms = env.get_arm_indices()
    
    bandit.set_arms(arms)
    
    # Learning Section
    regret = 0
    regret_per_time = []

    for t in range(T):
        arm_idx = bandit.select_arm()
        intervention = env.arms[arm_idx]
        
        sample = env.sample_node_values(interventions=intervention)
        reward = sample['Y']
        
        bandit.update(arm_idx, reward)
        
        # USE TRUE EXPECTED REWARD FOR MEASUREMENT
        regret += max(expected_optimal_reward - env.expected_rewards[arm_idx], 0)
        regret_per_time.append(regret)

    if show_graph:
        plt.plot(regret_per_time)
        plt.show()

    return regret_per_time

if __name__ == '__main__':
    app()