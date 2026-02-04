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

ALGO_CONFIG = {
    'ts_pomis': {'color': 'blue', 'linestyle': '-'},
    'ts_no_pomis': {'color': 'blue', 'linestyle': '--'},
    'ucb_pomis': {'color': 'green', 'linestyle': '-'},
    'ucb_no_pomis': {'color': 'green', 'linestyle': '--'},
    'kl-ucb_pomis': {'color': 'red', 'linestyle': '-'},
    'kl-ucb_no_pomis': {'color': 'red', 'linestyle': '--'},
}

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
        config = ALGO_CONFIG.get(column, {})
        plt.plot(
            df[column], 
            label=column, 
            linewidth=2,
            color=config.get('color'),
            linestyle=config.get('linestyle')
        )
    
    plt.title(f"Benchmark Results: {name}", fontsize=14)
    plt.xlabel("Time Step (T)", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

@app.command()
def run_benchmark(
    benchmark: str = 'chain_2.json',
    num_seed: int = 5,
    optimality_estimation: int = 1000000,
)->None:
    # Total Random Seeds
    seeds = [42 * i for i in range(num_seed)]
    name = benchmark.split('.')[0]

    algorithms = ['ts', 'ucb', 'kl-ucb']
    
    # Initialize results structures
    # results_dict[algo_name][seed_idx] = cumulative_regret_list
    total_results = {f"{algo}_{suffix}": [] for algo in algorithms for suffix in ["pomis", "no_pomis"]}

    # Seed loop at the outermost level for efficiency
    pbar = tqdm(seeds, desc="Benchmark", leave=True)
    for seed in pbar:
        # 1. Initialize environment once for this seed
        pbar.set_description(f"Seed {seed}: Initializing...")
        env = Environ(seed=seed)
        env.load_graph(benchmark)
        env.allocate_weight()

        # 2. Heavy Oracle Calculation: Once per seed
        # The inner tqdm in get_optimal_expected_reward handles its own status
        env.get_optimal_expected_reward(n_samples=optimality_estimation)
        
        # Determine optimal baseline from POMIS set to handle noise fairly
        pomis_sets_global = POMIS(env.G)
        pomis_arms_global = env.get_arm_indices(pomis_sets_global)
        expected_optimal_reward = max(env.expected_rewards[idx] for idx in pomis_arms_global)

        # 3. Run all algorithm/flag combinations for this seed
        for algorithm in algorithms:
            for flag in [True, False]:
                status = f"Seed {seed}: {algorithm.upper()} ({'POMIS' if flag else 'No-POMIS'})"
                pbar.set_description(status)
                
                key = f"{algorithm}_{'pomis' if flag else 'no_pomis'}"
                
                result = _run_experiment_core(
                    env=env,
                    expected_optimal_reward=expected_optimal_reward,
                    algorithm=algorithm,
                    use_pomis=flag,
                    T=10000
                )
                total_results[key].append(np.array(result))

    # 4. Average results across seeds and Save
    aggregated_dict = {}
    for key, results_list in total_results.items():
        if results_list:
            aggregated_dict[key] = np.mean(results_list, axis=0)
    
    df = pd.DataFrame(aggregated_dict)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

def _run_experiment_core(
    env: Environ,
    expected_optimal_reward: float,
    algorithm: str,
    use_pomis: bool,
    T: int
) -> list:
    """Core experiment loop that avoids re-initializing the environment."""
    
    # Initialize Bandit
    if algorithm == 'ucb':
        bandit = UCBBandit()
    elif algorithm == 'ts':
        bandit = TSBandit()
    else:
        bandit = KLUCBBandit()

    # Arm Selection
    if use_pomis:
        # Since we are in the core loop, environment graph G is already loaded
        pomis_sets = POMIS(env.G)
        arms = env.get_arm_indices(pomis_sets)
    else:
        arms = env.get_arm_indices()
    
    bandit.set_arms(arms)
    
    # Learning Section
    regret = 0
    regret_per_time = []

    for t in tqdm(range(T), desc="Steps", leave=False):
        arm_idx = bandit.select_arm()
        intervention = env.arms[arm_idx]
        
        sample = env.sample_node_values(interventions=intervention)
        reward = sample['Y']
        
        bandit.update(arm_idx, reward)
        
        # USE TRUE EXPECTED REWARD FOR MEASUREMENT
        regret += max(expected_optimal_reward - env.expected_rewards[arm_idx], 0)
        regret_per_time.append(regret)

    return regret_per_time

@app.command()
def run_experiment(
    seed: int = 42, 
    algorithm: str = 'ts',
    use_pomis: bool = True,
    benchmark: str = typer.Argument('chain_2.json', help="JSON file name in the benchmark directory"),
    T: int = 1000,
    optimality_estimation: int = 10000,
    show_graph = True
)->list:
    assert algorithm in ['ucb', 'kl-ucb', 'ts'], "Invalid Algorithm"

    env = Environ(seed=seed)
    env.load_graph(benchmark)
    env.allocate_weight()

    env.get_optimal_expected_reward(n_samples=optimality_estimation)
    
    # Use the best arm within POMIS sets as the baseline for regret
    pomis_sets = POMIS(env.G)
    pomis_arms = env.get_arm_indices(pomis_sets)
    expected_optimal_reward = max(env.expected_rewards[idx] for idx in pomis_arms)

    regret_per_time = _run_experiment_core(
        env=env,
        expected_optimal_reward=expected_optimal_reward,
        algorithm=algorithm,
        use_pomis=use_pomis,
        T=T
    )

    if show_graph:
        plt.plot(regret_per_time)
        plt.show()

    return regret_per_time

if __name__ == '__main__':
    app()