import os
import json
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope

from src.search_in_2D.kmedoids_k_points_searcher import find_optimal_k_points_kmedoids_2D
from src.search_in_2D.random_k_points_searcher import find_optimal_k_points_random_search_2D
from src.search_in_2D.uniform_grid_k_points_searcher import find_optimal_k_points_uniform_grid_search_2D
from src.search_in_2D.simulated_annealing_k_points_searcher import find_optimal_k_points_simulated_annealing_2D
from src.search_in_2D.PSO_k_points_searcher import find_optimal_k_points_pso_2D
from src.search_in_2D.monte_carlo_k_points_searcher import find_optimal_k_points_monte_carlo_2D
from src.search_in_2D.genetic_k_points_searcher import find_optimal_k_points_advanced_genetic_algorithm_2D

from src.search_in_3D.tda_mapper_k_points_searcher import find_optimal_k_points_tda_3D
from src.search_in_3D.kmedoids_k_points_searcher import find_optimal_k_points_kmedoids_3D
from src.search_in_3D.random_k_points_searcher import find_optimal_k_points_random_search_3D
from src.search_in_3D.uniform_grid_k_points_searcher import find_optimal_k_points_uniform_grid_search_3D
from src.search_in_3D.simulated_annealing_k_points_searcher import find_optimal_k_points_simulated_annealing_3D
from src.search_in_3D.PSO_k_points_searcher import find_optimal_k_points_pso_3D
from src.search_in_3D.monte_carlo_k_points_searcher import find_optimal_k_points_monte_carlo_3D
from src.search_in_3D.genetic_k_points_searcher import find_optimal_k_points_advanced_genetic_algorithm_3D

from utils_optimize import *

APP_CONFIG = {
    "results_path": "./results",
    "max_k_points": 3,
    "barn_section": 3.1500001,
}
TDA_MAPPER_CONFIG = {
    "cross_section": "Z",
    "overlapping_portion": 75,  # %
    "lr": 5e-7,
    "epochs": 20,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
KMEDOIDS_CONFIG = {
    "lr": 5e-7,
    "epochs": 20,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
RANDOM_CONFIG = {
    "epochs": 20,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
UNIFORM_GRID_CONFIG = {
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
SIMULATED_ANNEALING_CONFIG = {
    "epochs": 20,
    "initial_temperature": 100,
    "cooling_rate": 0.995,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
PSO_CONFIG = {
    "epochs": 20,
    "num_particles": 20,
    "c1": 1.5,
    "c2": 1.5,
    "w": 0.7,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
MONTE_CARLO_CONFIG = {
    "max_epochs": 20,
    "convergence_threshold": 1e-7,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
GENETIC_CONFIG = {
    "population_size": 100,
    "episodes": 20,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "tournament_size": 5,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}

def main(args):
    # Load csv barn file
    print("[Status] Loading file ...")
    nodes_df = pd.read_csv(args.barnFilename).drop("Unnamed: 0", axis=1)
    nodes_df.rename(
        columns={
            "X [ m ]": "X",
            " Y [ m ]": "Y",
            " Z [ m ]": "Z",
            " Carbon Dioxide.Mass Fraction": "Carbon",
            " Velocity u [ m s^-1 ]": "u",
            " Velocity v [ m s^-1 ]": "v",
            " Velocity w [ m s^-1 ]": "w",
        },
        inplace=True,
    )

    # Get LW ratio from the filename
    barn_LW_ratio = [
        int(sub_str[-1]) for sub_str in args.barnFilename.split("_") if "LW" in sub_str
    ][0]

    # Filtering the barn interior and let out the outside
    print("[Status] Finding the barn-inside region ...")
    barn_inside = np.zeros((50, 100 * barn_LW_ratio, 100))
    carbon_image = nodes_df["Carbon"].values.reshape((50, 100 * barn_LW_ratio, 100))
    barn_inside[:20, :, :] = 1
    for j in range(20, 50):
        x = np.array([carbon_image[j, :, i].sum() for i in range(100)])
        for idx, val in enumerate([x[-i] - x[-i - 1] for i in range(1, 60)]):
            if val < -200:
                barn_inside[j, :, (0 + idx) : (100 - idx)] = 1
                break

    # Calculating the average CO2 concentration inside the barn
    in_CO2_avg = np.mean(nodes_df[barn_inside.flatten().astype(bool)]["Carbon"].values)
    
    if args.clusteringAlg.lower() == "tda-mapper":
        print("[Status] Starting tda-mapper k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "tda-mapper"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            discrete_learning_rates = [
                1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
            ]

            space_tda_mapper_2D = {
                'lr': hp.choice('learning_rate', discrete_learning_rates),
                'overlapping_portion':hp.quniform('integer_sample', 10, 90, 1)
            }
            trials = Trials()
            obj = objective_tda_2d(nodes_df,barn_inside,in_CO2_avg,barn_LW_ratio,APP_CONFIG)
            best = fmin(
                fn=obj.objective,
                space=space_tda_mapper_2D,
                algo=tpe.suggest,
                max_evals=2,  # Adjust the number of trials
                trials=trials
            )
            print("Best parameters:", best)
            results = obj.final_results
        # Search for k points in 3D
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_tda_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    cross_section=TDA_MAPPER_CONFIG["cross_section"],
                    overlap=TDA_MAPPER_CONFIG["overlapping_portion"],
                    lr=TDA_MAPPER_CONFIG["lr"],
                    epochs=TDA_MAPPER_CONFIG["epochs"],
                    sampling_budget=TDA_MAPPER_CONFIG["sampling_budget"],
                    neighborhood_numbers=TDA_MAPPER_CONFIG["neighborhood_numbers"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]

    elif args.clusteringAlg.lower() == "kmedoids":
        print("[Status] Starting kmedoids k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "kmedoids"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            results = [
                find_optimal_k_points_kmedoids_2D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    APP_CONFIG["barn_section"],
                    lr=KMEDOIDS_CONFIG["lr"],
                    epochs=KMEDOIDS_CONFIG["epochs"],
                    sampling_budget=KMEDOIDS_CONFIG["sampling_budget"],
                    neighborhood_numbers=KMEDOIDS_CONFIG["neighborhood_numbers"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_kmedoids_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    lr=KMEDOIDS_CONFIG["lr"],
                    epochs=KMEDOIDS_CONFIG["epochs"],
                    sampling_budget=KMEDOIDS_CONFIG["sampling_budget"],
                    neighborhood_numbers=KMEDOIDS_CONFIG["neighborhood_numbers"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
    elif args.clusteringAlg.lower() == "random":
        print("[Status] Starting random k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "random"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            results = [
                find_optimal_k_points_random_search_2D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    APP_CONFIG["barn_section"],
                    epochs=RANDOM_CONFIG["epochs"],
                    sampling_budget=RANDOM_CONFIG["sampling_budget"],
                    neighborhood_numbers=RANDOM_CONFIG["neighborhood_numbers"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_random_search_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    epochs=RANDOM_CONFIG["epochs"],
                    sampling_budget=RANDOM_CONFIG["sampling_budget"],
                    neighborhood_numbers=RANDOM_CONFIG["neighborhood_numbers"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
    elif args.clusteringAlg.lower() == "uniform":
        print("[Status] Starting uniform k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "uniform"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            results = [
                find_optimal_k_points_uniform_grid_search_2D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    APP_CONFIG["barn_section"],
                    sampling_budget=UNIFORM_GRID_CONFIG["sampling_budget"],
                    neighborhood_numbers=UNIFORM_GRID_CONFIG["neighborhood_numbers"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_uniform_grid_search_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    sampling_budget=UNIFORM_GRID_CONFIG["sampling_budget"],
                    neighborhood_numbers=UNIFORM_GRID_CONFIG["neighborhood_numbers"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
    elif args.clusteringAlg.lower() == "simulated-annealing":
        print("[Status] Starting simmulated annealing k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "simulated_annealing"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            results = [
                find_optimal_k_points_simulated_annealing_2D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    APP_CONFIG["barn_section"],
                    sampling_budget=SIMULATED_ANNEALING_CONFIG["sampling_budget"],
                    neighborhood_numbers=SIMULATED_ANNEALING_CONFIG["neighborhood_numbers"],
                    epochs=SIMULATED_ANNEALING_CONFIG["epochs"],
                    initial_temperature=SIMULATED_ANNEALING_CONFIG["initial_temperature"],
                    cooling_rate=SIMULATED_ANNEALING_CONFIG["cooling_rate"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_simulated_annealing_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    sampling_budget=SIMULATED_ANNEALING_CONFIG["sampling_budget"],
                    neighborhood_numbers=SIMULATED_ANNEALING_CONFIG["neighborhood_numbers"],
                    epochs=SIMULATED_ANNEALING_CONFIG["epochs"],
                    initial_temperature=SIMULATED_ANNEALING_CONFIG["initial_temperature"],
                    cooling_rate=SIMULATED_ANNEALING_CONFIG["cooling_rate"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
    elif args.clusteringAlg.lower() == "pso":
        print("[Status] Starting PSO k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "PSO"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            results = [
                find_optimal_k_points_pso_2D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    APP_CONFIG["barn_section"],
                    sampling_budget=PSO_CONFIG["sampling_budget"],
                    neighborhood_numbers=PSO_CONFIG["neighborhood_numbers"],
                    epochs=PSO_CONFIG["epochs"],
                    c1=PSO_CONFIG["c1"],
                    c2=PSO_CONFIG["c2"],
                    w=PSO_CONFIG["w"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_pso_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    sampling_budget=SIMULATED_ANNEALING_CONFIG["sampling_budget"],
                    neighborhood_numbers=SIMULATED_ANNEALING_CONFIG["neighborhood_numbers"],
                    epochs=SIMULATED_ANNEALING_CONFIG["epochs"],
                    c1=PSO_CONFIG["c1"],
                    c2=PSO_CONFIG["c2"],
                    w=PSO_CONFIG["w"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
    elif args.clusteringAlg.lower() == "monte-carlo":
        print("[Status] Starting Monte-Carlo k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "Monte-Carlo"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            results = [
                find_optimal_k_points_monte_carlo_2D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    APP_CONFIG["barn_section"],
                    sampling_budget=MONTE_CARLO_CONFIG["sampling_budget"],
                    neighborhood_numbers=MONTE_CARLO_CONFIG["neighborhood_numbers"],
                    max_epochs=MONTE_CARLO_CONFIG["max_epochs"],
                    convergence_threshold=MONTE_CARLO_CONFIG["convergence_threshold"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_monte_carlo_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    sampling_budget=MONTE_CARLO_CONFIG["sampling_budget"],
                    neighborhood_numbers=MONTE_CARLO_CONFIG["neighborhood_numbers"],
                    max_epochs=MONTE_CARLO_CONFIG["max_epochs"],
                    convergence_threshold=MONTE_CARLO_CONFIG["convergence_threshold"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
    elif args.clusteringAlg.lower() == "genetic":
        print("[Status] Starting genetic k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "genetic"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            results = [
                find_optimal_k_points_advanced_genetic_algorithm_2D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    APP_CONFIG["barn_section"],
                    sampling_budget=GENETIC_CONFIG["sampling_budget"],
                    neighborhood_numbers=GENETIC_CONFIG["neighborhood_numbers"],
                    population_size=GENETIC_CONFIG["population_size"],
                    episodes=GENETIC_CONFIG["episodes"],
                    mutation_rate=GENETIC_CONFIG["mutation_rate"],
                    crossover_rate=GENETIC_CONFIG["crossover_rate"],
                    tournament_size=GENETIC_CONFIG["tournament_size"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            results = [
                find_optimal_k_points_advanced_genetic_algorithm_3D(
                    nodes_df,
                    barn_inside,
                    i,
                    in_CO2_avg,
                    sampling_budget=GENETIC_CONFIG["sampling_budget"],
                    neighborhood_numbers=GENETIC_CONFIG["neighborhood_numbers"],
                    population_size=GENETIC_CONFIG["population_size"],
                    episodes=GENETIC_CONFIG["episodes"],
                    mutation_rate=GENETIC_CONFIG["mutation_rate"],
                    crossover_rate=GENETIC_CONFIG["crossover_rate"],
                    tournament_size=GENETIC_CONFIG["tournament_size"],
                    barn_LW_ratio=barn_LW_ratio,
                )
                for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
            ]

    # Prepare a dictionary for saving into a json file
    res_summary = {}
    for i in range(len(results)):
        if results[i] is not None:
            res_summary[f"{i+1}-point"] = {}
            res_summary[f"{i+1}-point"]["Min Loss"] = float(
                results[i][0].detach().numpy() if "torch" in str(type(results[i][0])) else float(results[i][0])
            )
            res_summary[f"{i+1}-point"]["Mean Loss"] = results[i][1]
            res_summary[f"{i+1}-point"]["Std Loss"] = results[i][2]
            res_summary[f"{i+1}-point"][f"{i+1} Points' Position"] = [
                [l for l in j] for j in results[i][3]
            ]

    print("[Status] Saving ...")
    try:
        if args.clusteringAlg == "tda-mapper":
            with open(APP_CONFIG["results_path"] + "_" + args.dim.lower() + TDA_MAPPER_CONFIG['cross_section'] + ".json", "w") as fp:
                json.dump(res_summary, fp)
        else:
            with open(APP_CONFIG["results_path"] + "_" + args.dim.lower() + ".json", "w") as fp:
                json.dump(res_summary, fp)
    except OSError:
        os.mkdir("/".join(APP_CONFIG["results_path"].split("/")[:-1]))
        if args.clusteringAlg == "tda-mapper":
            with open(APP_CONFIG["results_path"] + "_" + args.dim.lower() + TDA_MAPPER_CONFIG['cross_section'] + ".json", "w") as fp:
                json.dump(res_summary, fp)
        else:
            with open(APP_CONFIG["results_path"] + "_" + args.dim.lower() + ".json", "w") as fp:
                json.dump(res_summary, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do the k-point conditional sampling.")
    parser.add_argument("barnFilename", type=str, help="the csv file of the barn")
    parser.add_argument(
        "-c",
        "--clusteringAlg",
        type=str,
        default="tda-mapper",
        help="choose among tda-mapper/kmedoids/random/uniform/simulated-annealing/PSO/monte-carlo/genetic",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=str,
        default="2D",
        help="choose among 2D/3D",
    )

    args = parser.parse_args()

    main(args)
