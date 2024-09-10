import os
import json
from tqdm import tqdm
import argparse
import torch
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import dagshub
import mlflow

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

from datetime import datetime
from utils_optimize import *
dagshub.init(repo_owner='AbhijithSBidaralli', repo_name='BarnCSP', mlflow=True)

APP_CONFIG = {
    "results_path": "./results",
    "max_k_points": 20,
    "barn_section": 3.1500001,
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
    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if args.clusteringAlg.lower() == "tda-mapper":
        print("[Status] Starting tda-mapper k-point searcher ...")
        # Update the saving path
     

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            now = datetime.now()
            mlflow.set_experiment("tda-2D-Concurrent")
            mlflow.end_run()
            with mlflow.start_run(run_name='tda-2D-{}-Cross-Section'.format(args.sec.upper())):
                start_date = now.strftime("%Y-%m-%d %H:%M:%S")
                mlflow.set_tag("start_date", start_date)
                discrete_learning_rates = [
                    1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
                ]

                space_tda_mapper_2D = {
                    'lr': hp.choice('learning_rate', discrete_learning_rates),
                    'overlapping_portion':hp.quniform('integer_sample', 10, 90, 1)
                }
                trials = Trials()
                obj = objective_tda_2d(APP_CONFIG,mlflow,args.sec.upper())
                obj.dimension = '2D'
                best = fmin(
                    fn=obj.objective,
                    space=space_tda_mapper_2D,
                    algo=tpe.suggest,
                    max_evals=125,  # Adjust the number of trials
                    trials=trials
                )
                print("Best parameters:", best)
                #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
                mlflow.log_param('best learning rate',discrete_learning_rates[best['learning_rate']])
                mlflow.log_param('best overlapping portion',best['integer_sample'])
                mlflow.log_param('best trial',obj.best_trial)
                mlflow.log_metric('best l2_norm_loss',obj.best_results)
                
        # Search for k points in 3D
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
            now = datetime.now()
            mlflow.set_experiment("tda-2D-Concurrent")
            mlflow.end_run()
            with mlflow.start_run(run_name='tda-3D-{}-Cross-Section'.format(args.sec.upper())):
                start_date = now.strftime("%Y-%m-%d %H:%M:%S")
                mlflow.set_tag("start_date", start_date)
                discrete_learning_rates = [
                    1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
                ]

                space_tda_mapper_3D = {
                    'lr': hp.choice('learning_rate', discrete_learning_rates),
                    'overlapping_portion':hp.quniform('integer_sample', 10, 90, 1)
                }
                trials = Trials()
                obj = objective_tda_2d(APP_CONFIG,mlflow,args.sec.upper())
                obj.dimension = '3D'
                best = fmin(
                    fn=obj.objective,
                    space=space_tda_mapper_3D,
                    algo=tpe.suggest,
                    max_evals=125,  # Adjust the number of trials
                    trials=trials
                )
                print("Best parameters:", best)
                #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
                mlflow.log_param('best learning rate',discrete_learning_rates[best['learning_rate']])
                mlflow.log_param('best overlapping portion',best['integer_sample'])
                mlflow.log_param('best trial',obj.best_trial)
                mlflow.log_metric('best l2_norm_loss',obj.best_results)

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do the k-point conditional sampling.")
    #parser.add_argument("barnFilename", type=str, help="the csv file of the barn")
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

    parser.add_argument(
        "-s",
        "--sec",
        type=str,
        default="Z",
        help="choose among Z/X/Y",
    )

    args = parser.parse_args()

    main(args)
