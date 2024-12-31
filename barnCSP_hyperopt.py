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
import pickle
import dill
import os
from src.search_in_2D.kmedoids_k_points_searcher import find_optimal_k_points_kmedoids_2D
from src.search_in_2D.uniform_grid_k_points_searcher import find_optimal_k_points_uniform_grid_search_2D
from src.search_in_2D.simulated_annealing_k_points_searcher import find_optimal_k_points_simulated_annealing_2D


from src.search_in_3D.tda_mapper_k_points_searcher import find_optimal_k_points_tda_3D
from src.search_in_3D.kmedoids_k_points_searcher import find_optimal_k_points_kmedoids_3D
from src.search_in_3D.uniform_grid_k_points_searcher import find_optimal_k_points_uniform_grid_search_3D

from datetime import datetime
from utils_optimize import *
#dagshub.init(repo_owner='AbhijithSBidaralli', repo_name='BarnCSP', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/AbhijithSBidaralli/BarnCSP.mlflow')
APP_CONFIG = {
    "results_path": "./results",
    "max_k_points": 1,
    "barn_section": 3.1500001,
}

UNIFORM_GRID_CONFIG = {
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
                obj.algorithm = 'tda-wrapper'
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

                #Trying loading
                if args.checkpoint.lower()=='y':
                    filename='hyperopt_trials.pkl'
                    with open(filename, 'rb') as f:
                        trials = pickle.load(f)
                    filename='hyperopt_obj.pkl'
                    with open(filename, 'rb') as f:
                        obj = pickle.load(f)
                    print('Best trial so far:',obj.best_trial)

                obj.dimension = '3D'
                obj.algorithm = 'tda-wrapper'
                best = fmin(
                    fn=obj.objective,
                    space=space_tda_mapper_3D,
                    algo=tpe.suggest,
                    max_evals=3,  # Adjust the number of trials
                    trials=trials
                )

                #hyperopt trial saving
                filename='hyperopt_trials.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(trials, f)
                filename='hyperopt_obj.pkl'
                with open(filename, 'wb') as f:
                    dill.dump(obj, f)

                print("Best parameters:", best)
                #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
                mlflow.log_param('best learning rate',discrete_learning_rates[best['learning_rate']])
                mlflow.log_param('best overlapping portion',best['integer_sample'])
                mlflow.log_param('best trial',obj.best_trial)
                mlflow.log_metric('best l2_norm_loss',obj.best_results)

    elif args.clusteringAlg.lower() == "kmedoids":
        print("[Status] Starting kmedoids k-point searcher ...")
        # Update the saving path
     

        # Search for k points in 2D
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
            now = datetime.now()
            mlflow.set_experiment("tda-2D-Concurrent")
            mlflow.end_run()
            with mlflow.start_run(run_name='kmedoids-2D'):
                start_date = now.strftime("%Y-%m-%d %H:%M:%S")
                mlflow.set_tag("start_date", start_date)
                discrete_learning_rates = [
                    1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3
                ]

                obj = objective_tda_2d(APP_CONFIG,mlflow,None)
                obj.dimension = '2D'
                obj.algorithm = 'kmedoids'
                
                for lr in discrete_learning_rates:
                    space = {}
                    space['lr']=lr
                    obj.objective(space)
                #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
                mlflow.log_param('best learning rate',obj.best_lr)
                mlflow.log_param('best trial',obj.best_trial)
                mlflow.log_metric('best l2_norm_loss',obj.best_results)

    elif args.clusteringAlg.lower() == "random":
        print("[Status] Starting random k-point searcher ...")

        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
        elif args.dim.lower()=='3d':
            print("[Status] Searching k points in the whole 3D space ...")
        now = datetime.now()
        mlflow.set_experiment("random-2D-3D-Concurrent")
        mlflow.end_run()
        with mlflow.start_run(run_name='random-{}'.format(args.dim.upper())):
                start_date = now.strftime("%Y-%m-%d %H:%M:%S")
                mlflow.set_tag("start_date", start_date)
                discrete_epochs = np.arange(5,51,5)

                obj = objective_tda_2d(APP_CONFIG,mlflow,None)
                obj.dimension = args.dim.upper()
                obj.algorithm = 'random'
                
                for epoch in discrete_epochs:
                    space = {}
                    space['epochs']=epoch
                    obj.objective(space)
                #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
                mlflow.log_param('best epoch',obj.best_epoch)
                mlflow.log_param('best trial',obj.best_trial)
                mlflow.log_metric('best l2_norm_loss',obj.best_results)

    elif args.clusteringAlg.lower() == "simulated-annealing":
        print("[Status] Starting simmulated annealing k-point searcher ...")
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
        elif args.dim.lower()=='3d':
            print("[Status] Searching k points in the whole 3D space ...")
        now = datetime.now()
        mlflow.set_experiment("tda-2D-Concurrent")
        mlflow.end_run()
        with mlflow.start_run(run_name='simAneal-{}'.format(args.dim.upper())):
            start_date = now.strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("start_date", start_date)
            values = np.arange(0.8,1.005,0.005)
            space_simulated_annealing = {
                'epochs': hp.quniform('epochs', 5, 50, 5),
                'temperature':hp.quniform('temperature', 50, 200, 10),
                'cooling_rate':hp.choice('cooling_rate',values)
            }
            trials = Trials()
            obj = objective_tda_2d(APP_CONFIG,mlflow,args.sec.upper())
            obj.dimension = args.dim.upper()
            obj.algorithm = 'simulated_annealing'
            best = fmin(
                fn=obj.objective,
                space=space_simulated_annealing,
                algo=tpe.suggest,
                max_evals=200,  # Adjust the number of trials
                trials=trials
            )

            print("Best parameters:", best)
            #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
            mlflow.log_param('best epoch',best['epochs'])
            mlflow.log_param('best temperature',best['temperature'])
            mlflow.log_param('best cooling_rate',values[best['cooling_rate']])
            mlflow.log_param('best trial',obj.best_trial)
            mlflow.log_metric('best l2_norm_loss',obj.best_results)
    
    elif args.clusteringAlg.lower() == "pso":
        print("[Status] Starting PSO k-point searcher ...")
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
        elif args.dim.lower()=='3d':
            print("[Status] Searching k points in the whole 3D space ...")
            now = datetime.now()
        mlflow.set_experiment("pso-2D-3D-Concurrent")
        mlflow.end_run()
        with mlflow.start_run(run_name='pso-{}'.format(args.dim.upper())):
            start_date = now.strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("start_date", start_date)
            c1_values = np.arange(0.5,5.10,0.1)
            c2_values = np.arange(0.5,5.10,0.1)
            w_values = np.arange(0.1,0.905,0.05)
            space_pso = {
                'epochs': hp.quniform('epochs', 5, 50, 5),
                'num_particles':hp.quniform('num_particles', 5, 100, 5),
                'c1':hp.choice('c1',c1_values),
                'c2':hp.choice('c2',c2_values),
                'w': hp.choice('w',w_values)
            }
            trials = Trials()
            obj = objective_tda_2d(APP_CONFIG,mlflow,args.sec.upper())
            obj.dimension = args.dim.upper()
            obj.algorithm = 'PSO'
            best = fmin(
                fn=obj.objective,
                space=space_pso,
                algo=tpe.suggest,
                max_evals=300,  # Adjust the number of trials
                trials=trials
            )
            print("Best parameters:", best)
            #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
            mlflow.log_param('best epoch',best['epochs'])
            mlflow.log_param('best num_particles',best['num_particles'])
            mlflow.log_param('best c1',best['c1'])
            mlflow.log_param('best c2',best['c2'])
            mlflow.log_param('best w',best['w'])
            mlflow.log_param('best trial',obj.best_trial)
            mlflow.log_metric('best l2_norm_loss',obj.best_results)
    elif args.clusteringAlg.lower() == "monte-carlo":
        print("[Status] Starting Monte-Carlo k-point searcher ...")
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
        now = datetime.now()
        mlflow.set_experiment("monteCarlo-2D-3D-Concurrent")
        mlflow.end_run()
        with mlflow.start_run(run_name='monte-{}'.format(args.dim.upper())):
            start_date = now.strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("start_date", start_date)
            discrete_convergence_thresh = [
                1e-9,5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5
            ]
            space_monte = {
                'epochs': hp.quniform('epochs', 5, 50, 5),
                'convergence_threshold':hp.choice('convergence_threshold', discrete_convergence_thresh)
            }
            trials = Trials()
            obj = objective_tda_2d(APP_CONFIG,mlflow,args.sec.upper())
            obj.dimension = args.dim.upper()
            obj.algorithm = 'Monte-Carlo'
            best = fmin(
                fn=obj.objective,
                space=space_monte,
                algo=tpe.suggest,
                max_evals=200,  # Adjust the number of trials
                trials=trials
            )
            print("Best parameters:", best)
            #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
            mlflow.log_param('best epoch',best['epochs'])
            mlflow.log_param('best convergence_threshold',discrete_convergence_thresh[best['convergence_threshold']])
            mlflow.log_param('best trial',obj.best_trial)
            mlflow.log_metric('best l2_norm_loss',obj.best_results)
    elif args.clusteringAlg.lower() == "genetic":
        print("[Status] Starting genetic k-point searcher ...")
        if args.dim.lower() == "2d":
            print(f"[Status] Searching k points in 2D at height {APP_CONFIG['barn_section']} ...")
        elif args.dim.lower() == "3d":
            print("[Status] Searching k points in the whole 3D space ...")
        now = datetime.now()
        mlflow.set_experiment("genetic-2D-3D-Concurrent")
        mlflow.end_run()
        with mlflow.start_run(run_name='genetic-{}'.format(args.dim.upper())):
            start_date = now.strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("start_date", start_date)
            '''"population_size": 100,
                "episodes": 20,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "tournament_size": 5,'''
            mutation_rate = np.arange(0.1,1.0,0.05)
            crossover_rate = np.arange(0.1,1.0,0.05)
            space_genetic = {
                'population_size': hp.quniform('population_size', 50, 200, 10),
                'episodes': hp.quniform('episodes', 5, 50, 5),
                'mutation_rate':hp.choice('mutation_rate', mutation_rate),
                'crossover_rate':hp.choice('crossover_rate', crossover_rate),
                'tournament_size': hp.quniform('tournament_size', 1, 10, 1)
            }
            trials = Trials()
            obj = objective_tda_2d(APP_CONFIG,mlflow,args.sec.upper())
            obj.dimension = args.dim.upper()
            obj.algorithm = 'genetic'
            best = fmin(
                fn=obj.objective,
                space=space_genetic,
                algo=tpe.suggest,
                max_evals=300,  # Adjust the number of trials
                trials=trials
            )
            print("Best parameters:", best)
            #mlflow.log_param('Clustering Algorithm','TDA-Mapper')
            mlflow.log_param('best population_size',best['population_size'])
            mlflow.log_param('best episodes',best['episodes'])
            mlflow.log_param('best mutation_rate',mutation_rate[best['mutation_rate']])
            mlflow.log_param('best crossover_rate',crossover_rate[best['crossover_rate']])
            mlflow.log_param('best tournament_size',best['tournament_size'])
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

    parser.add_argument(
        "-ch",
        "--checkpoint",
        type=str,
        default='N',
    )

    args = parser.parse_args()

    main(args)
