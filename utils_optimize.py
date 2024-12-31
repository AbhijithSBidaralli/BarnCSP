from src.search_in_2D.tda_mapper_k_points_searcher import find_optimal_k_points_tda_2D
from src.search_in_3D.tda_mapper_k_points_searcher import find_optimal_k_points_tda_3D
from src.search_in_2D.kmedoids_k_points_searcher import find_optimal_k_points_kmedoids_2D
from src.search_in_3D.kmedoids_k_points_searcher import find_optimal_k_points_kmedoids_3D
from src.search_in_2D.random_k_points_searcher import find_optimal_k_points_random_search_2D
from src.search_in_2D.simulated_annealing_k_points_searcher import find_optimal_k_points_simulated_annealing_2D
from src.search_in_2D.PSO_k_points_searcher import find_optimal_k_points_pso_2D
from src.search_in_2D.monte_carlo_k_points_searcher import find_optimal_k_points_monte_carlo_2D
from src.search_in_2D.genetic_k_points_searcher import find_optimal_k_points_advanced_genetic_algorithm_2D

from src.search_in_3D.random_k_points_searcher import find_optimal_k_points_random_search_3D
from src.search_in_3D.simulated_annealing_k_points_searcher import find_optimal_k_points_simulated_annealing_3D
from src.search_in_3D.PSO_k_points_searcher import find_optimal_k_points_pso_3D
from src.search_in_3D.monte_carlo_k_points_searcher import find_optimal_k_points_monte_carlo_3D
from src.search_in_3D.genetic_k_points_searcher import find_optimal_k_points_advanced_genetic_algorithm_3D
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
import json
from joblib import parallel_config,Parallel, delayed

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


class objective_tda_2d():
    def __init__(self,APP_CONFIG,mlflow,section):
        self.APP_CONFIG = APP_CONFIG
        self.final_results = []
        self.best_results = 1e12
        self.mlflow = mlflow
        self.count=1
        self.best_trial=''
        self.dimension = ''
        self.cross_section = section
        self.algorithm = ''
    def calculate_results(self,nodes_df,barn_inside,in_CO2_avg,barn_LW_ratio):
          if self.algorithm=='tda-wrapper':
                 TDA_MAPPER_CONFIG['cross_section']=self.cross_section
                 if self.dimension == '2D':
                            results = [
                                            find_optimal_k_points_tda_2D(
                                                nodes_df,
                                                barn_inside,
                                                i,
                                                in_CO2_avg,
                                                barn_section=self.APP_CONFIG["barn_section"],
                                                cross_section=TDA_MAPPER_CONFIG["cross_section"],
                                                overlap=self.overlap,
                                                lr=self.lr,
                                                epochs=TDA_MAPPER_CONFIG["epochs"],
                                                sampling_budget=TDA_MAPPER_CONFIG["sampling_budget"],
                                                neighborhood_numbers=TDA_MAPPER_CONFIG["neighborhood_numbers"],
                                                barn_LW_ratio=barn_LW_ratio,
                                            )
                                            for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                        ]
                 elif self.dimension== '3D':
                            results = [
                                                find_optimal_k_points_tda_3D(
                                                nodes_df,
                                                barn_inside,
                                                i,
                                                in_CO2_avg,
                                                cross_section=TDA_MAPPER_CONFIG["cross_section"],
                                                overlap=self.overlap,
                                                lr=self.lr,
                                                epochs=TDA_MAPPER_CONFIG["epochs"],
                                                sampling_budget=TDA_MAPPER_CONFIG["sampling_budget"],
                                                neighborhood_numbers=TDA_MAPPER_CONFIG["neighborhood_numbers"],
                                                barn_LW_ratio=barn_LW_ratio,
                                            )
                                            for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                        ]
          elif self.algorithm == 'kmedoids':
                 if self.dimension == '2D':
                                        results = [
                                                find_optimal_k_points_kmedoids_2D(
                                                nodes_df,
                                                barn_inside,
                                                i,
                                                in_CO2_avg,
                                                self.APP_CONFIG["barn_section"],
                                                lr=self.lr,
                                                epochs=KMEDOIDS_CONFIG["epochs"],
                                                sampling_budget=KMEDOIDS_CONFIG["sampling_budget"],
                                                neighborhood_numbers=KMEDOIDS_CONFIG["neighborhood_numbers"],
                                                barn_LW_ratio=barn_LW_ratio,
                                            )
                                            for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                        ]
                 elif self.dimension== '3D':
                                        results = [
                                            find_optimal_k_points_kmedoids_3D(
                                                nodes_df,
                                                barn_inside,
                                                i,
                                                in_CO2_avg,
                                                lr=self.lr,
                                                epochs=KMEDOIDS_CONFIG["epochs"],
                                                sampling_budget=KMEDOIDS_CONFIG["sampling_budget"],
                                                neighborhood_numbers=KMEDOIDS_CONFIG["neighborhood_numbers"],
                                                barn_LW_ratio=barn_LW_ratio,
                                            )
                                            for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                        ]
          elif self.algorithm == 'random':
                 if self.dimension == '2D':
                    results = [
                                        find_optimal_k_points_random_search_2D(
                                        nodes_df,
                                        barn_inside,
                                        i,
                                        in_CO2_avg,
                                        self.APP_CONFIG["barn_section"],
                                        epochs=self.epochs,
                                        sampling_budget=RANDOM_CONFIG["sampling_budget"],
                                        neighborhood_numbers=RANDOM_CONFIG["neighborhood_numbers"],
                                        barn_LW_ratio=barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                            ]
                 elif self.dimension == '3D':
                    results = [
                                        find_optimal_k_points_random_search_3D(
                                            nodes_df,
                                            barn_inside,
                                            i,
                                            in_CO2_avg,
                                            epochs=self.epochs,
                                            sampling_budget=RANDOM_CONFIG["sampling_budget"],
                                            neighborhood_numbers=RANDOM_CONFIG["neighborhood_numbers"],
                                            barn_LW_ratio=barn_LW_ratio,
                                        )
                                        for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                            ]
          elif self.algorithm == 'simulated_annealing':
                 if self.dimension == '2D':
                    results = [
                                        find_optimal_k_points_simulated_annealing_2D(
                                            nodes_df,
                                            barn_inside,
                                            i,
                                            in_CO2_avg,
                                            self.APP_CONFIG["barn_section"],
                                            sampling_budget=SIMULATED_ANNEALING_CONFIG["sampling_budget"],
                                            neighborhood_numbers=SIMULATED_ANNEALING_CONFIG["neighborhood_numbers"],
                                            epochs=int(self.epochs),
                                            initial_temperature=int(self.temperature),
                                            cooling_rate=self.cooling_rate,
                                            barn_LW_ratio=barn_LW_ratio,
                                        )
                                        for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                            ]
                 if self.dimension == '3D':
                    results = [
                                        find_optimal_k_points_simulated_annealing_3D(
                                            nodes_df,
                                            barn_inside,
                                            i,
                                            in_CO2_avg,
                                            sampling_budget=SIMULATED_ANNEALING_CONFIG["sampling_budget"],
                                            neighborhood_numbers=SIMULATED_ANNEALING_CONFIG["neighborhood_numbers"],
                                            epochs=int(self.epochs),
                                            initial_temperature=int(self.temperature),
                                            cooling_rate=self.cooling_rate,
                                            barn_LW_ratio=barn_LW_ratio,
                                        )
                                        for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                            ]
          elif self.algorithm == 'PSO':
                 PSO_CONFIG["num_particles"] = self.num_particles
                 if self.dimension == '2D':
                       results = [
                                    find_optimal_k_points_pso_2D(
                                        nodes_df,
                                        barn_inside,
                                        i,
                                        in_CO2_avg,
                                        self.APP_CONFIG["barn_section"],
                                        sampling_budget=PSO_CONFIG["sampling_budget"],
                                        neighborhood_numbers=PSO_CONFIG["neighborhood_numbers"],
                                        epochs=int(self.epochs),
                                        c1=self.c1,
                                        c2=self.c2,
                                        w=self.w,
                                        barn_LW_ratio=barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                            ]
                 if self.dimension == '3D':
                       results = [
                                    find_optimal_k_points_pso_3D(
                                        nodes_df,
                                        barn_inside,
                                        i,
                                        in_CO2_avg,
                                        sampling_budget=PSO_CONFIG["sampling_budget"],
                                        neighborhood_numbers=PSO_CONFIG["neighborhood_numbers"],
                                        epochs=int(self.epochs),
                                        c1=self.c1,
                                        c2=self.c2,
                                        w=self.w,
                                        barn_LW_ratio=barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                ]  
          elif self.algorithm == 'Monte-Carlo':
                 if self.dimension == '2D':
                        results = [
                                    find_optimal_k_points_monte_carlo_2D(
                                        nodes_df,
                                        barn_inside,
                                        i,
                                        in_CO2_avg,
                                        self.APP_CONFIG["barn_section"],
                                        sampling_budget=MONTE_CARLO_CONFIG["sampling_budget"],
                                        neighborhood_numbers=MONTE_CARLO_CONFIG["neighborhood_numbers"],
                                        max_epochs=int(self.epochs),
                                        convergence_threshold=self.convergence_threshold,
                                        barn_LW_ratio=barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                ]
                 if self.dimension == '3D':
                        results = [
                                    find_optimal_k_points_monte_carlo_3D(
                                        nodes_df,
                                        barn_inside,
                                        i,
                                        in_CO2_avg,
                                        sampling_budget=MONTE_CARLO_CONFIG["sampling_budget"],
                                        neighborhood_numbers=MONTE_CARLO_CONFIG["neighborhood_numbers"],
                                        max_epochs=int(self.epochs),
                                        convergence_threshold=self.convergence_threshold,
                                        barn_LW_ratio=barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                            ] 
          elif self.algorithm == 'genetic':
                if self.dimension == '2D':
                      results = [
                                    find_optimal_k_points_advanced_genetic_algorithm_2D(
                                        nodes_df,
                                        barn_inside,
                                        i,
                                        in_CO2_avg,
                                        self.APP_CONFIG["barn_section"],
                                        sampling_budget=GENETIC_CONFIG["sampling_budget"],
                                        neighborhood_numbers=GENETIC_CONFIG["neighborhood_numbers"],
                                        population_size=int(self.population_size),
                                        episodes=int(self.episodes),
                                        mutation_rate=self.mutation_rate,
                                        crossover_rate=self.crossover_rate,
                                        tournament_size=int(self.tournament_size),
                                        barn_LW_ratio=barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                ]
                if self.dimension == '3D':
                      results = [
                                    find_optimal_k_points_advanced_genetic_algorithm_3D(
                                        nodes_df,
                                        barn_inside,
                                        i,
                                        in_CO2_avg,
                                        sampling_budget=GENETIC_CONFIG["sampling_budget"],
                                        neighborhood_numbers=GENETIC_CONFIG["neighborhood_numbers"],
                                        population_size=int(self.population_size),
                                        episodes=int(self.episodes),
                                        mutation_rate=self.mutation_rate,
                                        crossover_rate=self.crossover_rate,
                                        tournament_size=int(self.tournament_size),
                                        barn_LW_ratio=barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                ]
          return results
    def process_file(self,barnFilename,directory,mean_losses):
         
                    print("[Status] Loading file ...")
                    self.APP_CONFIG["results_path"] = os.path.join(
                    os.path.join(self.APP_CONFIG["results_path"], self.algorithm),
                    barnFilename.split("/")[-1].split(".")[0],
        )
                    file_path = os.path.join(directory, barnFilename)
                    nodes_df = pd.read_csv(file_path).drop("Unnamed: 0", axis=1)
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
                        int(sub_str[-1]) for sub_str in barnFilename.split("_") if "LW" in sub_str
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
                    '''self.nodes_df = nodes_df
                    self.barn_inside = barn_inside
                    self.in_CO2_avg = in_CO2_avg
                    self.barn_LW_ratio = barn_LW_ratio'''
                    self.final_results = []
                    try:
                         results = self.calculate_results(nodes_df,barn_inside,in_CO2_avg,barn_LW_ratio)
                    except Exception as e:
                         print('**************')
                         print(e,' occurred in ',barnFilename)
                         print('**************')
                         exit(420)
                    #print(results)
                    if None in results:
                         results = [(0,1e5,0,np.array([])) if x is None else x for x in results]
                    mean_losses.extend([t[1] for t in results])
                
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
                    res_summary['file']=barnFilename
                    return mean_losses, res_summary
    def objective(self, space):
        algo_dict = {'tda-wrapper':'TDA','kmedoids':'KM','random':'RND','simulated_annealing':'SA','PSO':'PSO','Monte-Carlo':'MC','genetic':'GEN'}
        name='{}-{}-trial-{}'.format(algo_dict[self.algorithm],self.dimension,self.count)
        self.count+=1
        with self.mlflow.start_run(run_name = name,nested=True):
            if self.algorithm=='tda-wrapper':
                self.lr = space['lr']
                self.overlap = space['overlapping_portion']                
                self.mlflow.log_param('learning rate',self.lr)
                self.mlflow.log_param('overlapping portion',self.overlap)
            elif self.algorithm=='kmedoids':
                self.lr = space['lr']
                self.mlflow.log_param('learning rate',self.lr)
            elif self.algorithm=='random':
                self.epochs = space['epochs']
                self.mlflow.log_param('epochs',self.epochs)
            elif self.algorithm=='simulated_annealing':
                self.epochs = space['epochs']
                self.temperature = space['temperature']
                self.cooling_rate = space['cooling_rate']  
                self.mlflow.log_param('epochs',int(self.epochs))
                self.mlflow.log_param('temperature',int(self.temperature))
                self.mlflow.log_param('cooling_rate',self.cooling_rate)
            elif self.algorithm=='PSO':
                self.epochs = space['epochs']
                self.num_particles = space['num_particles']
                self.c1 = float(f"{space['c1']:.2f}")
                self.c2 = float(f"{space['c2']:.2f}")
                self.w = space['w']
                self.mlflow.log_param('epochs',int(self.epochs))
                self.mlflow.log_param('num_particles',int(self.num_particles))
                self.mlflow.log_param('c1',self.c1)
                self.mlflow.log_param('c2',self.c2)
                self.mlflow.log_param('w',self.w)
            elif self.algorithm=='Monte-Carlo':
                self.epochs = space['epochs']
                self.convergence_threshold = space['convergence_threshold']
                self.mlflow.log_param('epochs',int(self.epochs))
                self.mlflow.log_param('convergence_threshold',self.convergence_threshold)
            elif self.algorithm=='genetic':
                self.population_size = space['population_size']
                self.episodes = space['episodes']
                self.mutation_rate = space['mutation_rate']
                self.crossover_rate = space['crossover_rate']
                self.tournament_size = space['tournament_size']
                self.mlflow.log_param('population_size',int(self.population_size))
                self.mlflow.log_param('episodes',int(self.episodes))
                self.mlflow.log_param('mutation_rate',self.mutation_rate)
                self.mlflow.log_param('crossover_rate',self.crossover_rate)
                self.mlflow.log_param('tournament_size',int(self.tournament_size))
            directory = r'C:\Users\ALIAS\Data_new'
            # Get a list of all files in the directory
            all_files = os.listdir(directory)
            # Filter only the .csv files
            Files = [file for file in all_files if 'mix' in file.lower()]
            mean_losses=[]
            with parallel_config(backend='loky', n_jobs=44):              
                   combine_results = Parallel()(delayed(self.process_file)(i,directory,mean_losses) for i in Files)  
            mean_results, res_summary_results = zip(*combine_results)
            mean_losses = np.array(mean_results).flatten()
            res_summary_results = list(res_summary_results)
            with open('res_summary.json', "w") as f:
                json.dump(res_summary_results, f, indent=4)
            l2_norm_loss = np.linalg.norm(mean_losses)
            self.mlflow.log_metric('l2_norm_loss',l2_norm_loss)
            self.mlflow.log_artifact('res_summary.json')
            if l2_norm_loss < self.best_results:
                self.best_results = l2_norm_loss
                self.best_trial = name
                if self.algorithm=='kmedoids':
                    self.best_lr = self.lr
                if self.algorithm=='random':
                    self.best_epoch = self.epochs
            return l2_norm_loss