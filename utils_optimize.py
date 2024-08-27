from src.search_in_2D.tda_mapper_k_points_searcher import find_optimal_k_points_tda_2D 
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
import json
from joblib import parallel_config,Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Array
TDA_MAPPER_CONFIG = {
    "cross_section": "Z",
    "overlapping_portion": 75,  # %
    "lr": 5e-7,
    "epochs": 20,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}


class objective_tda_2d():
    def __init__(self,APP_CONFIG,mlflow):
        self.nodes_df = 0
        self.barn_inside = 0
        self.in_CO2_avg = 0
        self.barn_LW_ratio = 0
        self.APP_CONFIG = APP_CONFIG
        self.final_results = []
        self.best_results = 1e5
        self.mlflow = mlflow
        self.count=1
        self.best_trial=''

    def process_file(self,barnFilename,directory,mean_losses):
         
                    print("[Status] Loading file ...")
                    self.APP_CONFIG["results_path"] = os.path.join(
                    os.path.join(self.APP_CONFIG["results_path"], "tda-mapper"),
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

                    self.nodes_df = nodes_df
                    self.barn_inside = barn_inside
                    self.in_CO2_avg = in_CO2_avg
                    self.barn_LW_ratio = barn_LW_ratio
                    self.final_results = []

                    results = [
                                    find_optimal_k_points_tda_2D(
                                        self.nodes_df,
                                        self.barn_inside,
                                        i,
                                        self.in_CO2_avg,
                                        barn_section=self.APP_CONFIG["barn_section"],
                                        cross_section=TDA_MAPPER_CONFIG["cross_section"],
                                        overlap=self.overlap,
                                        lr=self.lr,
                                        epochs=TDA_MAPPER_CONFIG["epochs"],
                                        sampling_budget=TDA_MAPPER_CONFIG["sampling_budget"],
                                        neighborhood_numbers=TDA_MAPPER_CONFIG["neighborhood_numbers"],
                                        barn_LW_ratio=self.barn_LW_ratio,
                                    )
                                    for i in tqdm(range(1, self.APP_CONFIG["max_k_points"] + 1))
                                ]
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
        name='TDA-2d-trial-{}'.format(self.count)
        self.count+=1
        with self.mlflow.start_run(run_name = name,nested=True):
            self.lr = space['lr']
            self.overlap = space['overlapping_portion']
            directory = r'C:\Users\ALIAS\Data_new'
            self.mlflow.log_param('learning rate',self.lr)
            self.mlflow.log_param('overlapping portion',self.overlap)
            # Get a list of all files in the directory
            all_files = os.listdir(directory)

            # Filter only the .csv files
            Files = [file for file in all_files if 'mix' in file.lower()]
            #Files = [r'C:\Users\ALIAS\Data\Yplane_LW2_Open_Mix_15-0_0-97_0deg.csv',r'C:\Users\ALIAS\Data\Yplane_LW2_Open_Mix_10-0_0-85_45deg.csv']
            mean_losses=[]
            with parallel_config(backend='loky', n_jobs=40):              
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
                #print('best results loss is ',self.best_results)
                self.best_trial = name
            return l2_norm_loss