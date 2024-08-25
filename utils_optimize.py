from src.search_in_2D.tda_mapper_k_points_searcher import find_optimal_k_points_tda_2D 
from tqdm import tqdm
import numpy as np

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
    def __init__(self, nodes_df,barn_inside,in_CO2_avg,barn_LW_ratio,APP_CONFIG):
        self.nodes_df = nodes_df
        self.barn_inside = barn_inside
        self.in_CO2_avg = in_CO2_avg
        self.barn_LW_ratio = barn_LW_ratio
        self.APP_CONFIG = APP_CONFIG
        self.final_results = []
        self.best_results = 1e5
    
    def objective(self, space):
        self.lr = space['lr']
        self.overlap = space['overlapping_portion']
        print('chosen hyperparameters')
        print(self.lr)
        print(self.overlap)
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
        mean_losses=[t[1] for t in results]
        self.results=results
        l2_norm_loss = np.linalg.norm(mean_losses)
        print(l2_norm_loss)
        if l2_norm_loss < self.best_results:
            self.best_results = l2_norm_loss
            print('best results loss is ',self.best_results)
            self.final_results=results
        return l2_norm_loss