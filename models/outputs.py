import pandas as pd
import numpy as np
import glob

class Output:
    def __init__(self, exp_path) :
        self.exp_path = exp_path
        
    
    def parse(self) :
        print('running...')
        # read all files within the model_output directory with .csv using glob

        # read in all files
        files = glob.glob(f'{self.exp_path}/*.csv')
        print("read all files")
        # combine into a single dataframe
        results = pd.concat([pd.read_csv(f) for f in files])
        n = len(results)
        print(f"combined into a single dataframe with {n} rows")
        results.to_csv(f'{self.exp_path}/../speaker_df_nocost.csv')
        print('done')
    
Output('../data/exp2/model_output/nocostgrid').parse()