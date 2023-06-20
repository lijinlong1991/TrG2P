import pandas as pd
import numpy as np

def summary(o):
    file_path = "./" + o
    direct_res = pd.read_csv(file_path + '/direct_res.csv', header=0)  
    #direct_res = pd.read_csv('./results/direct_res.csv', header=0) 
    #pretrain_res = pd.read_csv(file_path + '/pretaining_res.csv', header=0)
    fine_tuning_res = pd.read_csv(file_path + '/fine_tuning_res.csv', header=0)
    fusion_res = pd.read_csv(file_path + '/fusion_res.csv', header=0)
    res_all = pd.concat([direct_res, fine_tuning_res,fusion_res], sort=False)
    res_summary = res_all.groupby(['Task','Model','Epoch','Batch_size','Learning_rate']).agg([np.mean, np.std])
    print(res_summary)
    outputpath = file_path + '/results_summary.csv'
    res_summary.reset_index().to_csv(outputpath, encoding="utf_8_sig", index=False)