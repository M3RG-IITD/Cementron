import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dset', required=True, type=str)

args = parser.parse_args()

dset = args.dset

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

def run_inference(pycommand=None):
    print('Running: ',pycommand)
    os.system(pycommand)
    
resource_list = []
for th in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    for fold in range(1,5):
        for dset in [dset]:
            f = f'rgb_fourier_single_model_fold_{fold}_th_{th}_dataset_{dset}.csv'
            try:
                df = pd.read_csv(f)
                # if df.shape[0]==45:
                #     # print(f, df.shape)
                #     cmd = f'python rgb_fourier_thresholding_single_models.py --dset {dset} --threshold {th} --fold {fold}'
                #     print(cmd)
                #     resource_list.append(cmd)
            except:
                cmd = f'python rgb_fourier_thresholding_single_models.py --dset {dset} --threshold {th} --fold {fold}'
                print(cmd)
                resource_list.append(cmd)


num_cores = 4 #multiprocessing.cpu_count()
print(num_cores, len(resource_list))
inputs = tqdm(resource_list)
# if __name__ == "__main__":
processed_list = Parallel(n_jobs=num_cores)(delayed(run_inference)(i) for i in inputs)