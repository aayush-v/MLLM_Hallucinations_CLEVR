import numpy as np
import os
import glob
import json

def findLangAugDatasetSplit(files):

    stats = {'or': {'yes': 0, 'no': 0},
             'and': {'yes': 0, 'no': 0},
             'not': {'yes': 0, 'no': 0},
             'complex': {'yes': 0, 'no': 0},
             'orig': {'yes': 0, 'no': 0}}
    
    for filename in files:
        with open(filename, 'r') as file:
            dataset = json.load(file)

            for val in dataset:
                stats[filename.split('_')[-1].split('.')[0]][val['ground_truth']] += 1

    return stats

if __name__ == "__main__":
    LANG_AUG_DATASET_PATH = '/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/language_augmentation'
    files = glob.glob(os.path.join(LANG_AUG_DATASET_PATH, '**/*.json'), recursive=True)

    lang_stats = findLangAugDatasetSplit(files)
    print(lang_stats)
    # {'or': {'yes': 20984, 'no': 8190}, 
    # 'and': {'yes': 7102, 'no': 22072}, 
    # 'not': {'yes': 30262, 'no': 28086}, 
    # 'complex': {'yes': 60524, 'no': 56172}, 
    # 'orig': {'yes': 28086, 'no': 30262}}
    
   
