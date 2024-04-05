import numpy as np
import json
import os

def balanceAndSave(path, filename):
    np.random.seed(42)
    yes = []
    no = []

    fp = open(path,)
    objects = json.load(fp)

    for idx, obj in enumerate(objects):
        if obj['ground_truth'].lower() == 'yes':
            yes.append(idx)
        elif obj['ground_truth'].lower() == 'no':
            no.append(idx)

    smaller_class_size = min(len(yes), len(no))

    selected_idx_array1 = np.random.choice(yes, size=smaller_class_size, replace=False)
    selected_idx_array2 = np.random.choice(no, size=smaller_class_size, replace=False)

    selected_idx_array1 = np.sort(selected_idx_array1)  
    selected_idx_array2 = np.sort(selected_idx_array2)  

    np.savez(filename + '_balanced_ids.npz', selected_idx_array1=selected_idx_array1, selected_idx_array2=selected_idx_array2)

    print(smaller_class_size)
    print(selected_idx_array1, len(selected_idx_array1), len(yes))
    print(selected_idx_array2, len(selected_idx_array2), len(no))

dataset_dir = '/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/langauge_augmentation_v2/'

for filename in os.listdir(dataset_dir):
        if os.path.isfile(os.path.join(dataset_dir, filename)):
            print(filename)  

            balanceAndSave(os.path.join(dataset_dir, filename), filename)