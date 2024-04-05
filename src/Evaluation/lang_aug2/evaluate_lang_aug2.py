import numpy as np
import json
import os

ALLOWED_IDX_FOLDER_PATH = '/scratch/averma90/MLLM_Hallucinations_CLEVR/src/Evaluation/lang_aug2'
CONSISTENCY_FOLDER = '/scratch/averma90/MLLM_Hallucinations_CLEVR/outputs/lang_aug_2/Zephyr_Results'

def evalConsistency(model_answer_dir, model):
    model_answer_path = os.path.join(model_answer_dir, model)
    pass
    templates = os.listdir(model_answer_path)

    for template in templates:
        print(template)
        final_answers = []
        total = 0
        consistency_results = []
        fp = open(os.path.join(model_answer_path, template),)
        answer_objects = json.load(fp)

        allowed_idx_path = os.path.join(ALLOWED_IDX_FOLDER_PATH, f'{template}_balanced_ids.npz')
        allowed_idx = np.load(allowed_idx_path)

        selected_idx_array1 = allowed_idx['selected_idx_array1']
        selected_idx_array2 = allowed_idx['selected_idx_array2']

        save_path = os.path.join(CONSISTENCY_FOLDER, model)



        for idx in selected_idx_array1:
            object = answer_objects[idx]

            if object['model_generated_answer'].lower().startswith('yes'):
                object['consistent'] = 'yes'
            elif object['model_generated_answer'].lower().startswith('no'):
                object['consistent'] = 'no'
            else:
                continue
            
            final_answers.append(object)

            
            # if 'yes' in object['model_generated_answer'].lower() and 'no' in object['model_generated_answer'].lower():
            #     unknown.append(idx)
            # elif 'yes' not in object['model_generated_answer'].lower() and 'no' not in object['model_generated_answer'].lower():
            #     unknown2 += 1

            if object['model_generated_answer'].lower().startswith('yes') or object['model_generated_answer'].lower().startswith('no'):
                total+=1
            
            # if object['ground_truth'].lower() == 'no':
            #     print(object)

        for idx in selected_idx_array2:
            object = answer_objects[idx]

            if object['model_generated_answer'].lower().startswith('no'):
                object['consistent'] = 'yes'
            elif object['model_generated_answer'].lower().startswith('yes'):
                object['consistent'] = 'no'
            else:
                continue

            final_answers.append(object)


            # if 'yes' in object['model_generated_answer'].lower() and 'no' in object['model_generated_answer'].lower():
            #     unknown.append(idx)
            # elif 'yes' not in object['model_generated_answer'].lower() and 'no' not in object['model_generated_answer'].lower():
            #     unknown2 += 1

            if object['model_generated_answer'].lower().startswith('yes') or object['model_generated_answer'].lower().startswith('no'):
                total+=1

            # if object['ground_truth'].lower() == 'yes':
            #     print(object)

            # if object['ground_truth'].lower() == 'yes' and 'yes' in object['model_generated_answer'].lower() and 'no' not in object['model_generated_answer'].lower():
            #     object['consistency'] = 'yes'
            # elif object['ground_truth'].lower() == 'no' and 'no' in object['model_generated_answer'].lower() and 'yes' not in object['model_generated_answer'].lower():
            #     object['consistency'] = 'yes'
            # elif 


        with open(os.path.join(save_path, template), 'w') as json_file:
            json.dump(final_answers, json_file)
        print(total, len(selected_idx_array1)*2)


evalConsistency('/scratch/averma90/MLLM_Hallucinations_CLEVR/outputs/lang_aug_2/Model_Results', 'InstructBLIP')