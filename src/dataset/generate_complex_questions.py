import json
import os
import numpy as np

def not_connector():
    pass

def connector_not():
    pass


not_questions_path = '/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/langauge_augmentation_v2/val_total_not.json'
not_qs = open(not_questions_path, "r")
neg_data = json.load(not_qs)

not_and = []
not_or = []
and_not = []
or_not = []


language_aug_path = '/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/language_augmentation'
save_directory = '/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/langauge_augmentation_v2'
tot = 0

for file in os.listdir(language_aug_path):
        if file.endswith(".json") and "_orig" in file and "inter" not in file and "intra" not in file:
            file_path = os.path.join(language_aug_path, file)
            with open(file_path, "r") as f:
                data = json.load(f)
                print(f"Opened {file_path}")
                for q in data:
                    original_question = q['question']
                    filename = q['image_filename']

                    for not_q in neg_data:
                        if filename == not_q['image_filename']:
                            if len(not_q['question'])-4 == len(original_question):
                                continue
                            
                            not_and.append({
                                'image_filename': filename,
                                'question': f"{not_q['question'][:-1]} and {original_question[0].lower() + original_question[1:]}",
                                'q1 answer': not_q['ground_truth'],
                                'q2 answer': q['ground_truth'],
                                'ground_truth': 'yes' if not_q['ground_truth'] == 'yes' and q['ground_truth'] == 'yes' else 'no',
                                'model_generated_answer': "",
                                'consistent':""
                            })
                            
                            not_or.append({
                                'image_filename': filename,
                                'question': f"{not_q['question'][:-1]} or {original_question[0].lower() + original_question[1:]}",
                                'q1 answer': not_q['ground_truth'],
                                'q2 answer': q['ground_truth'],
                                'ground_truth': 'yes' if not_q['ground_truth'] == 'yes' or q['ground_truth'] == 'yes' else 'no',
                                'model_generated_answer': "",
                                'consistent':""
                            })

                            and_not.append({
                                'image_filename': filename,
                                'question': f"{original_question[:-1]} and {not_q['question'][0].lower() + not_q['question'][1:]}",
                                'q1 answer': q['ground_truth'],
                                'q2 answer': not_q['ground_truth'],
                                'ground_truth': 'yes' if q['ground_truth'] == 'yes' and not_q['ground_truth'] == 'yes' else 'no',
                                'model_generated_answer': "",
                                'consistent':""
                            })

                            or_not.append({
                                'image_filename': filename,
                                'question': f"{original_question[:-1]} or {not_q['question'][0].lower() + not_q['question'][1:]}",
                                'q1 answer': q['ground_truth'],
                                'q2 answer': not_q['ground_truth'],
                                'ground_truth': 'yes' if q['ground_truth'] == 'yes' or not_q['ground_truth'] == 'yes' else 'no',
                                'model_generated_answer': "",
                                'consistent':""
                            })

with open(os.path.join(save_directory, 'val_total_not_and.json'), 'w') as fp:
    json.dump(not_and, fp, indent=4)

with open(os.path.join(save_directory, 'val_total_not_or.json'), 'w') as fp:
    json.dump(not_or, fp, indent=4)

with open(os.path.join(save_directory, 'val_total_and_not.json'), 'w') as fp:
    json.dump(and_not, fp, indent=4)

with open(os.path.join(save_directory, 'val_total_or_not.json'), 'w') as fp:
    json.dump(or_not, fp, indent=4)