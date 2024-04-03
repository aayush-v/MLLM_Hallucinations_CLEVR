import json
import numpy as np
import os

def createNOT_dataset(path):
    pass



final_not_questions = []
print('imported')
language_aug_path = '/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/language_augmentation'
save_directory = '/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/langauge_augmentation_v2'
tot = 0

for file in os.listdir(language_aug_path):
        if file.endswith(".json") and "_orig" in file and "inter" not in file and "intra" not in file:
            file_path = os.path.join(language_aug_path, file)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    print(f"Opened {file_path}")
                    for q in data:
                        original_question = q['question']
                        original_answer = q['ground_truth']
                        not_question = ''
                        not_answer = "no" if original_answer.lower() == "yes" else "yes"

                        if "Is there" in original_question or "Are there" in original_question:
                            tot += 1
                            if original_question.startswith("Is there"):
                                not_question = "Is there not" + original_question[8:]
                            elif original_question.startswith("Are there"):
                                not_question = "Are there not" + original_question[9:]
                              
                            final_not_questions.append({
                                'image_filename': q['image_filename'],
                                'question': not_question,
                                'q1 answer': not_answer,
                                'q2 answer': 'NA',
                                'ground_truth': not_answer,
                                'model_generated_answer': "",
                                'consistent':""
                            })

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON file {file_path}: {e}")
            print(tot)

not_file = open(os.path.join(save_directory, 'val_total_not.json'), 'w')
json.dump(final_not_questions, not_file, indent=4)