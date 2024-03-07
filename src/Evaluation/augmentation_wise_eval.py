import json
import numpy as np
import os
import glob


def aug_wise_analysis(baseline_path, augmentation_path, results_path, augmentation_analysis={}, baselines_analysis={}):
    for num in range(3, 11):
        base_path = os.path.join(baseline_path, f"val_num_{num}.json")
        baselines_file = open(base_path)
        baselines_object = json.load(baselines_file)

        baselines_dict = dict()
        for answer in baselines_object:
            if 'yes' in answer['consistent'].lower() or 'true' in answer['consistent'].lower():
                key = "correct"
            elif 'no' in answer['consistent'].lower() or 'false' in answer['consistent'].lower():
                key = "incorrect"
            else:
                key = "unknown"
        
            baselines_dict[answer['question_index']] = key


        aug_path = os.path.join(augmentation_path, f"val_num_{num}.json")
        aug_file = open(aug_path)
        json_object = json.load(aug_file)

        for answer in json_object:
            augmentation = answer['img_aug_conf']
            
            q_idx = answer['question_index']
            baseline_answer = baselines_dict[q_idx]
            baselines_analysis[augmentation] = baselines_analysis.get(augmentation, {'correct':0, 
                                                    'incorrect':0,
                                                    'total':0,
                                                    'unknown':0})
            if baseline_answer != "unknown":
                baselines_analysis[augmentation]['total'] += 1
            baselines_analysis[augmentation][baselines_dict[q_idx]] += 1


            augmentation_analysis[augmentation] = augmentation_analysis.get(augmentation, {'correct':0, 
                                                    'incorrect':0,
                                                    'total':0,
                                                    'unknown':0})
            augmentation_analysis[augmentation]['total'] += 1
            key = "unknown"
            consistent = answer["consistent"]
            if 'yes' in consistent.lower() or 'true' in consistent.lower():
                key = "correct"
            elif 'no' in consistent.lower() or 'false' in consistent.lower():
                key = "incorrect"
            else:
                augmentation_analysis[augmentation]['total'] -= 1


            augmentation_analysis[augmentation][key] += 1

    for key in augmentation_analysis:
        augmentation_analysis[key]['accuracy'] = augmentation_analysis[key]['correct']/(augmentation_analysis[key]['correct'] + augmentation_analysis[key]['incorrect'])

    for key in baselines_analysis:
        baselines_analysis[key]['accuracy'] = baselines_analysis[key]['correct']/(baselines_analysis[key]['correct'] + baselines_analysis[key]['incorrect'])


    results_file = open(results_path, 'w')
    json.dump([baselines_analysis, augmentation_analysis], results_file)
    results_file.close()


aug_wise_analysis('/media/exx/data/aayush/HallucinationsMLLM/MLLM_Hallucinations_CLEVR/outputs/baselines/Zephyr_Results/InstructBLIP',
                  '/media/exx/data/aayush/HallucinationsMLLM/MLLM_Hallucinations_CLEVR/outputs/Image_Augmentation/Zephyr_Results/InstructBLIP',
                  '/media/exx/data/aayush/HallucinationsMLLM/MLLM_Hallucinations_CLEVR/outputs/Image_Augmentation/Zephyr_Analysis/InstructBLIP/augmentation_wise_compiled_results.json')