import torch
import json
from PIL import Image
import requests
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import sys

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", cache_dir='/scratch/averma90')
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", cache_dir='/scratch/averma90')

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_path = '/scratch/averma90/CLEVR_v1.0/images/val/'
ques_type = ['and_not', 'not_and', 'not_or', 'or_not', 'not']
# object_type = ['inter']
ques_type = [sys.argv[1]]
for ques in ques_type:
    #if i == 'inter' or i == 'intra' :
    dataset = f'/scratch/averma90/MLLM_Hallucinations_CLEVR/dataset/langauge_augmentation_v2/val_total_{ques}.json'
    output_path = f'/scratch/averma90/MLLM_Hallucinations_CLEVR/outputs/language_augmentation/InstructBLIP_2/val_total_{ques}.json'
    print(dataset)
    # else:
    #     dataset = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/dataset/language_augmentation/val_num_{i}_{ques}.json'
    #     output_path = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/outputs/language_augmentation/InstructBLIP/val_num_{i}_{ques}.json'
    with open(dataset, 'r') as f:
        data = json.load(f)


    complete_data = []
    for each in tqdm(data):
        each_copy = each.copy()
        img = Image.open(image_path+each['image_filename']).convert("RGB")
        prompt = each['question']
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1,
            top_k=50,
            temperature=1e-05
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        each_copy['model_generated_answer'] = generated_text
        #print(each_copy)
        complete_data.append(each_copy)


    with open(output_path, 'w') as f:
        json.dump(complete_data, f ,indent=4)



