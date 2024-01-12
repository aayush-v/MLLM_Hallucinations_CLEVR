import torch
import json
#from PIL import Image
import requests
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import sys

from transformations import *


image_path = '/scratch/fbardoli/CLEVR_v1.0/images/val/'
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_path = '/scratch/fbardoli/CLEVR_v1.0/images/val/'
#object_type = ['inter', 'intra']
for i in range(3, 11):
    dataset = f'/home/fbardoli/MLLM_Hallucinations_CLEVR/dataset/object_number_split/val_num_{i}.json'
    output_path = f'/home/fbardoli/MLLM_Hallucinations_CLEVR/outputs/InstructBLIP/val_num_{i}.json'
    #dataset = f'/home/fbardoli/MLLM_Hallucinations_CLEVR/dataset/object_type_split/val_type_{i}.json'
    #output_path = f'/home/fbardoli/MLLM_Hallucinations_CLEVR/outputs/LLaVA/val_type_{i}.json'
    with open(dataset, 'r') as f:
        data = json.load(f)


    complete_data = []
    for each in tqdm(data):
        each_copy = each.copy()
        # img = Image.open(image_path+each['image_filename']).convert("RGB")
        img, aug_conf = pickMethod(image_path+each['image_filename'])
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
        each_copy['img_aug_conf'] = aug_conf
        # print(each_copy)
        complete_data.append(each_copy)


    with open(output_path, 'w') as f:
        json.dump(complete_data, f ,indent=4)
