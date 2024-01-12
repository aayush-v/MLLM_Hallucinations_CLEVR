import cv2
import torch
from PIL import Image
import llama
import json
from tqdm import tqdm

image_path = '/scratch/nmachav1/CLEVR_v1.0/images/val/'
device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = '/scratch/nmachav1/MLLM_Hallucinations_CLEVR/scripts/LLaMA/LLaMA-Adapter/llama_adapter_v2_multimodal7b/llama'

model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type="7B", device=device)
model.eval()

for i in range(3,11):
    dataset = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/dataset/object_number_split/val_num_{i}.json'
    output_path = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/outputs/LLaMA/val_num_{i}.json'
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    
    complete_data = []
    for each in tqdm(data): 
        each_copy = each.copy()
        prompt = llama.format_prompt(each['question'])
        img = Image.fromarray(cv2.imread(image_path+each['image_filename']))
        img = preprocess(img).unsqueeze(0).to(device)
        result = model.generate(img, [prompt])[0]
        each_copy['model_generated_answer'] = result
        complete_data.append(each_copy)
        
    
    with open(output_path, 'w') as f:
        json.dump(complete_data, f ,indent=4)