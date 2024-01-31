import json
from tqdm import tqdm
import os
import glob
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
device = "cuda"
model.to(device)

question_structure = "I will use you as an evaluator. I will give you ground truth, and a model generated answer. I want you to tell me if the ground truth and model generated answer are consistent."

def performEvaluation(path):
    file = open(path, 'r')
    qa_list = json.load(file)
    print("questions size: " ,len(qa_list))
    promptList = []
    complete_data = []
    for qa in qa_list:
        
        prompt = f"""{question_structure}
            Ground truth: {qa['ground_truth']}
            Model generated answer: {qa['model_generated_answer']}
            Consistent:"""   
        promptList.append(prompt)
        #qa_copy['consistent'] = generated_caption
        
    ans = evalZephyr(promptList)
    print("ans size: " ,len(ans))
    i=0
    for qa in qa_list:
        qa_copy = qa.copy()
        qa_copy['consistent'] = ans[i]
        i+=1
        complete_data.append(qa_copy)
    return complete_data

def evalZephyr(promptList):
    evals = []
    
    batch_size = 64
    batches = [promptList[i:i+batch_size] for i in range(0, len(promptList), batch_size)]
    for batch in tqdm(batches):
        model_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        
    
        generated_ids = model.generate(
        # seed = 42,
        **model_inputs, 
        max_new_tokens=3,
        do_sample = False,
        min_length = None,
        use_cache = True,
        top_p = 1.0,
        temperature = 1e-05,
        top_k = 50,
        repetition_penalty = 1.0,
        length_penalty = 1,
        max_padding_length = None)
    
        evals += tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # torch.cuda.empty_cache() 
    #print(len(evals))
    
    # Extract the text after "Consistent" for each string in evals
    consistent_texts = [text[text.find("Consistent: ") + len("Consistent: "):].strip() for text in evals]

    return consistent_texts

QA_PATH = '/scratch/nmachav1/MLLM_Hallucinations_CLEVR/outputs/language_augmentation/InstructBLIP'
SAVE_FOLDER = '/scratch/nmachav1/MLLM_Hallucinations_CLEVR/outputs/language_augmentation/Zephyr_Results'
all_files = []
for root, dirs, files in os.walk(QA_PATH):
    # Exclude files from the ".ipynb_checkpoints" folder
    dirs[:] = [d for d in dirs if not d.endswith(".ipynb_checkpoints")]

    for file in files:
        # Check if the file has a .json extension
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            # Process the file or do whatever you need with it
            #print(f"Processing file: {file_path}")
            all_files.append(file_path)
print(all_files)
for file in all_files:
    print(file)
    evals = performEvaluation(file)

    with open(os.path.join(SAVE_FOLDER, file.split("/")[-2]+"/"+file.split("/")[-1]), 'w') as file:
        json.dump(evals, file)