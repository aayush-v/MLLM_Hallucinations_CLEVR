import json 
import argparse
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args, tokenizer, model, image_processor, context_len, model_name):
    # Model
    disable_torch_init()

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


image_path = '/scratch/nmachav1/CLEVR_v1.0/images/val/'
model_path = "liuhaotian/llava-v1.5-13b"
model_name=get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)
object_type = ['inter', 'intra']
#for i in range(3,11):
for i in object_type:
    #dataset = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/dataset/object_number_split/val_num_{i}.json'
    #output_path = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/outputs/LLaVA/val_num_{i}.json'
    dataset = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/dataset/object_type_split/val_type_{i}.json'
    output_path = f'/scratch/nmachav1/MLLM_Hallucinations_CLEVR/outputs/LLaVA/val_type_{i}.json'
    with open(dataset, 'r') as f:
        data = json.load(f)


    complete_data = []
    for each in tqdm(data):
        each_copy = each.copy()
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": model_name,
            "query": each['question'],
            "conv_mode": None,
            "image_file": image_path+each['image_filename'],
            "sep": ",",
        })()
        generated_caption = eval_model(args, tokenizer, model, image_processor, context_len, model_name)
        each_copy['model_generated_answer'] = generated_caption
        #print(each_copy)
        complete_data.append(each_copy)


    with open(output_path, 'w') as f:
        json.dump(complete_data, f ,indent=4)
