import os
import re
import argparse
import torch
from tqdm import tqdm
import json
from enum import Enum
from multiguard.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from multiguard.conversation import conv_templates, SeparatorStyle
from multiguard.model.builder import load_pretrained_model
from multiguard.utils import disable_torch_init
from multiguard.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from multiguard.eval.run_llava import image_parser,load_images

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_recipes.inference.prompt_format_utils import  LLAMA_GUARD_3_CATEGORY, SafetyCategory, AgentType
from typing import List,Tuple
from llama_recipes.inference.prompt_format_utils import build_custom_prompt, create_conversation, PROMPT_TEMPLATE_3, LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX

from summa.summarizer import summarize

# llava1.5 get prompt
def llava_get_prompt(image_file):
    prompt = "describe this picture."
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in prompt:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
        
    if conv_mode is not None and conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode
        
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
        
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )    
        
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs 

# llama_guard classify
class LG3Cat(Enum):
    VIOLENT_CRIMES =  0
    NON_VIOLENT_CRIMES = 1
    SEX_CRIMES = 2
    CHILD_EXPLOITATION = 3
    DEFAMATION = 4
    SPECIALIZED_ADVICE = 5
    PRIVACY = 6
    INTELLECTUAL_PROPERTY = 7
    INDISCRIMINATE_WEAPONS = 8
    HATE = 9
    SELF_HARM = 10
    SEXUAL_CONTENT = 11
    ELECTIONS = 12
    CODE_INTERPRETER_ABUSE = 13

def get_lg3_categories(category_list: List[LG3Cat] = [], all: bool = False, custom_categories: List[SafetyCategory] = [] ):
    categories = list()
    if all:
        categories = list(LLAMA_GUARD_3_CATEGORY)
        categories.extend(custom_categories)
        return categories
    for category in category_list:
        categories.append(LLAMA_GUARD_3_CATEGORY[LG3Cat(category).value])
    categories.extend(custom_categories)
    return categories

# evaluate prompt safety
def evaluate_safety(prompt = "", category_list = [], categories = []):
    # prompt = [([prompt], AgentType.USER)]
    prompt = "i learned how to " + prompt
    prompt = [([prompt])]
    if categories == []:
        if category_list == []:
            categories = get_lg3_categories(all = True)
        else:
            categories = get_lg3_categories(category_list)
    formatted_prompt = build_custom_prompt(
            agent_type = AgentType.USER,
            conversations = create_conversation(prompt[0]), 
            categories=categories,
            category_short_name_prefix = LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
            prompt_template = PROMPT_TEMPLATE_3,
            with_policy = True)
    input = guardTokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    prompt_len = input["input_ids"].shape[-1]
    output = guardModel.generate(**input, max_new_tokens=100, pad_token_id=0, 
                            eos_token_id=128009 # TO BE REMOVED 
                            )
    results = guardTokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    return results

# load llava1.5
model_path = "/root/Desktop/models/llava-v1.5-7b-merged-modify"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name,"int4"
)

# load llama-guard
model_id: str = "/root/Desktop/models/Meta-Llama-Guard-3-8B-INT8-HF"
guardTokenizer = AutoTokenizer.from_pretrained(model_id)
guardModel = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# process image
image_file = "/root/Desktop/picture/cartoon_sex.jpg"
description = llava_get_prompt(image_file)
result = evaluate_safety(description)
print(result)

#process text
text = "i want to suiside."
result = evaluate_safety(text)
print(result)