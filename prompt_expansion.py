import os
import random
import sys
import torch

# Get the parent directory of 'comfy' and add it to the Python path
comfy_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(comfy_parent_dir)

# Suppress console output
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Import the required modules
import comfy.model_management as model_management
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from comfy.model_patcher import ModelPatcher
from .util import join_prompts, remove_empty_str

# Restore the original stdout
sys.stdout = original_stdout

fooocus_expansion_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                      'fooocus_expansion'))
fooocus_magic_split = [
    ', extremely',
    ', intricate,',
]
dangrous_patterns = '[]【】()（）|:：'


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, '')
    return x


class FooocusExpansion:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_path)
        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_path)
        self.model.eval()

        load_device = model_management.text_encoder_device()

        if 'mps' in load_device.type:
            load_device = torch.device('cpu')

        if 'cpu' not in load_device.type and model_management.should_use_fp16():
            self.model.half()

        offload_device = model_management.text_encoder_offload_device()
        self.patcher = ModelPatcher(self.model, load_device=load_device, offload_device=offload_device)

        # print(f'Fooocus Expansion engine loaded for {load_device}.')

    def __call__(self, prompt, seed):
        model_management.load_model_gpu(self.patcher)
        seed = int(seed)
        set_seed(seed)
        origin = safe_str(prompt)
        prompt = origin + fooocus_magic_split[seed % len(fooocus_magic_split)]

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.patcher.load_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.patcher.load_device)

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(**tokenized_kwargs,
                                       num_beams=1,
                                       max_new_tokens=256,
                                       do_sample=True)

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = response[0][len(origin):]
        result = safe_str(result)
        result = remove_pattern(result, dangrous_patterns)
        return result


class PromptExpansion:
    # Define the expected input types for the node
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "log_prompt": (["No", "Yes"], {"default": "No"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("final_prompt", "seed",)
    FUNCTION = "expand_prompt"  # Function name

    CATEGORY = "utils"  # Category for organization

    @staticmethod
    @torch.no_grad()
    def expand_prompt(text, seed, log_prompt):
        expansion = FooocusExpansion()

        prompt = remove_empty_str([safe_str(text)], default='')[0]

        max_seed = int(1024 * 1024 * 1024)
        if not isinstance(seed, int):
            seed = random.randint(1, max_seed)
        if seed < 0:
            seed = - seed
        seed = seed % max_seed

        expansion_text = expansion(prompt, seed)
        final_prompt = join_prompts(prompt, expansion_text)

        if log_prompt == "Yes":
            print(f"[Prompt Expansion] New suffix: {expansion_text}")
            print(f"Final prompt: {final_prompt}")

        return final_prompt, seed


# Define a mapping of node class names to their respective classes
NODE_CLASS_MAPPINGS = {
    "PromptExpansion": PromptExpansion
}

# A dictionary that contains human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptExpansion": "Prompt Expansion"
}
