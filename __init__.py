from .prompt_expansion import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .prompt_expansion import fooocus_expansion_path
from .model_loader import load_file_from_url


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


def download_models():
    url = 'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin'
    model_dir = fooocus_expansion_path
    file_name = 'pytorch_model.bin'

    load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)


download_models()
