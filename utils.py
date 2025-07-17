import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

def load_model():
    model_id = "CohereLabs/aya-vision-8b"
    processor = AutoProcessor.from_pretrained(model_id, cache_dir = '/h/mikl123/cam/From-Redundancy-to-Relevance')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir = '/h/mikl123/cam/From-Redundancy-to-Relevance'
    )
    return model, processor