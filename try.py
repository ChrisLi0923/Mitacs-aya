# aya_vision_refexp.py
import random
import csv
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import torch

# Prompts
DEFAULT_PROMPTS = [
    "Describe the object in the red box in a way that allows another person to distinguish it from all other objects in the image.",
    "Give a clear and specific description of the object in the red box so that another user can find it without hesitation.",
    "Describe the object in the red box, so that another user can identify the object from other objects.",
    "Describe the object in the red box in a way that allows another person to find that one specific object.",
    "Provide a description of the object in the red box so that another person can recognize and identify that unique item.",
    "Describe the object in the red box so that another person can pinpoint that exact object among others.",
    "Explain the features of the object in the red box that make it stand out, so another person can find it.",
    "Describe the object inside the red box so that someone else can locate that particular item with certainty.",
    "Describe the object in the red box so that another person can find that one specific object.",
    "Give a description of the object in the red box so that another user can identify the exact unique object."
]

BRIEF_PROMPTS = [
    "Describe the red-boxed object using the fewest words while ensuring it can be uniquely identified.",
    "Give a minimal description that allows someone to find the exact object in the red box.",
    "Use the least words necessary to ensure the red-boxed object is unmistakably identifiable.",
    "Provide a short yet precise description so the red-boxed object can be uniquely located.",
    "Describe the object in the red box concisely, ensuring it is the only possible match.",
    "Identify the red-boxed object using the fewest words while making it uniquely findable.",
    "Give a brief but unambiguous description that guarantees the red-boxed object can be found.",
    "Provide the shortest possible description that still allows precise identification of the red-boxed object.",
    "Describe the red-boxed object in minimal words while ensuring no confusion with other objects.",
    "Use as few words as possible to describe the red-boxed object in a way that guarantees unique identification."
]

# Load Aya Vision model
model_id = "CohereLabs/aya-vision-8b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_refexp(image_url: str, prompt_mode: str = "default"):
    prompts = DEFAULT_PROMPTS if prompt_mode == "default" else BRIEF_PROMPTS
    chosen_prompt = random.choice(prompts)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "url": image_url},
            {"type": "text", "text": chosen_prompt}
        ]}
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        padding=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    gen_tokens = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3,
    )

    output_text = processor.tokenizer.decode(
        gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return chosen_prompt, output_text

# Load RefOI dataset
split = "single_presence"  # or "co_occurrence"
dataset = load_dataset("Seed42Lab/RefOI", split=split)

# Output CSV
csv_filename = f"refexp_outputs_{split}.csv"
with open(csv_filename, mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Image URL", "Prompt Mode", "Prompt", "Generated Description", "Label Name", "Is COCO", "Co-Occurrence"])

    for example in dataset:
        image_url = example["boxed_image"]
        label_name = example["label_name"]
        is_coco = example["is_coco"]
        co_occur = example["co_occurrence"]

        for mode in ["default", "brief"]:
            prompt, description = generate_refexp(image_url, mode)
            writer.writerow([image_url, mode, prompt, description, label_name, is_coco, co_occur])
            print(f"[{mode.upper()}] {prompt}\n→ {description}\n")

