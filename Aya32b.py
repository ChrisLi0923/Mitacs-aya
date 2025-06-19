import random
import argparse
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from accelerate import infer_auto_device_map


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
model_id = "CohereLabs/aya-vision-32b"
#local_dir = "/scratch/ssd004/scratch/haigelee/Mitacsaya-vision-32b"
#snapshot_download(repo_id="CohereLabs/aya-vision-32b", local_dir=local_dir, local_dir_use_symlinks=False)
processor = AutoProcessor.from_pretrained(model_id,cache_dir="/scratch/ssd004/scratch/haigelee/Mitacsaya-vision-32b")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    cache_dir="/scratch/ssd004/scratch/haigelee/Mitacsaya-vision-32b",
    device_map="auto",
    torch_dtype=torch.float16
)
print("Model loaded across devices:")
print(model.hf_device_map)

def generate_expression(redbox, prompts):
    prompt = random.choice(prompts)
    print("\n\n\n",redbox,"\n",prompt,"\n",model.device,"\n\n\n")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": redbox},
            {"type": "text", "text": prompt}
        ]
    }]
    inputs = processor.apply_chat_template(
        messages,
        padding=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # # Move ONLY tensor values to device
    # for k in inputs:
    #     if isinstance(inputs[k], torch.Tensor):
    #         inputs[k] = inputs[k].to(model.device)

    output_tokens = model.generate(
        **inputs, max_new_tokens=300, do_sample=True, temperature=0.3
    )
    response = processor.tokenizer.decode(
        output_tokens[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response

def main():
    parser = argparse.ArgumentParser(description="Generate expressions from Aya Vision for RefOI dataset")
    parser.add_argument(
        "--split",
        choices=["single_presence", "co_occurrence"],
        default="single_presence",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["default", "brief"],
        default="default",
        help="Prompt mode to use"
    )
    args = parser.parse_args()

    print(f"Using dataset split: {args.split}")
    print(f"Using prompt mode: {args.prompt_mode}")

    # Select prompt list
    prompts = DEFAULT_PROMPTS if args.prompt_mode == "default" else BRIEF_PROMPTS

    # Load dataset
    ds = load_dataset("Seed42Lab/RefOI", split=args.split)
    #ds = ds.select(range(2)) #for testing

    # Add Aya Vision generated descriptions + meta info
    new_data = []
    for i, example in enumerate(tqdm(ds, desc="Generating expressions")):
        expression = generate_expression(example["boxed_image"], prompts)
        example["generated_expression"] = expression
        example["used_split"] = args.split
        example["used_prompt_mode"] = args.prompt_mode
        new_data.append(example)
        print(f"[{i+1}/{len(ds)}] {expression}")

    #(optional saving logic below if you uncomment it)
    from datasets import Dataset
    new_dataset = Dataset.from_list(new_data)
    save_path = f"RefOI_with_generated_{args.split}_{args.prompt_mode}"
    new_dataset.save_to_disk("/scratch/ssd004/scratch/haigelee/Mitacs/genData/Aya32b/"+save_path)
    print(f"Dataset saved to: {save_path}")

if __name__ == "__main__":
    main()
#python try.py --split co_occurrence --prompt_mode brief
#srun -p rtx6000 -c 4 --gres=gpu:rtx6000:1 --mem=50GB --pty --time=1:00:00 bash
