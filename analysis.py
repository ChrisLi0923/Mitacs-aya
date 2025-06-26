from datasets import load_from_disk
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from bert_score import score as bert_score
from tqdm import tqdm
import os
import nltk
import json
from torch.nn.functional import normalize
from PIL import Image

nltk.download('wordnet')
nltk.download('omw-1.4')

wordcloud_dir = "wordcloud_outputs"
os.makedirs(wordcloud_dir, exist_ok=True)
# Setup paths
base_path = Path("/scratch/ssd004/scratch/haigelee/Mitacs/genData")
dataset_variants = list(base_path.glob("Aya*/RefOI_with_generated_*"))

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)

# Evaluation helpers
def evaluate_metrics(generated, references):
    metrics = defaultdict(float)
    smoothie = SmoothingFunction().method1

    # BLEU
    metrics['BLEU-1'] = sentence_bleu(
        [ref.split() for ref in references], generated.split(),
        weights=(1, 0, 0, 0), smoothing_function=smoothie)
    metrics['BLEU-4'] = sentence_bleu(
        [ref.split() for ref in references], generated.split(),
        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(generated, references[0])
    metrics['ROUGE-1'] = rouge_scores['rouge1'].fmeasure
    metrics['ROUGE-L'] = rouge_scores['rougeL'].fmeasure

    # METEOR
    metrics['METEOR'] = meteor_score([ref.split() for ref in references], generated.split())

    return metrics

def generate_wordcloud(texts, title, save_path):
    blob = ' '.join(texts)
    wc = WordCloud(width=1000, height=500, background_color="white").generate(blob)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# To collect human-written refs per model for later word clouds
model_ref_texts = defaultdict(list)

results = {}

for dataset_path in tqdm(dataset_variants):
    print(f"Evaluating: {dataset_path}")
    ds = load_from_disk(str(dataset_path))

    # Access the first example
    first = ds[0]
    b_i = first["boxed_image"]

    # ✅ Sanity check
    print("Type of boxed_image:", type(b_i))
    print("Is boxed_image a PIL image?", isinstance(b_i, Image.Image))
    print("Boxed image size:", b_i.size)
    print("Boxed image mode:", b_i.mode)





    all_metrics = defaultdict(list)
    gen_texts, ref_texts = [], []

    # Extract model name like 'Aya32b' or 'Aya8b' from path
    model_name = dataset_path.parts[-2]

    for ex in ds:
        generated = ex["generated_expression"]
        references = ex["written_descriptions"]
        gen_texts.append(generated)
        ref_texts.extend(references)
        model_ref_texts[model_name].extend(references)

        m = evaluate_metrics(generated, references)
        for k, v in m.items():
            all_metrics[k].append(v)

    # BERTScore (use only first written description for reference)
    preds = [ex["generated_expression"] for ex in ds]
    refs = [ex["written_descriptions"][0] for ex in ds]
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=True)
    all_metrics["BERTScore"] = F1.tolist()

    # CLIPScore (approx): limit for speed
    clip_scores = []
    subset = ds
    for ex in subset:
        inputs = clip_processor(
            text=ex["generated_expression"],
            images=ex["boxed_image"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        with torch.no_grad():
            # Get embeddings
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            # Normalize
            image_features = normalize(image_features, dim=-1)
            text_features = normalize(text_features, dim=-1)

            # Cosine similarity
            similarity = (image_features @ text_features.T).squeeze().item()
            clip_scores.append(similarity * 100)  # As per CLIPScore definition
    all_metrics["CLIPScore"] = clip_scores

    # Average metrics
    avg_metrics = {
        k: float(np.mean(v)) * (1 if k == "CLIPScore" else 100)
        for k, v in all_metrics.items()
    }

    results[f"{model_name}/{dataset_path.name}"] = avg_metrics

    # Make subfolder for each model
    model_wc_dir = os.path.join(wordcloud_dir, model_name)
    os.makedirs(model_wc_dir, exist_ok=True)

    # Save generated word cloud
    gen_wc_path = os.path.join(model_wc_dir, f"{dataset_path.name}_generated_wordcloud.png")
    generate_wordcloud(gen_texts, f"{model_name} - {dataset_path.name}", gen_wc_path)

# Word cloud for human-written descriptions per model
for model_name, texts in model_ref_texts.items():
    model_wc_dir = os.path.join(wordcloud_dir, model_name)
    os.makedirs(model_wc_dir, exist_ok=True)
    human_wc_path = os.path.join(model_wc_dir, f"{model_name}_human_written_wordcloud.png")
    generate_wordcloud(texts, f"Human-Written Descriptions: {model_name}", human_wc_path)

# Save results to JSON
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
