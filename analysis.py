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
    metrics['BLEU-1'] = sentence_bleu(references, generated, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    metrics['BLEU-4'] = sentence_bleu(references, generated, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(generated, references[0])
    metrics['ROUGE-1'] = rouge_scores['rouge1'].fmeasure
    metrics['ROUGE-L'] = rouge_scores['rougeL'].fmeasure

    # METEOR
    metrics['METEOR'] = meteor_score(references, generated)

    return metrics

def generate_wordcloud(texts, title,save_path):
    blob = ' '.join(texts)
    wc = WordCloud(width=1000, height=500, background_color="white").generate(blob)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)#plt.show()
    plt.close()

# Main eval
results = {}

for dataset_path in tqdm(dataset_variants):
    print(f"Evaluating: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    all_metrics = defaultdict(list)
    gen_texts, ref_texts = [], []

    for ex in ds:
        generated = ex["generated_expression"]
        references = ex["written_descriptions"]
        gen_texts.append(generated)
        ref_texts.extend(references)

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
    subset = ds.select(range(min(100, len(ds))))
    for ex in subset:
        inputs = clip_processor(text=ex["generated_expression"], images=ex["image"], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = clip_model(**inputs)
            clip_scores.append(out.logits_per_image[0][0].item())
    all_metrics["CLIPScore"] = clip_scores

    # Average
    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    results[dataset_path.name] = avg_metrics

    # Wordcloud for generated
    gen_wc_path = os.path.join(wordcloud_dir, f"{dataset_path.name}_generated_wordcloud.png")
    generate_wordcloud(gen_texts, f"Generated Expressions: {dataset_path.name}", gen_wc_path)

# Word cloud for human-written
human_wc_path = os.path.join(wordcloud_dir, "human_written_wordcloud.png")
generate_wordcloud(ref_texts, "Human-Written Descriptions (All Datasets)", human_wc_path)

# Print results
import json
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
