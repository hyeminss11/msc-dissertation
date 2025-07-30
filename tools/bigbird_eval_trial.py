import json
from transformers import BigBirdTokenizer, AutoModelForSequenceClassification
import torch

# Load model & tokenizer
model_name = "google/bigbird-roberta-base"
tokenizer = BigBirdTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.eval()

# Load data
with open("experiments/llm_rewriting_trial1/llm_rewriting_trial1.2/output_data/av_eval_samples_cross_author_type.json", encoding="utf-8") as f:
    samples = json.load(f)

results = []

for i, sample in enumerate(samples):
    inputs = tokenizer(
        sample["text1"], sample["text2"],
        padding="max_length", truncation=True, return_tensors="pt", max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()

    # Debugging
    print(f"[Sample {i+1}]")
    print("Logits:", logits.tolist())
    print("Probabilities:", probs.tolist())
    print("Prediction:", pred)

    label = sample["same_author"]
    match = label == pred
    print(f"[Sample {i+1}] Label: {label}, Predicted: {pred} ({'O' if match else 'X'})")

    # save result
    results.append({
        "text1": sample["text1"],
        "text2": sample["text2"],
        "label": label,
        "prediction": pred,
        "correct": match
    })

# Save to file
output_path = "experiments/llm_rewriting_trial1/llm_rewriting_trial1.3/bigbird_eval_results_trial1.2.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {output_path}")