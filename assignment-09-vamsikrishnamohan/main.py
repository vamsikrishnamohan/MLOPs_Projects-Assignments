# __define-ocg__: Required Libraries
import ray
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# # __define-ocg__: Dataset Parser
# def parse_reviews(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         raw_data = file.read()

#     entries = raw_data.strip().split('\n\n')
#     parsed = []

#     for entry in entries:
#         fields = dict()
#         for line in entry.strip().split('\n'):
#             if ':' in line:
#                 key, val = line.split(':', 1)
#                 fields[key.strip()] = val.strip()
#         if 'review/text' in fields and 'review/score' in fields:
#             try:
#                 fields['review/score'] = float(fields['review/score'])
#                 parsed.append(fields)
#             except ValueError:
#                 continue
#     return parsed

# # __define-ocg__: Ray-based remote function
# @ray.remote
# def map_sentiment(entry):
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification
#     import torch
#     import torch.nn.functional as F

#     model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)

#     text = entry['review/text']

#     # Use tokenizer's truncation properly (cut to model's max length)
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probs = F.softmax(logits, dim=1)
#         pred_idx = torch.argmax(probs, dim=1).item()
#         predicted_label = model.config.id2label[pred_idx]

#     actual_label = 'POSITIVE' if entry['review/score'] >= 3.0 else 'NEGATIVE'
#     return ('POSITIVE' if predicted_label == 'POSITIVE' else 'NEGATIVE', actual_label)


# # __define-ocg__: Metric Computation
# def reduce_metrics(pairs):
#     predicted = [p for p, _ in pairs]
#     actual = [a for _, a in pairs]

#     cm = confusion_matrix(actual, predicted, labels=["POSITIVE", "NEGATIVE"])
#     precision = precision_score(actual, predicted, pos_label='POSITIVE')
#     recall = recall_score(actual, predicted, pos_label='POSITIVE')

#     return cm, precision, recall

# def plot_confusion(cm):
#     labels = ["POSITIVE", "NEGATIVE"]
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.show()

# # __define-ocg__: Main function
# def main():
#     ray.init()  # start Ray

#     # Step 1: Load data
#     data = parse_reviews('Gourmet_Foods.txt')

#     # Step 2: Run Ray jobs
#     futures = [map_sentiment.remote(entry) for entry in data]
#     mapped_pairs = ray.get(futures)

#     # Step 3: Evaluation
#     cm, precision, recall = reduce_metrics(mapped_pairs)

#     print("Confusion Matrix:\n", cm)
#     print(f"\nPrecision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")

#     plot_confusion(cm)
#     ray.shutdown()

# if __name__ == "__main__":
#     main()
