import json
import os
import numpy as np
from collections import defaultdict

# Load predictions for v1 images
def load_predictions(version):
    with open(f"reports/predictions/{version}_predictions.json") as f:
        return json.load(f)

# Identify hard-to-learn images
def find_hard_images():
    versions = ["v1", "v1+v2", "v1+v2+v3"]
    predictions = {version: load_predictions(version) for version in versions}
    
    # Find images misclassified by all models
    hard_indices = set(range(len(predictions["v1"]["true_labels"])))
    for version in versions:
        preds = np.array(predictions[version]["predictions"])
        true_labels = np.array(predictions[version]["true_labels"])
        hard_indices.intersection_update(np.where(preds != true_labels)[0])
    
    # Class distribution of hard images
    class_dist = defaultdict(int)
    true_labels = predictions["v1"]["true_labels"]
    for idx in hard_indices:
        class_dist[true_labels[idx]] += 1
    
    # Misclassification table
    misclass_table = defaultdict(lambda: defaultdict(int))
    for version in versions:
        preds = predictions[version]["predictions"]
        for idx in hard_indices:
            misclass_table[true_labels[idx]][preds[idx]] += 1
    
    # Save results
    os.makedirs("reports/hard_images", exist_ok=True)
    with open("reports/hard_images/hard_images_report.json", "w") as f:
        json.dump({
            "hard_indices": list(hard_indices),
            "class_distribution": dict(class_dist),
            "misclassification_table": {str(k): dict(v) for k, v in misclass_table.items()}
        }, f, indent=2)

if __name__ == "__main__":
    find_hard_images()