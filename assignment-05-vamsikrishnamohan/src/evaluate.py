import argparse
import yaml
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import json
import os
import matplotlib.pyplot as plt
from dvclive import Live
from train import CNN


def evaluate_model(version):
    try:
        logging.info(f"Starting evaluation process for version: {version}...")
        
        # Load configuration
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
            
        seed = params['seed']
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize DVCLive
        live = Live("dvclive_eval")

        # Load model
        model = CNN().to(device)
        model.load_state_dict(torch.load("models/model.pth"))
        model.eval()

        # Prepare test data for the specified version
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        test_dataset = datasets.ImageFolder(
            root=f"data/processed/{version}/test",  # Use version-specific test data
            transform=transform
        )
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluation
        true_labels = []
        pred_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

        # Convert NumPy arrays to native Python types
        true_labels = [int(label) for label in true_labels]
        pred_labels = [int(label) for label in pred_labels]

        # Save predictions and true labels with version in filename
        os.makedirs("reports/predictions", exist_ok=True)
        with open(f"reports/predictions/{version}_predictions.json", "w") as f:
            json.dump({
                "true_labels": true_labels,
                "predictions": pred_labels
            }, f, indent=2)

        # Calculate metrics
        accuracy = float(np.mean(np.array(true_labels) == np.array(pred_labels)))
        precision = float(precision_score(true_labels, pred_labels, average="macro"))
        recall = float(recall_score(true_labels, pred_labels, average="macro"))
        f1 = float(f1_score(true_labels, pred_labels, average="macro"))

        class_names = test_dataset.classes
        class_acc = {}
        
        for class_idx, class_name in enumerate(class_names):
            mask = np.array(true_labels) == class_idx
            class_acc[class_name] = float(np.mean(np.array(pred_labels)[mask] == class_idx))

        # Log metrics with DVCLive
        live.log_metric(f"{version}_test_accuracy", accuracy)
        live.log_metric(f"{version}_test_precision", precision)
        live.log_metric(f"{version}_test_recall", recall)
        live.log_metric(f"{version}_test_f1", f1)
        live.log_params(params)
        for class_name, acc in class_acc.items():
            live.log_metric(f"{version}_test_accuracy_{class_name}", acc)
        live.next_step()

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=class_names)
        disp.plot()
        
        # Save reports with version in filename
        os.makedirs("reports", exist_ok=True)
        plt.savefig(f"reports/{version}_confusion_matrix.png")
        plt.close()

        # Save metrics with version in filename
        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/{version}_test_accuracy.json", "w") as f:
            json.dump({
                "overall_accuracy": accuracy,
                "class_wise_accuracy": class_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }, f, indent=2)

        logging.info(f"Evaluation for {version} completed. Accuracy: {accuracy:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                    f"F1-Score: {f1:.4f}")

    except Exception as e:
        logging.error(f"Evaluation failed for version {version}: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="Dataset version to evaluate (e.g., v1, v2, v1+v2)")
    args = parser.parse_args()
    evaluate_model(args.version)