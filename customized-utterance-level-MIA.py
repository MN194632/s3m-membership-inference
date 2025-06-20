# Modified customized-utterance-level-MIA.py
import argparse
import os
import random
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from dataset.dataset import *
from utils.utils import *
from model.customized_similarity_model import UtteranceLevelModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_with_fixed_threshold_seen_unseen(model, seen_dataloader, unseen_dataloader, threshold, device):
    """
    Evaluate model using a pre-determined threshold on seen and unseen data separately.
    This handles the fact that our datasets don't include labels.
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        # Process seen data (members = label 1)
        print("Processing seen data (members)...")
        for features, paths in tqdm(seen_dataloader, desc="Evaluating seen"):
            features_tensor = [torch.FloatTensor(f).to(device) for f in features]
            
            # Get raw logits (no sigmoid)
            logits = model(features_tensor)
            
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend([1] * len(logits))  # Seen = members = 1
            all_paths.extend(paths)
        
        # Process unseen data (non-members = label 0)
        print("Processing unseen data (non-members)...")
        for features, paths in tqdm(unseen_dataloader, desc="Evaluating unseen"):
            features_tensor = [torch.FloatTensor(f).to(device) for f in features]
            
            # Get raw logits (no sigmoid)
            logits = model(features_tensor)
            
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend([0] * len(logits))  # Unseen = non-members = 0
            all_paths.extend(paths)
    
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    # Apply fixed threshold to make predictions
    predictions = (all_logits >= threshold).astype(int)
    
    # Calculate metrics
    results = {
        'auc': roc_auc_score(all_labels, all_logits),
        'accuracy': accuracy_score(all_labels, predictions),
        'precision': precision_score(all_labels, predictions, zero_division=0),
        'recall': recall_score(all_labels, predictions, zero_division=0),
        'logits': all_logits,
        'labels': all_labels,
        'predictions': predictions,
        'paths': all_paths
    }
    
    return results

def main(args):
    random.seed(args.seed)

    # Load model info including the optimal threshold
    model_info_path = os.path.join(
        args.output_path, 
        f"customized-utterance-model-info-{args.model}.json"
    )
    
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(
            f"Model info file not found: {model_info_path}\n"
            f"Please run train-utterance-level-similarity-model.py first!"
        )
    
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    optimal_threshold = model_info['optimal_threshold']
    training_auc = model_info['training_auc']
    input_dim = model_info['input_dim']
    
    print(f"Loaded model info:")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Training AUC: {training_auc:.4f}")
    print(f"  Input dimension: {input_dim}")

    # Load the target datasets (these are your actual evaluation targets)
    seen_splits = ["train-clean-100"]  # True members
    unseen_splits = ["test-clean", "test-other"]  # True non-members

    # Create datasets from actual target splits
    seen_dataset = CustomizedUtteranceLevelDataset(
        args.seen_base_path, seen_splits, args.model
    )
    unseen_dataset = CustomizedUtteranceLevelDataset(
        args.unseen_base_path, unseen_splits, args.model
    )

    # Create data loaders
    seen_dataloader = DataLoader(
        seen_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=args.num_workers,
        collate_fn=seen_dataset.collate_fn,
    )
    unseen_dataloader = DataLoader(
        unseen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=unseen_dataset.collate_fn,
    )

    # Load the trained model
    model_path = model_info['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = UtteranceLevelModel(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\nEvaluating on target data using fixed threshold: {optimal_threshold:.4f}")
    print("="*60)

    # Evaluate on both seen and unseen data using fixed threshold
    results = evaluate_with_fixed_threshold_seen_unseen(
        model, seen_dataloader, unseen_dataloader, optimal_threshold, device
    )

    # Calculate TPR and FPR
    all_logits = results['logits']
    all_labels = results['labels']
    all_predictions = results['predictions']
    
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Combine with model info for final results
    final_results = {
        'auc': float(results['auc']),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'threshold_used': float(optimal_threshold),
        'training_auc': float(training_auc),
        'seen_samples': int(np.sum(all_labels == 1)),
        'unseen_samples': int(np.sum(all_labels == 0)),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    # Print results
    print("\nFINAL RESULTS ON TARGET DATA:")
    print("="*40)
    print(f"AUC: {final_results['auc']:.4f}")
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall: {final_results['recall']:.4f}")
    print(f"TPR (True Positive Rate): {final_results['tpr']:.4f}")
    print(f"FPR (False Positive Rate): {final_results['fpr']:.4f}")
    print(f"Threshold used: {final_results['threshold_used']:.4f}")
    print(f"Seen samples: {final_results['seen_samples']:,}")
    print(f"Unseen samples: {final_results['unseen_samples']:,}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Save detailed results
    results_path = os.path.join(
        args.output_path,
        f"{args.model}-final-evaluation-results.json"
    )
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")

    # Create and save ROC curve
    create_roc_curve(all_labels, all_logits, args.model, args.output_path)

def create_roc_curve(labels, scores, model_name, output_path):
    """Create and save ROC curve visualization"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} (Fixed Threshold)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(output_path, f"{model_name}-final-roc-curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {roc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seen_base_path", default="/work3/s194632/LibriSpeech_features",
        help="directory of feature of the seen dataset (default LibriSpeech-100)",
    )
    parser.add_argument(
        "--unseen_base_path", default="/work3/s194632/LibriSpeech_features",
        help="directory of feature of the unseen dataset (default LibriSpeech-[dev/test])",
    )
    parser.add_argument("--output_path", help="directory where model and results are saved")
    parser.add_argument(
        "--model", help="which self-supervised model you used to extract features"
    )
    parser.add_argument("--seed", type=int, default=57, help="random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    args = parser.parse_args()

    main(args)