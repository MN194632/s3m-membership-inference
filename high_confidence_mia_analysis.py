import argparse
import os
import numpy as np
import pandas as pd
import torch
import json
import math
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import *
from utils.utils import *
from model.customized_similarity_model import UtteranceLevelModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MemorisationAnalyser:
    def __init__(self, base_path, model_name, model_info_path):
        self.base_path = base_path
        self.model_name = model_name
        self.model_info_path = model_info_path
        
        # Load model info including the pre-determined threshold
        self.load_model_info()
        
        seen_splits = ["train-clean-100"]
        unseen_splits = ["test-clean","test-other"]
        self.seen_dataset = CustomizedUtteranceLevelDataset(base_path, seen_splits, model_name)
        self.unseen_dataset = CustomizedUtteranceLevelDataset(base_path, unseen_splits, model_name)

        # Results storage
        self.seen_results = []
        self.unseen_results = []
        
    def load_model_info(self):
        """Load the model info including the optimal threshold from training"""
        if not os.path.exists(self.model_info_path):
            raise FileNotFoundError(
                f"Model info file not found: {self.model_info_path}\n"
                f"Please run train-utterance-level-similarity-model.py first!"
            )
        
        with open(self.model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.optimal_threshold = self.model_info['optimal_threshold']
        self.similarity_model_path = self.model_info['model_path']
        self.input_dim = self.model_info['input_dim']
        
        print(f"Loaded model info:")
        print(f"  Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"  Model path: {self.similarity_model_path}")
        print(f"  Input dimension: {self.input_dim}")
        
    def extract_high_confidence_samples(self, confidence_threshold=0.9):
        """Extract samples where MIA confidently predicts they were seen (memorisation candidates)
        
        Analyses only seen dataset samples:
        - Predicts each sample as "seen" or "unseen" using threshold
        - Calculates confidence for each prediction type
        - Returns high-confidence memorisation candidates
        """
        
        print("Calculating MIA scores/logits for similarity for seen samples...")
        seen_scores, seen_features, seen_paths = self._calculate_mia_scores(self.seen_dataset, "seen")
        
        print("Calculating MIA scores/logits for similarity for unseen samples...")
        unseen_scores, unseen_features, unseen_paths = self._calculate_mia_scores(self.unseen_dataset, "unseen")
        
        # FIXED: Use the pre-determined optimal threshold instead of calculating from test data
        threshold = self.optimal_threshold
        print(f"Using pre-determined optimal threshold: {threshold:.4f}")
        
        seen_scores = np.array(seen_scores)
        unseen_scores = np.array(unseen_scores)
        
        # Classify each seen sample based on MIA prediction using fixed threshold
        predicted_seen_mask = seen_scores > threshold
        predicted_unseen_mask = seen_scores <= threshold
        
        # Calculate confidence for each prediction type
        confidences = np.zeros_like(seen_scores)
    
        if np.any(predicted_seen_mask):
            # For samples predicted as "seen": confidence = normalised distance above threshold
            seen_scores_subset = seen_scores[predicted_seen_mask]
            max_seen_score = np.max(seen_scores_subset)
            confidences[predicted_seen_mask] = (seen_scores_subset - threshold) / (max_seen_score - threshold + 1e-8)
        
        if np.any(predicted_unseen_mask):
            # For samples predicted as "unseen": confidence = normalised distance below threshold  
            unseen_scores_subset = seen_scores[predicted_unseen_mask]
            min_unseen_score = np.min(unseen_scores_subset)
            confidences[predicted_unseen_mask] = (threshold - unseen_scores_subset) / (threshold - min_unseen_score + 1e-8)
        
        # Find high-confidence memorisation candidates (predicted seen + high confidence)
        memorised_candidates_idx = np.where(
            predicted_seen_mask & (confidences >= confidence_threshold)
        )[0]
        
        # Find high-confidence generalisation samples (predicted unseen + high confidence)
        generalised_samples_idx = np.where(
            predicted_unseen_mask & (confidences >= confidence_threshold)
        )[0]
        
        print(f"Total seen samples: {len(seen_scores)}")
        print(f"Predicted as seen: {np.sum(predicted_seen_mask)} ({np.sum(predicted_seen_mask)/len(seen_scores)*100:.1f}%)")
        print(f"Predicted as unseen: {np.sum(predicted_unseen_mask)} ({np.sum(predicted_unseen_mask)/len(seen_scores)*100:.1f}%)")
        print(f"High-confidence memorisation candidates: {len(memorised_candidates_idx)} ({len(memorised_candidates_idx)/len(seen_scores)*100:.1f}%)")
        print(f"High-confidence generalisation samples: {len(generalised_samples_idx)} ({len(generalised_samples_idx)/len(seen_scores)*100:.1f}%)")
        
        return {
            'seen': {  # High-confidence memorisation candidates
                'indices': memorised_candidates_idx,
                'scores': seen_scores[memorised_candidates_idx],
                'features': [seen_features[i] for i in memorised_candidates_idx],
                'paths': [seen_paths[i] for i in memorised_candidates_idx],
                'confidences': confidences[memorised_candidates_idx]
            },
            'unseen': {  # High-confidence generalisation samples  
                'indices': generalised_samples_idx,
                'scores': seen_scores[generalised_samples_idx],
                'features': [seen_features[i] for i in generalised_samples_idx],
                'paths': [seen_paths[i] for i in generalised_samples_idx],
                'confidences': confidences[generalised_samples_idx]
            },
            'threshold': threshold,
            'all_predictions': {
                'scores': seen_scores,
                'confidences': confidences,
                'predicted_seen': predicted_seen_mask,
                'predicted_unseen': predicted_unseen_mask
            }
        }
    
    def _calculate_mia_scores(self, dataset, label):
        """Calculate MIA scores using improved attack model"""
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                              collate_fn=dataset.collate_fn)
        
        scores = []
        features = []
        paths = []
        
        # Use improved attack model (similarity model)
        sim_predictor = self._load_similarity_model()
        
        for _, (utterance_features, utterance_paths) in enumerate(tqdm(dataloader, desc=f"Processing {label}")):
            for feature, path in zip(utterance_features, utterance_paths):
                features_list = [torch.FloatTensor(feature).to(device)]
                with torch.no_grad():
                    logit = sim_predictor(features_list)[0]
                    score = logit.cpu().item()  
                
                scores.append(score)
                features.append(feature)
                paths.append(path)

        return scores, features, paths
    
    def _load_similarity_model(self):
        """Load the trained similarity model"""
        ckpt = torch.load(self.similarity_model_path, map_location=device)
        sim_predictor = UtteranceLevelModel(self.input_dim).to(device)
        sim_predictor.load_state_dict(ckpt)
        sim_predictor.eval()
        return sim_predictor
    
    def analyse_rarity_patterns(self, high_conf_data):
        """Analyse what makes high-confidence memorisation candidates rare/atypical"""
        
        # 1. Embedding-based rarity analysis for memorisation candidates only
        rarity_metrics = self._calculate_rarity_metrics(high_conf_data)
        
        # 2. Statistical analysis
        results = {
            'memorisation_rarity_scores': rarity_metrics['seen'],
            'generalisation_rarity_scores': rarity_metrics['unseen'],  # High-conf predicted as unseen
            'analysis': self._statistical_analysis(rarity_metrics)
        }
        
        return results
    
    def _calculate_rarity_metrics(self, high_conf_data):
        """Calculate various rarity metrics for high-confidence samples"""
        
        print("Calculating rarity metrics...")
        
        # Collect all seen features for reference
        all_seen_features = []
        seen_dataloader = DataLoader(self.seen_dataset, batch_size=32, shuffle=False,
                                   collate_fn=self.seen_dataset.collate_fn)
        
        for batch_id, (utterance_features, _) in enumerate(tqdm(seen_dataloader, desc="Loading reference features")):
            for feature in utterance_features:
                # Convert tensor to numpy and average pool to utterance level
                if isinstance(feature, torch.Tensor):
                    feature_np = feature.numpy()
                else:
                    feature_np = np.array(feature)
                all_seen_features.append(np.mean(feature_np, axis=0))
        
        all_seen_features = np.array(all_seen_features)
        
        # Calculate rarity for high-confidence samples
        metrics = {}
        for data_type in ['seen', 'unseen']:
            features = high_conf_data[data_type]['features']
            if len(features) == 0:
                metrics[data_type] = {}
                continue
                
            # Average pool features to utterance level
            utterance_features = []
            for f in features:
                if isinstance(f, torch.Tensor):
                    f_np = f.numpy()
                else:
                    f_np = np.array(f)
                utterance_features.append(np.mean(f_np, axis=0))
            utterance_features = np.array(utterance_features)
            
            # Metric 1: Distance to centroid
            centroid = np.mean(all_seen_features, axis=0)
            centroid_distances = euclidean_distances(utterance_features, centroid.reshape(1, -1)).flatten()
            
            # Metric 2: k-NN distance (distance to k nearest neighbors)
            k = min(10, len(all_seen_features) - 1)
            nbrs = NearestNeighbors(n_neighbors=k).fit(all_seen_features)
            knn_distances, _ = nbrs.kneighbors(utterance_features)
            avg_knn_distances = np.mean(knn_distances, axis=1)
            
            # Metric 3: Local density (inverse of average distance to k neighbors)
            local_density = 1.0 / (avg_knn_distances + 1e-8)
            
            # Metric 4: Isolation score (distance to nearest neighbor)
            nn_distances = knn_distances[:, 0]  # Distance to closest neighbor
            
            metrics[data_type] = {
                'centroid_distance': centroid_distances,
                'knn_distance': avg_knn_distances,
                'local_density': local_density,
                'isolation_score': nn_distances,
                'paths': high_conf_data[data_type]['paths'],
                'mia_scores': high_conf_data[data_type]['scores'],
                'confidences': high_conf_data[data_type]['confidences']
            }
        
        return metrics
    
    def _statistical_analysis(self, rarity_metrics):
        """Perform statistical analysis on rarity patterns"""
        
        analysis = {}
        
        if len(rarity_metrics['seen']) == 0:
            return {"error": "No high-confidence seen samples found"}
        
        seen_metrics = rarity_metrics['seen']
        
        # Analyse correlations between MIA confidence and rarity
        metrics_to_analyse = ['centroid_distance', 'knn_distance', 'isolation_score']
        
        for metric in metrics_to_analyse:
            if metric in seen_metrics:
                # Correlation between rarity and MIA confidence
                correlation = np.corrcoef(seen_metrics[metric], seen_metrics['confidences'])[0, 1]
                
                analysis[f'{metric}_correlation'] = {
                    'correlation': correlation,
                    'mean_rarity': np.mean(seen_metrics[metric]),
                    'std_rarity': np.std(seen_metrics[metric]),
                    'most_rare_samples': self._get_most_rare_samples(seen_metrics, metric)
                }
        
        return analysis
    
    def _get_most_rare_samples(self, metrics, metric_name, top_k=10):
        """Get the most rare samples according to a specific metric"""
        
        metric_values = metrics[metric_name]
        top_indices = np.argsort(metric_values)[-top_k:]  # Most rare (highest values)
        
        most_rare = []
        for idx in top_indices:
            most_rare.append({
                'path': metrics['paths'][idx],
                'rarity_score': metric_values[idx],
                'mia_score': metrics['mia_scores'][idx],
                'confidence': metrics['confidences'][idx]
            })
        
        return most_rare
    
    def generate_report(self, output_path, high_conf_data, rarity_analysis):
        """Generate comprehensive analysis report"""
        
        report = {
            'summary': {
                'total_seen_samples': len(self.seen_dataset),
                'total_unseen_samples': len(self.unseen_dataset),
                'high_conf_seen': len(high_conf_data['seen']['indices']),
                'high_conf_unseen': len(high_conf_data['unseen']['indices']),
                'threshold_used': high_conf_data['threshold']
            },
            'rarity_analysis': rarity_analysis['analysis'],
            'detailed_samples': {
                'seen': rarity_analysis['memorisation_rarity_scores'],
                'unseen': rarity_analysis['generalisation_rarity_scores']
            }
        }
        
        # Save detailed CSV files
        if len(high_conf_data['seen']['indices']) > 0:
            seen_df = self._create_detailed_dataframe(high_conf_data['seen'], 
                                                    rarity_analysis['memorisation_rarity_scores'])
            seen_df.to_csv(os.path.join(output_path, f"{self.model_name}_high_conf_seen_detailed.csv"), index=False)
        
        if len(high_conf_data['unseen']['indices']) > 0:
            unseen_df = self._create_detailed_dataframe(high_conf_data['unseen'], 
                                                      rarity_analysis['generalisation_rarity_scores'])
            unseen_df.to_csv(os.path.join(output_path, f"{self.model_name}_high_conf_unseen_detailed.csv"), index=False)
        
        # Save summary report
        import json
        with open(os.path.join(output_path, f"{self.model_name}_memorisation_analysis.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _create_detailed_dataframe(self, high_conf_data, rarity_metrics):
        """Create detailed DataFrame for analysis"""
        
        if len(rarity_metrics) == 0:
            return pd.DataFrame()
        
        df_data = {
            'file_path': rarity_metrics['paths'],
            'mia_score': rarity_metrics['mia_scores'],
            'mia_confidence': rarity_metrics['confidences'],
            'centroid_distance': rarity_metrics['centroid_distance'],
            'knn_distance': rarity_metrics['knn_distance'],
            'local_density': rarity_metrics['local_density'],
            'isolation_score': rarity_metrics['isolation_score']
        }
        
        return pd.DataFrame(df_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",default="/work3/s194632/LibriSpeech_features")
    parser.add_argument("--model", default="wav2vec2")
    parser.add_argument("--model_info_path", 
                       default="/work3/s194632/MIA_results/customized-utterance-model-info-wav2vec2.json",
                       help="Path to model info JSON file from training")
    parser.add_argument("--output_path", default="/work3/s194632/MIA_results/memorisation_results")
    parser.add_argument("--confidence_threshold", type=float, default=0.9)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialise analyser
    analyser = MemorisationAnalyser(
        args.base_path,
        args.model,
        args.model_info_path
    )
    
    # Extract high-confidence samples
    print("Extracting high-confidence MIA predictions...")
    high_conf_data = analyser.extract_high_confidence_samples(args.confidence_threshold)
    
    # Analyse rarity patterns
    print("Analysing rarity patterns...")
    rarity_analysis = analyser.analyse_rarity_patterns(high_conf_data)
    
    # Generate report
    print("Generating analysis report...")
    report = analyser.generate_report(args.output_path, high_conf_data, rarity_analysis)
    
    print(f"\nAnalysis complete! Results saved to {args.output_path}")
    print(f"High-confidence seen samples found: {len(high_conf_data['seen']['indices'])}")
    print(f"High-confidence unseen samples found: {len(high_conf_data['unseen']['indices'])}")


if __name__ == "__main__":
    main()