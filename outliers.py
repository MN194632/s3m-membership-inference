import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

class OutlierDetectionAnalyzer:
    def __init__(self, results_path, model_name):
        self.results_path = results_path
        self.model_name = model_name
        self.load_analysis_results()
        
    def load_analysis_results(self):
        """Load the memorization analysis results"""
        
        # Load summary
        summary_path = os.path.join(self.results_path, f"{self.model_name}_memorisation_analysis.json")
        with open(summary_path, 'r') as f:
            self.summary = json.load(f)
        
        # Load detailed results
        seen_path = os.path.join(self.results_path, f"{self.model_name}_high_conf_seen_detailed.csv")
        unseen_path = os.path.join(self.results_path, f"{self.model_name}_high_conf_unseen_detailed.csv")
        
        self.memorization_df = pd.read_csv(seen_path) if os.path.exists(seen_path) else pd.DataFrame()
        self.generalization_df = pd.read_csv(unseen_path) if os.path.exists(unseen_path) else pd.DataFrame()
        
        if self.memorization_df.empty:
            raise ValueError("No memorization data found!")
        
        print(f"Loaded {len(self.memorization_df)} memorization candidates")
        print(f"Loaded {len(self.generalization_df)} generalization samples")
    
    def prepare_features(self):
        """Prepare feature matrix for outlier detection"""
        
        # Combine memorization and generalization samples
        if not self.generalization_df.empty:
            combined_df = pd.concat([self.memorization_df, self.generalization_df], ignore_index=True)
            labels = ['Memorization'] * len(self.memorization_df) + ['Generalization'] * len(self.generalization_df)
        else:
            combined_df = self.memorization_df.copy()
            labels = ['Memorization'] * len(self.memorization_df)
        
        # Extract feature columns (rarity metrics)
        feature_cols = ['centroid_distance', 'knn_distance', 'local_density', 'isolation_score']
        features = combined_df[feature_cols].values
        
        # Handle log transformation for isolation score
        features_processed = features.copy()
        features_processed[:, 3] = np.log10(features_processed[:, 3] + 1e-10)  # Log isolation score
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_processed)
        
        return features_scaled, features, combined_df, labels, feature_cols
    
    def apply_outlier_detection_methods(self, features_scaled, contamination=0.1):
        """Apply multiple outlier detection methods"""
        
        methods = {}
        
        print("Applying outlier detection methods...")
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_predictions = iso_forest.fit_predict(features_scaled)
        iso_scores = iso_forest.decision_function(features_scaled)
        methods['Isolation Forest'] = {
            'predictions': iso_predictions,
            'scores': iso_scores,
            'outliers': np.where(iso_predictions == -1)[0]
        }
        
        # 2. Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        lof_predictions = lof.fit_predict(features_scaled)
        lof_scores = lof.negative_outlier_factor_
        methods['Local Outlier Factor'] = {
            'predictions': lof_predictions,
            'scores': lof_scores,
            'outliers': np.where(lof_predictions == -1)[0]
        }
        
        # 3. One-Class SVM
        svm = OneClassSVM(nu=contamination)
        svm_predictions = svm.fit_predict(features_scaled)
        svm_scores = svm.decision_function(features_scaled)
        methods['One-Class SVM'] = {
            'predictions': svm_predictions,
            'scores': svm_scores,
            'outliers': np.where(svm_predictions == -1)[0]
        }
        
        # 4. Elliptic Envelope (Robust Covariance)
        elliptic = EllipticEnvelope(contamination=contamination, random_state=42)
        elliptic_predictions = elliptic.fit_predict(features_scaled)
        elliptic_scores = elliptic.decision_function(features_scaled)
        methods['Elliptic Envelope'] = {
            'predictions': elliptic_predictions,
            'scores': elliptic_scores,
            'outliers': np.where(elliptic_predictions == -1)[0]
        }
        
        # 5. DBSCAN (different approach - clustering)
        # Try different eps values to find good clustering
        eps_values = [0.3, 0.5, 0.7, 1.0]
        best_eps = 0.5
        best_silhouette = -1
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=3)
            cluster_labels = dbscan.fit_predict(features_scaled)
            if len(set(cluster_labels)) > 1 and -1 in cluster_labels:
                try:
                    silhouette = silhouette_score(features_scaled, cluster_labels)
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_eps = eps
                except:
                    pass
        
        dbscan = DBSCAN(eps=best_eps, min_samples=3)
        dbscan_labels = dbscan.fit_predict(features_scaled)
        dbscan_outliers = np.where(dbscan_labels == -1)[0]
        
        methods['DBSCAN'] = {
            'predictions': dbscan_labels,
            'scores': None,  # DBSCAN doesn't provide scores
            'outliers': dbscan_outliers,
            'eps_used': best_eps
        }
        
        return methods
    
    def create_dimensionality_reduction(self, features_scaled):
        """Create 2D embeddings for visualization"""
        
        embeddings = {}
        
        print("Creating dimensionality reductions...")
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings['PCA'] = pca.fit_transform(features_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_scaled)-1))
        embeddings['t-SNE'] = tsne.fit_transform(features_scaled)
        
        # UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings['UMAP'] = umap_reducer.fit_transform(features_scaled)
        
        return embeddings
    
    def visualize_outlier_methods(self, features_scaled, methods, embeddings, labels, combined_df, save_path):
        """Create comprehensive visualizations"""
        
        print("Creating visualizations...")
        
        # 1. Outlier scores comparison
        self.plot_outlier_scores_comparison(methods, combined_df, labels, save_path)
        
        # 2. 2D embeddings with outliers highlighted
        self.plot_2d_embeddings_with_outliers(embeddings, methods, labels, save_path)
        
        # 3. MIA confidence vs outlier scores
        self.plot_mia_vs_outlier_scores(methods, combined_df, save_path)
        
        # 4. Outlier detection summary
        self.plot_outlier_detection_summary(methods, labels, save_path)
        
        # 5. Feature importance for outliers
        self.plot_feature_analysis(methods, combined_df, save_path)
    
    def plot_outlier_scores_comparison(self, methods, combined_df, labels, save_path):
        """Compare outlier scores across methods"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        methods_with_scores = {name: data for name, data in methods.items() if data['scores'] is not None}
        
        for i, (method_name, method_data) in enumerate(methods_with_scores.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            scores = method_data['scores']
            
            # Create scatter plot
            mem_mask = np.array(labels) == 'Memorization'
            gen_mask = np.array(labels) == 'Generalization'
            
            if np.any(mem_mask):
                ax.scatter(range(np.sum(mem_mask)), scores[mem_mask], 
                          c='red', alpha=0.6, label='Memorization', s=50)
            
            if np.any(gen_mask):
                ax.scatter(range(np.sum(mem_mask), len(scores)), scores[gen_mask], 
                          c='blue', alpha=0.6, label='Generalization', s=50)
            
            # Highlight outliers
            outlier_indices = method_data['outliers']
            if len(outlier_indices) > 0:
                ax.scatter(outlier_indices, scores[outlier_indices], 
                          c='black', marker='x', s=100, label='Outliers')
            
            ax.set_title(f'{method_name}\n{len(outlier_indices)} outliers detected')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Outlier Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(methods_with_scores), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_outlier_scores_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_2d_embeddings_with_outliers(self, embeddings, methods, labels, save_path):
        """Plot 2D embeddings with outliers highlighted"""
        
        n_methods = len(methods)
        n_embeddings = len(embeddings)
        
        fig, axes = plt.subplots(n_embeddings, n_methods, figsize=(5*n_methods, 5*n_embeddings))
        if n_embeddings == 1:
            axes = axes.reshape(1, -1)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (emb_name, emb_data) in enumerate(embeddings.items()):
            for j, (method_name, method_data) in enumerate(methods.items()):
                ax = axes[i, j]
                
                # Plot all points
                mem_mask = np.array(labels) == 'Memorization'
                gen_mask = np.array(labels) == 'Generalization'
                
                if np.any(mem_mask):
                    ax.scatter(emb_data[mem_mask, 0], emb_data[mem_mask, 1], 
                              c='lightcoral', alpha=0.6, label='Memorization', s=30)
                
                if np.any(gen_mask):
                    ax.scatter(emb_data[gen_mask, 0], emb_data[gen_mask, 1], 
                              c='lightblue', alpha=0.6, label='Generalization', s=30)
                
                # Highlight outliers
                outlier_indices = method_data['outliers']
                if len(outlier_indices) > 0:
                    ax.scatter(emb_data[outlier_indices, 0], emb_data[outlier_indices, 1], 
                              c='black', marker='x', s=100, label=f'Outliers ({len(outlier_indices)})')
                
                ax.set_title(f'{emb_name} + {method_name}')
                ax.set_xlabel(f'{emb_name} 1')
                ax.set_ylabel(f'{emb_name} 2')
                if i == 0 and j == 0:  # Only show legend once
                    ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_2d_embeddings_outliers.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_mia_vs_outlier_scores(self, methods, combined_df, save_path):
        """Plot MIA confidence vs outlier scores"""
        
        methods_with_scores = {name: data for name, data in methods.items() if data['scores'] is not None}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (method_name, method_data) in enumerate(methods_with_scores.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            scores = method_data['scores']
            mia_confidence = combined_df['mia_confidence'].values
            
            # Scatter plot
            ax.scatter(scores, mia_confidence, alpha=0.6, s=50)
            
            # Highlight outliers
            outlier_indices = method_data['outliers']
            if len(outlier_indices) > 0:
                ax.scatter(scores[outlier_indices], mia_confidence[outlier_indices], 
                          c='red', marker='x', s=100, label=f'Outliers ({len(outlier_indices)})')
            
            # Calculate correlation
            correlation = np.corrcoef(scores, mia_confidence)[0, 1]
            
            ax.set_title(f'{method_name}\nCorr: {correlation:.3f}')
            ax.set_xlabel('Outlier Score')
            ax.set_ylabel('MIA Confidence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(methods_with_scores), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_mia_vs_outlier_scores.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_outlier_detection_summary(self, methods, labels, save_path):
        """Summary statistics of outlier detection"""
        
        # Count outliers per method
        outlier_counts = {}
        memorization_outliers = {}
        generalization_outliers = {}
        
        mem_indices = set(np.where(np.array(labels) == 'Memorization')[0])
        
        for method_name, method_data in methods.items():
            outlier_indices = set(method_data['outliers'])
            outlier_counts[method_name] = len(outlier_indices)
            memorization_outliers[method_name] = len(outlier_indices & mem_indices)
            generalization_outliers[method_name] = len(outlier_indices - mem_indices)
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Total outliers per method
        methods_list = list(outlier_counts.keys())
        counts = list(outlier_counts.values())
        
        bars = ax1.bar(methods_list, counts, color='steelblue', alpha=0.7)
        ax1.set_title('Number of Outliers Detected by Each Method')
        ax1.set_ylabel('Number of Outliers')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # Plot 2: Memorization vs Generalization outliers
        mem_counts = [memorization_outliers[method] for method in methods_list]
        gen_counts = [generalization_outliers[method] for method in methods_list]
        
        x = np.arange(len(methods_list))
        width = 0.35
        
        ax2.bar(x - width/2, mem_counts, width, label='Memorization', color='red', alpha=0.7)
        ax2.bar(x + width/2, gen_counts, width, label='Generalization', color='blue', alpha=0.7)
        
        ax2.set_title('Outliers by Sample Type')
        ax2.set_ylabel('Number of Outliers')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods_list, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_outlier_detection_summary.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return summary statistics
        return {
            'total_outliers': outlier_counts,
            'memorization_outliers': memorization_outliers,
            'generalization_outliers': generalization_outliers
        }
    
    def plot_feature_analysis(self, methods, combined_df, save_path):
        """Analyze which features are most important for outlier detection"""
        
        feature_cols = ['centroid_distance', 'knn_distance', 'local_density', 'isolation_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_cols):
            ax = axes[i]
            
            # Get feature values
            feature_values = combined_df[feature].values
            if feature == 'isolation_score':
                feature_values = np.log10(feature_values + 1e-10)  # Log scale
            
            # Plot distributions for each method's outliers
            for method_name, method_data in methods.items():
                if len(method_data['outliers']) > 0:
                    outlier_values = feature_values[method_data['outliers']]
                    ax.hist(outlier_values, alpha=0.5, label=f'{method_name} outliers', bins=10)
            
            # Plot overall distribution
            ax.hist(feature_values, alpha=0.3, label='All samples', bins=20, color='gray')
            
            title = feature.replace('_', ' ').title()
            if feature == 'isolation_score':
                title += ' (Log Scale)'
            
            ax.set_title(title)
            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_feature_analysis_outliers.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_outlier_report(self, methods, summary_stats, save_path):
        """Generate comprehensive outlier detection report"""
        
        report = {
            'outlier_detection_summary': summary_stats,
            'method_details': {}
        }
        
        for method_name, method_data in methods.items():
            report['method_details'][method_name] = {
                'total_outliers': len(method_data['outliers']),
                'outlier_indices': method_data['outliers'].tolist(),
                'has_scores': method_data['scores'] is not None
            }
            
            if method_name == 'DBSCAN':
                report['method_details'][method_name]['eps_used'] = method_data['eps_used']
        
        # Save report
        with open(os.path.join(save_path, f"{self.model_name}_outlier_detection_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create text summary
        summary_lines = [
            f"# Outlier Detection Analysis Report: {self.model_name}",
            "",
            "## Summary",
            f"- Total samples analyzed: {len(self.memorization_df) + len(self.generalization_df)}",
            f"- Memorization candidates: {len(self.memorization_df)}",
            f"- Generalization samples: {len(self.generalization_df)}",
            "",
            "## Outlier Detection Results",
            ""
        ]
        
        for method_name, count in summary_stats['total_outliers'].items():
            mem_count = summary_stats['memorization_outliers'][method_name]
            gen_count = summary_stats['generalization_outliers'][method_name]
            
            summary_lines.extend([
                f"### {method_name}",
                f"- Total outliers: {count}",
                f"- Memorization outliers: {mem_count}",
                f"- Generalization outliers: {gen_count}",
                f"- % of memorization samples flagged: {mem_count/len(self.memorization_df)*100:.1f}%",
                ""
            ])
        
        with open(os.path.join(save_path, f"{self.model_name}_outlier_summary.md"), 'w') as f:
            f.write('\n'.join(summary_lines))
        
        return report
    
    def run_full_analysis(self, save_path, contamination=0.1):
        """Run complete outlier detection analysis"""
        
        os.makedirs(save_path, exist_ok=True)
        
        print("Preparing features...")
        features_scaled, features_raw, combined_df, labels, feature_cols = self.prepare_features()
        
        print("Applying outlier detection methods...")
        methods = self.apply_outlier_detection_methods(features_scaled, contamination)
        
        print("Creating dimensionality reductions...")
        embeddings = self.create_dimensionality_reduction(features_scaled)
        
        print("Creating visualizations...")
        self.visualize_outlier_methods(features_scaled, methods, embeddings, labels, combined_df, save_path)
        
        print("Generating summary statistics...")
        summary_stats = self.plot_outlier_detection_summary(methods, labels, save_path)
        
        print("Generating report...")
        report = self.generate_outlier_report(methods, summary_stats, save_path)
        
        print(f"Analysis complete! Results saved to {save_path}")
        
        return methods, summary_stats, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default ="/work3/s194632/MIA_results/memorisation_results", help="Path to memorization analysis results")
    parser.add_argument("--model", default="wav2vec2", help="Model name")
    parser.add_argument("--output_path", default="/work3/s194632/MIA_results/memorisation_results", help="Path to save outlier analysis results")
    parser.add_argument("--contamination", type=float, default=0.1, help="Expected fraction of outliers")
    
    args = parser.parse_args()
    
    analyzer = OutlierDetectionAnalyzer(args.results_path, args.model)
    methods, summary_stats, report = analyzer.run_full_analysis(args.output_path, args.contamination)
    
    # Print key findings
    print(f"\nKey Findings:")
    print(f"- Methods tested: {list(summary_stats['total_outliers'].keys())}")
    for method, count in summary_stats['total_outliers'].items():
        mem_outliers = summary_stats['memorization_outliers'][method]
        print(f"- {method}: {count} total outliers ({mem_outliers} memorization, {count-mem_outliers} generalization)")


if __name__ == "__main__":
    main()