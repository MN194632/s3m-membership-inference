import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
#import umap
import argparse
import os
import json

class EmbeddingAnalyzer:
    def __init__(self, results_path, model_name):
        self.results_path = results_path
        self.model_name = model_name
        self.load_results()
    
    def load_results(self):
        """Load the high-confidence analysis results"""
        
        # Load summary
        summary_path = os.path.join(self.results_path, f"{self.model_name}_memorisation_analysis.json")
        with open(summary_path, 'r') as f:
            self.summary = json.load(f)
        
        # Load detailed results
        seen_path = os.path.join(self.results_path, f"{self.model_name}_high_conf_seen_detailed.csv")
        unseen_path = os.path.join(self.results_path, f"{self.model_name}_high_conf_unseen_detailed.csv")
        
        self.seen_df = pd.read_csv(seen_path) if os.path.exists(seen_path) else pd.DataFrame()
        self.unseen_df = pd.read_csv(unseen_path) if os.path.exists(unseen_path) else pd.DataFrame()
    
    def create_rarity_distributions(self, save_path):
        """Create distribution plots for different rarity metrics"""
        
        if self.seen_df.empty:
            print("No memorisation candidates to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['centroid_distance', 'knn_distance', 'local_density', 'isolation_score']
        titles = ['Distance to Centroid', 'K-NN Distance', 'Local Density', 'Isolation Score (log scale)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            # Plot distribution for memorisation candidates
            if metric in self.seen_df.columns:
                data = self.seen_df[metric]
                
                # Use log scale for isolation score due to small values
                if metric == 'isolation_score':
                    data = np.log10(data + 1e-10)  # Add small constant to avoid log(0)
                    ax.set_xlabel('Log10(Isolation Score)')
                else:
                    ax.set_xlabel(title)
                
                ax.hist(data, bins=30, alpha=0.7, label='Memorisation Candidates', 
                       color='red', density=True)
            
            # Optionally plot false positives (commented out for focus)
            # if not self.unseen_df.empty and metric in self.unseen_df.columns:
            #     unseen_data = self.unseen_df[metric]
            #     if metric == 'isolation_score':
            #         unseen_data = np.log10(unseen_data + 1e-10)
            #     ax.hist(unseen_data, bins=30, alpha=0.7, label='False Positives', 
            #            color='blue', density=True)
            
            ax.set_ylabel('Density')
            ax.set_title(f'{title} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Rarity Distributions for Memorisation Candidates')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_memorisation_distributions.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_analysis(self, save_path):
        """Analyse correlations between MIA confidence and rarity metrics"""
        
        if self.seen_df.empty:
            return
        
        # Prepare data with log-scaled isolation score
        plot_data = self.seen_df.copy()
        plot_data['log_isolation_score'] = np.log10(plot_data['isolation_score'] + 1e-10)
        
        # Select numeric columns for correlation
        numeric_cols = ['mia_confidence', 'centroid_distance', 'knn_distance', 
                       'local_density', 'log_isolation_score']
        correlation_data = plot_data[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'{self.model_name}: Memorisation vs Rarity Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_memorisation_correlations.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rarity_metrics = ['centroid_distance', 'knn_distance', 'local_density', 'log_isolation_score']
        metric_labels = ['Centroid Distance', 'K-NN Distance', 'Local Density', 'Log10(Isolation Score)']
        
        for i, (metric, label) in enumerate(zip(rarity_metrics, metric_labels)):
            ax = axes[i//2, i%2]
            
            x = plot_data[metric]
            y = plot_data['mia_confidence']
            
            ax.scatter(x, y, alpha=0.6, color='red', s=20)
            
            # Add correlation coefficient
            corr = np.corrcoef(x, y)[0, 1]
            ax.set_title(f'{label} vs MIA Confidence\nCorr: {corr:.3f}')
            ax.set_xlabel(label)
            ax.set_ylabel('MIA Confidence')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Memorisation Confidence vs Rarity Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{self.model_name}_memorisation_scatter_plots.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def identify_memorisation_patterns(self, save_path):
        """Identify and analyze patterns in memorised samples"""
        
        if self.seen_df.empty:
            return {}
        
        analysis = {}
        
        # 1. Most memorable samples (highest MIA confidence)
        top_memorable = self.seen_df.nlargest(20, 'mia_confidence')
        analysis['most_memorable'] = {
            'count': len(top_memorable),
            'avg_confidence': top_memorable['mia_confidence'].mean(),
            'avg_centroid_distance': top_memorable['centroid_distance'].mean(),
            'avg_isolation': top_memorable['isolation_score'].mean(),
            'samples': top_memorable[['file_path', 'mia_confidence', 'centroid_distance', 'isolation_score']].to_dict('records')
        }
        
        # 2. Most rare samples (highest isolation score)
        top_rare = self.seen_df.nlargest(20, 'isolation_score')
        analysis['most_rare'] = {
            'count': len(top_rare),
            'avg_confidence': top_rare['mia_confidence'].mean(),
            'avg_centroid_distance': top_rare['centroid_distance'].mean(),
            'avg_isolation': top_rare['isolation_score'].mean(),
            'samples': top_rare[['file_path', 'mia_confidence', 'centroid_distance', 'isolation_score']].to_dict('records')
        }
        
        # 3. Outliers (high rarity + high confidence)
        high_conf_thresh = self.seen_df['mia_confidence'].quantile(0.8)
        high_rare_thresh = self.seen_df['isolation_score'].quantile(0.8)
        
        outliers = self.seen_df[
            (self.seen_df['mia_confidence'] >= high_conf_thresh) & 
            (self.seen_df['isolation_score'] >= high_rare_thresh)
        ]
        
        analysis['outliers'] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(self.seen_df) * 100,
            'avg_confidence': outliers['mia_confidence'].mean() if len(outliers) > 0 else 0,
            'samples': outliers[['file_path', 'mia_confidence', 'centroid_distance', 'isolation_score']].to_dict('records')
        }
        
        # Save analysis
        with open(os.path.join(save_path, f"{self.model_name}_memorisation_patterns.json"), 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def create_summary_report(self, save_path):
        """Create a comprehensive summary report"""
        
        report_lines = [
            f"# Memorisation Analysis Report: {self.model_name}",
            f"",
            f"## Summary Statistics",
            f"- Memorisation candidates: {len(self.seen_df)}",
            f"- False positives: {len(self.unseen_df)}",
            f"",
        ]
        
        if not self.seen_df.empty:
            # Basic statistics with log-scaled isolation
            log_isolation = np.log10(self.seen_df['isolation_score'] + 1e-10)
            
            report_lines.extend([
                f"## Rarity Metrics for Memorisation Candidates",
                f"",
                f"### Centroid Distance",
                f"- Mean: {self.seen_df['centroid_distance'].mean():.4f}",
                f"- Std: {self.seen_df['centroid_distance'].std():.4f}",
                f"",
                f"### Isolation Score (Log10 scale)",
                f"- Mean: {log_isolation.mean():.4f}",
                f"- Std: {log_isolation.std():.4f}",
                f"",
            ])
            
            # Correlations with log-scaled isolation
            if 'mia_confidence' in self.seen_df.columns:
                corr_centroid = np.corrcoef(self.seen_df['mia_confidence'], self.seen_df['centroid_distance'])[0, 1]
                corr_isolation = np.corrcoef(self.seen_df['mia_confidence'], log_isolation)[0, 1]
                
                report_lines.extend([
                    f"## Correlations with MIA Confidence",
                    f"- Centroid Distance: {corr_centroid:.4f}",
                    f"- Log10(Isolation Score): {corr_isolation:.4f}",
                    f"",
                ])
            
            # Top memorable samples
            top_5 = self.seen_df.nlargest(5, 'mia_confidence')
            report_lines.extend([
                f"## Top 5 Most Memorable Samples",
                f"",
            ])
            
            for idx, row in top_5.iterrows():
                filename = os.path.basename(row['file_path'])
                log_iso = np.log10(row['isolation_score'] + 1e-10)
                report_lines.append(f"1. {filename}")
                report_lines.append(f"   - MIA Confidence: {row['mia_confidence']:.4f}")
                report_lines.append(f"   - Centroid Distance: {row['centroid_distance']:.4f}")
                report_lines.append(f"   - Log10(Isolation): {log_iso:.4f}")
                report_lines.append("")
        
        # Save report
        with open(os.path.join(save_path, f"{self.model_name}_memorisation_report.md"), 'w') as f:
            f.write('\n'.join(report_lines))
    
    def run_full_analysis(self, save_path):
        """Run complete analysis pipeline"""
        
        os.makedirs(save_path, exist_ok=True)
        
        print("Creating rarity distribution plots...")
        self.create_rarity_distributions(save_path)
        
        print("Creating correlation analysis...")
        self.create_correlation_analysis(save_path)
        
        print("Identifying memorisation patterns...")
        patterns = self.identify_memorisation_patterns(save_path)
        
        print("Creating summary report...")
        self.create_summary_report(save_path)
        
        print(f"Analysis complete! Results saved to {save_path}")
        
        return patterns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path",  default="/work3/s194632/MIA_results/memorisation_results", help="Path to high-confidence analysis results")
    parser.add_argument("--model", default = "wav2vec2", help="Model name")
    parser.add_argument("--output_path",  default="/work3/s194632/MIA_results/memorisation_results", help="Path to save visualization results")
    
    args = parser.parse_args()
    
    analyzer = EmbeddingAnalyzer(args.results_path, args.model)
    patterns = analyzer.run_full_analysis(args.output_path)
    
    # Print key findings
    if patterns and 'outliers' in patterns:
        print(f"\nKey Findings:")
        print(f"- Found {patterns['outliers']['count']} samples with both high confidence and high rarity")
        print(f"- This represents {patterns['outliers']['percentage']:.1f}% of high-confidence samples")
        if patterns['outliers']['count'] > 0:
            print(f"- Average confidence for outliers: {patterns['outliers']['avg_confidence']:.4f}")


if __name__ == "__main__":
    main()