"""
Comprehensive uncertainty analysis and comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


class UncertaintyAnalyzer:
    """
    Comprehensive uncertainty analysis for comparing models
    """
    
    def __init__(self, save_dir: str = "results/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.results = {}
    
    def add_results(
        self,
        name: str,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainties: np.ndarray,
        confidences: np.ndarray
    ):
        """Add results for a model"""
        correct = (predictions == labels).astype(float)
        
        self.results[name] = {
            'predictions': predictions,
            'labels': labels,
            'uncertainties': uncertainties,
            'confidences': confidences,
            'correct': correct
        }
        
        logger.info(f"Added results for {name}: {len(predictions)} samples")
    
    def compute_flagging_metrics(
        self,
        name: str,
        threshold: float = 0.5
    ) -> Dict:
        """Compute flagging metrics for a model"""
        result = self.results[name]
        
        uncertainties = result['uncertainties']
        correct = result['correct']
        
        # Flagged samples (high uncertainty)
        flagged = uncertainties > threshold
        
        metrics = {
            'flag_rate': np.mean(flagged),
            'num_flagged': np.sum(flagged),
            'num_confident': np.sum(~flagged),
        }
        
        # Accuracy on confident predictions
        if np.sum(~flagged) > 0:
            metrics['confident_accuracy'] = np.mean(correct[~flagged])
        else:
            metrics['confident_accuracy'] = 0.0
        
        # Flag precision (% of flagged that are actually errors)
        if np.sum(flagged) > 0:
            metrics['flag_precision'] = np.mean(1 - correct[flagged])
        else:
            metrics['flag_precision'] = 0.0
        
        # Uncertainty-error correlation
        correlation, p_value = spearmanr(uncertainties, 1 - correct)
        metrics['uncertainty_error_correlation'] = correlation
        metrics['correlation_p_value'] = p_value
        
        # AUC for uncertainty as error predictor
        try:
            auc = roc_auc_score(1 - correct, uncertainties)
            metrics['uncertainty_auc'] = auc
        except:
            metrics['uncertainty_auc'] = 0.5
        
        return metrics
    
    def compare_models(self, threshold: float = 0.5) -> Dict:
        """Compare all models"""
        comparison = {}
        
        for name in self.results.keys():
            comparison[name] = self.compute_flagging_metrics(name, threshold)
            
            # Add basic accuracy
            comparison[name]['accuracy'] = np.mean(self.results[name]['correct'])
            comparison[name]['mean_uncertainty'] = np.mean(self.results[name]['uncertainties'])
        
        return comparison
    
    def plot_uncertainty_distribution(self):
        """Plot uncertainty distribution comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, result in self.results.items():
            ax.hist(
                result['uncertainties'],
                bins=30,
                alpha=0.5,
                label=name,
                density=True
            )
        
        ax.set_xlabel('Uncertainty (Entropy)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Uncertainty Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'uncertainty_distribution.png', dpi=300)
        plt.close()
        
        logger.info("Saved uncertainty distribution plot")
    
    def plot_uncertainty_vs_correctness(self):
        """Plot uncertainty vs correctness scatter"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, result) in zip(axes, self.results.items()):
            ax.scatter(
                result['uncertainties'],
                result['correct'],
                alpha=0.3,
                s=10
            )
            ax.set_xlabel('Uncertainty', fontsize=11)
            ax.set_ylabel('Correct (1) / Incorrect (0)', fontsize=11)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'uncertainty_vs_correctness.png', dpi=300)
        plt.close()
        
        logger.info("Saved uncertainty vs correctness plot")
    
    def plot_roc_curves(self):
        """Plot ROC curves for uncertainty as error detector"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for name, result in self.results.items():
            errors = 1 - result['correct']
            uncertainties = result['uncertainties']
            
            fpr, tpr, _ = roc_curve(errors, uncertainties)
            auc = roc_auc_score(errors, uncertainties)
            
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Uncertainty as Error Detector (ROC)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_uncertainty_error.png', dpi=300)
        plt.close()
        
        logger.info("Saved ROC curve plot")
    
    def plot_flagging_tradeoff(self):
        """Plot flag rate vs accuracy trade-off"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        thresholds = np.linspace(0, 1, 20)
        
        for name, result in self.results.items():
            flag_rates = []
            accuracies = []
            
            for thresh in thresholds:
                confident = result['uncertainties'] < thresh
                
                flag_rate = 1 - np.mean(confident)
                flag_rates.append(flag_rate)
                
                if np.sum(confident) > 0:
                    accuracy = np.mean(result['correct'][confident])
                else:
                    accuracy = 0
                accuracies.append(accuracy)
            
            ax.plot(flag_rates, accuracies, marker='o', label=name, linewidth=2)
        
        ax.set_xlabel('Flag Rate (% sent for review)', fontsize=12)
        ax.set_ylabel('Accuracy on Confident Predictions', fontsize=12)
        ax.set_title('Flag Rate vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'flagging_tradeoff.png', dpi=300)
        plt.close()
        
        logger.info("Saved flagging trade-off plot")
    
    def plot_per_class_uncertainty(self, class_names: list):
        """Plot uncertainty by tumor type"""
        n_models = len(self.results)
        n_classes = len(class_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(n_classes)
        width = 0.8 / n_models
        
        for i, (name, result) in enumerate(self.results.items()):
            class_uncertainties = []
            
            for class_idx in range(n_classes):
                mask = result['labels'] == class_idx
                if np.sum(mask) > 0:
                    class_uncertainties.append(np.mean(result['uncertainties'][mask]))
                else:
                    class_uncertainties.append(0)
            
            ax.bar(
                x + i * width,
                class_uncertainties,
                width,
                label=name
            )
        
        ax.set_xlabel('Tumor Type', fontsize=12)
        ax.set_ylabel('Mean Uncertainty', fontsize=12)
        ax.set_title('Uncertainty by Tumor Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'per_class_uncertainty.png', dpi=300)
        plt.close()
        
        logger.info("Saved per-class uncertainty plot")
    
    def generate_all_plots(self, class_names: list = None):
        """Generate all analysis plots"""
        logger.info("Generating comprehensive uncertainty analysis plots...")
        
        self.plot_uncertainty_distribution()
        self.plot_uncertainty_vs_correctness()
        self.plot_roc_curves()
        self.plot_flagging_tradeoff()
        
        if class_names:
            self.plot_per_class_uncertainty(class_names)
        
        logger.info("All plots generated successfully!")
    
    def save_comparison_table(self, filename: str = "uncertainty_comparison.txt"):
        """Save comparison table"""
        comparison = self.compare_models()
        
        filepath = self.save_dir.parent / filename
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UNCERTAINTY ANALYSIS COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            for name, metrics in comparison.items():
                f.write(f"\n{name}:\n")
                f.write("-"*40 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        logger.info(f"Saved comparison table to {filepath}")