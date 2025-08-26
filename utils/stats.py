import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

class ModelStatistics:
    """Comprehensive statistical analysis for ML models"""
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_prob=None, 
                             class_names=None) -> Dict[str, float]:
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError:
                pass  # Skip if not applicable
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def statistical_significance_test(results1: List[float], 
                                    results2: List[float], 
                                    test_type='ttest') -> Dict[str, float]:
        """Perform statistical significance tests between two sets of results"""
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(results1, results2)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(results1, results2)
        elif test_type == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(results1, results2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant_001': p_value < 0.001,
            'significant_01': p_value < 0.01,
            'significant_05': p_value < 0.05
        }
    
    @staticmethod
    def cross_validation_analysis(cv_scores: List[float]) -> Dict[str, float]:
        """Analyze cross-validation results"""
        return {
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'min': np.min(cv_scores),
            'max': np.max(cv_scores),
            'median': np.median(cv_scores),
            'confidence_interval_95': stats.t.interval(
                0.95, len(cv_scores)-1, 
                loc=np.mean(cv_scores), 
                scale=stats.sem(cv_scores)
            )
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                            normalize=False, figsize=(8, 6)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        return plt.gcf()
    
    @staticmethod
    def plot_learning_curves(train_scores: List[float], 
                           val_scores: List[float], 
                           epochs: Optional[List[int]] = None,
                           metric_name: str = 'Score',
                           figsize=(10, 6)):
        """Plot learning curves"""
        if epochs is None:
            epochs = list(range(1, len(train_scores) + 1))
        
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}')
        plt.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        return plt.gcf()
    
    @staticmethod
    def model_comparison_report(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a comparison report for multiple models"""
        df = pd.DataFrame(models_results).T
        
        # Add ranking for key metrics
        if 'accuracy' in df.columns:
            df['accuracy_rank'] = df['accuracy'].rank(ascending=False)
        if 'f1_macro' in df.columns:
            df['f1_rank'] = df['f1_macro'].rank(ascending=False)
        
        return df.round(4)

class ExperimentLogger:
    """Log and track ML experiments"""
    
    def __init__(self):
        self.experiments = []
    
    def log_experiment(self, model_name: str, params: Dict, 
                      metrics: Dict, notes: str = ""):
        """Log a single experiment"""
        experiment = {
            'timestamp': pd.Timestamp.now(),
            'model_name': model_name,
            'parameters': params,
            'metrics': metrics,
            'notes': notes
        }
        self.experiments.append(experiment)
    
    def get_experiment_history(self) -> pd.DataFrame:
        """Get experiment history as DataFrame"""
        if not self.experiments:
            return pd.DataFrame()
        
        rows = []
        for exp in self.experiments:
            row = {
                'timestamp': exp['timestamp'],
                'model_name': exp['model_name'],
                'notes': exp['notes']
            }
            row.update(exp['parameters'])
            row.update(exp['metrics'])
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def best_experiment(self, metric: str, ascending: bool = False) -> Dict:
        """Get best experiment based on a metric"""
        if not self.experiments:
            return None
        
        best_exp = min(self.experiments, 
                      key=lambda x: x['metrics'].get(metric, float('inf') if not ascending else float('-inf')))
        return best_exp