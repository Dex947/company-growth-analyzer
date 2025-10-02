"""
Comprehensive model evaluation with multiple metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report as sklearn_classification_report
)
from scipy.stats import rankdata

from src.utils.logger import log
from src.utils.helpers import save_json
from config.config import OUTPUTS_PATH


class ModelEvaluator:
    """
    Evaluates model performance using classification and ranking metrics.

    Metrics:
    - Classification: Accuracy, Precision, Recall, F1, ROC-AUC
    - Ranking: NDCG (Normalized Discounted Cumulative Gain)

    Why ranking metrics? We care about relative ordering of companies,
    not just binary classification. A model that ranks correctly is more
    useful than one with high accuracy but poor ranking.
    """

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Evaluate classification performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                log.warning(f"Failed to calculate ROC-AUC: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Full classification report
        report = sklearn_classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report

        return metrics

    @staticmethod
    def evaluate_ranking(
        y_true: np.ndarray,
        y_score: np.ndarray,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate ranking performance using NDCG.

        Args:
            y_true: True relevance scores
            y_score: Predicted scores
            k: Ranking cutoff

        Returns:
            Dictionary with ranking metrics
        """
        # Calculate NDCG@k
        ndcg = ModelEvaluator._calculate_ndcg(y_true, y_score, k)

        return {
            'ndcg@k': ndcg,
            'k': k
        }

    @staticmethod
    def _calculate_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.

        NDCG measures ranking quality. Perfect ranking = 1.0.
        """
        # Sort by predicted scores
        order = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[order][:k]

        # DCG
        gains = 2 ** y_true_sorted - 1
        discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
        dcg = np.sum(gains / discounts)

        # Ideal DCG
        ideal_order = np.argsort(y_true)[::-1]
        y_true_ideal = y_true[ideal_order][:k]
        ideal_gains = 2 ** y_true_ideal - 1
        idcg = np.sum(ideal_gains / discounts[:len(ideal_gains)])

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def compare_models(
        models_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple models' performance.

        Args:
            models_results: Dictionary mapping model names to evaluation results

        Returns:
            DataFrame with comparison
        """
        comparison_data = []

        for model_name, results in models_results.items():
            row = {'model': model_name}

            # Extract key metrics
            if 'accuracy' in results:
                row['accuracy'] = results['accuracy']
            if 'f1_score' in results:
                row['f1_score'] = results['f1_score']
            if 'roc_auc' in results:
                row['roc_auc'] = results['roc_auc']
            if 'precision' in results:
                row['precision'] = results['precision']
            if 'recall' in results:
                row['recall'] = results['recall']

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by F1 score (or accuracy if F1 not available)
        sort_by = 'f1_score' if 'f1_score' in comparison_df.columns else 'accuracy'
        comparison_df = comparison_df.sort_values(sort_by, ascending=False)

        return comparison_df

    @staticmethod
    def generate_evaluation_report(
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predictions
            y_proba: Probabilities (optional)
            save_path: Path to save report

        Returns:
            Report dictionary
        """
        log.info(f"Generating evaluation report for {model_name}")

        report = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'class_distribution': {
                int(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))
            }
        }

        # Classification metrics
        classification_metrics = ModelEvaluator.evaluate_classification(y_true, y_pred, y_proba)
        report.update(classification_metrics)

        # Ranking metrics if probabilities available
        if y_proba is not None:
            scores = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
            ranking_metrics = ModelEvaluator.evaluate_ranking(y_true, scores)
            report['ranking_metrics'] = ranking_metrics

        # Save report
        if save_path:
            save_json(report, save_path)
        else:
            default_path = OUTPUTS_PATH / "reports" / f"{model_name}_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_json(report, default_path)

        log.info(f"Evaluation report generated: Accuracy={report['accuracy']:.4f}, F1={report['f1_score']:.4f}")

        return report
