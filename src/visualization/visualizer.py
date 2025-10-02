"""
Visualization utilities for model results and explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
import shap

from src.utils.logger import log
from config.config import OUTPUTS_PATH

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelVisualizer:
    """
    Creates visualizations for model performance and explanations.

    Generates:
    - Feature importance plots
    - SHAP summary plots
    - Confusion matrices
    - ROC curves
    - Comparative heatmaps
    """

    @staticmethod
    def plot_feature_importance(
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 15,
        save_path: Optional[Path] = None
    ):
        """
        Plot feature importance bar chart.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))

        # Get top N features
        top_features = importance_df.head(top_n)

        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest at top

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Saved feature importance plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_shap_summary(
        shap_values: np.ndarray,
        X: pd.DataFrame,
        save_path: Optional[Path] = None
    ):
        """
        Create SHAP summary plot.

        Args:
            shap_values: SHAP values array
            X: Feature DataFrame
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 8))

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap.summary_plot(shap_values, X, show=False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Saved SHAP summary plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None
    ):
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix array
            class_names: Class labels
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))

        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_model_comparison(
        comparison_df: pd.DataFrame,
        metric: str = 'f1_score',
        title: str = "Model Comparison",
        save_path: Optional[Path] = None
    ):
        """
        Plot model performance comparison.

        Args:
            comparison_df: DataFrame with model names and metrics
            metric: Metric to compare
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 6))

        if metric not in comparison_df.columns:
            log.warning(f"Metric '{metric}' not found in comparison data")
            return

        plt.barh(range(len(comparison_df)), comparison_df[metric])
        plt.yticks(range(len(comparison_df)), comparison_df['model'])
        plt.xlabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.gca().invert_yaxis()

        # Add value labels
        for i, v in enumerate(comparison_df[metric]):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Saved model comparison plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_company_ranking(
        ranking_df: pd.DataFrame,
        score_col: str = 'score',
        name_col: str = 'company',
        title: str = "Company Success Ranking",
        top_n: int = 20,
        save_path: Optional[Path] = None
    ):
        """
        Plot company ranking visualization.

        Args:
            ranking_df: DataFrame with company names and scores
            score_col: Name of score column
            name_col: Name of company name column
            title: Plot title
            top_n: Number of companies to show
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 8))

        # Get top N
        top_companies = ranking_df.head(top_n)

        # Create color gradient based on score
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_companies)))

        plt.barh(range(len(top_companies)), top_companies[score_col], color=colors)
        plt.yticks(range(len(top_companies)), top_companies[name_col])
        plt.xlabel('Success Score')
        plt.title(title)
        plt.gca().invert_yaxis()

        # Add score labels
        for i, v in enumerate(top_companies[score_col]):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log.info(f"Saved ranking plot to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def create_dashboard(
        results: Dict[str, Any],
        save_dir: Optional[Path] = None
    ):
        """
        Create a comprehensive visualization dashboard.

        Args:
            results: Dictionary with all results
            save_dir: Directory to save visualizations
        """
        if save_dir is None:
            save_dir = OUTPUTS_PATH / "visualizations" / pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

        save_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Creating visualization dashboard in {save_dir}")

        # Feature importance
        if 'feature_importance' in results:
            ModelVisualizer.plot_feature_importance(
                results['feature_importance'],
                save_path=save_dir / "feature_importance.png"
            )

        # Confusion matrix
        if 'confusion_matrix' in results:
            ModelVisualizer.plot_confusion_matrix(
                np.array(results['confusion_matrix']),
                save_path=save_dir / "confusion_matrix.png"
            )

        # Model comparison
        if 'model_comparison' in results:
            ModelVisualizer.plot_model_comparison(
                results['model_comparison'],
                save_path=save_dir / "model_comparison.png"
            )

        # Company ranking
        if 'ranking' in results:
            ranking_df = pd.DataFrame(results['ranking'])
            ModelVisualizer.plot_company_ranking(
                ranking_df,
                save_path=save_dir / "company_ranking.png"
            )

        log.info(f"Dashboard created successfully in {save_dir}")
