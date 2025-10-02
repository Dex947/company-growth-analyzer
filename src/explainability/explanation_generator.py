"""
Generate human-readable explanations from model predictions and feature attributions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.feature_importance import FeatureImportanceAnalyzer
from src.utils.logger import log
from src.utils.helpers import save_json
from config.config import OUTPUTS_PATH, ENABLE_SHAP, ENABLE_LIME


class ExplanationGenerator:
    """
    Generates comprehensive, human-readable explanations.

    Output format:
    - Prediction score with confidence
    - Top 5 factors driving the prediction
    - Comparison to similar companies
    - Caveats and limitations
    """

    def __init__(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Initialize explanation generator.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = X_train.columns.tolist()

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None

        if ENABLE_SHAP:
            try:
                self.shap_explainer = SHAPExplainer(model, X_train)
                log.info("SHAP explainer initialized")
            except Exception as e:
                log.warning(f"Failed to initialize SHAP: {e}")

        if ENABLE_LIME:
            try:
                self.lime_explainer = LIMEExplainer(model, X_train)
                log.info("LIME explainer initialized")
            except Exception as e:
                log.warning(f"Failed to initialize LIME: {e}")

    def explain_prediction(
        self,
        X: pd.DataFrame,
        index: int,
        company_name: str = None,
        top_n_features: int = 5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            X: Input features
            index: Sample index
            company_name: Name of company (for display)
            top_n_features: Number of top features to explain

        Returns:
            Dictionary with explanation components
        """
        # Get prediction
        prediction = self.model.predict(X.iloc[[index]])[0]

        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X.iloc[[index]])[0]
            confidence = max(proba)
        else:
            proba = None
            confidence = None

        explanation = {
            'company_name': company_name or f"Company_{index}",
            'prediction': int(prediction),
            'confidence': float(confidence) if confidence else None,
            'probability': proba.tolist() if proba is not None else None
        }

        # Get feature contributions from SHAP
        if self.shap_explainer:
            try:
                shap_exp = self.shap_explainer.explain_prediction(X, index)
                top_contributions = shap_exp['contributions'].head(top_n_features)

                explanation['shap_contributions'] = [
                    {
                        'feature': row['feature'],
                        'value': float(row['value']),
                        'impact': float(row['shap_value'])
                    }
                    for _, row in top_contributions.iterrows()
                ]
            except Exception as e:
                log.warning(f"SHAP explanation failed: {e}")

        # Get LIME explanation
        if self.lime_explainer:
            try:
                lime_exp = self.lime_explainer.explain_prediction(X, index, num_features=top_n_features)
                explanation['lime_contributions'] = lime_exp['contributions'].to_dict('records')
            except Exception as e:
                log.warning(f"LIME explanation failed: {e}")

        # Generate natural language explanation
        explanation['narrative'] = self._generate_narrative(explanation, X.iloc[index])

        return explanation

    def explain_comparison(
        self,
        X: pd.DataFrame,
        indices: List[int],
        company_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comparative explanation for multiple companies.

        Args:
            X: Input features
            indices: List of sample indices to compare
            company_names: Names of companies

        Returns:
            Comparison report
        """
        if company_names is None:
            company_names = [f"Company_{i}" for i in indices]

        predictions = []
        explanations = []

        for i, idx in enumerate(indices):
            exp = self.explain_prediction(X, idx, company_names[i])
            predictions.append(exp)
            explanations.append(exp)

        # Rank by prediction score
        if hasattr(self.model, 'predict_proba'):
            scores = [self.model.predict_proba(X.iloc[[idx]])[0][1] for idx in indices]
        else:
            scores = [self.model.predict(X.iloc[[idx]])[0] for idx in indices]

        ranking = pd.DataFrame({
            'company': company_names,
            'score': scores,
            'prediction': [self.model.predict(X.iloc[[idx]])[0] for idx in indices]
        }).sort_values('score', ascending=False)

        return {
            'ranking': ranking.to_dict('records'),
            'individual_explanations': explanations,
            'summary': self._generate_comparison_summary(ranking, explanations)
        }

    def _generate_narrative(self, explanation: Dict[str, Any], features: pd.Series) -> str:
        """
        Generate natural language explanation.

        Args:
            explanation: Explanation dictionary
            features: Feature values for the sample

        Returns:
            Human-readable explanation string
        """
        company = explanation['company_name']
        prediction = explanation['prediction']
        confidence = explanation.get('confidence')

        # Start with prediction
        if prediction == 1:
            outcome = "likely to succeed/outperform"
        else:
            outcome = "at higher risk/underperform"

        narrative = f"{company} is {outcome}"

        if confidence:
            narrative += f" (confidence: {confidence:.1%})"

        narrative += ".\n\n"

        # Add key drivers
        if 'shap_contributions' in explanation:
            narrative += "Key drivers:\n"

            for contrib in explanation['shap_contributions'][:3]:
                feature = contrib['feature']
                impact = contrib['impact']
                value = contrib['value']

                direction = "positively" if impact > 0 else "negatively"
                narrative += f"- {feature.replace('_', ' ').title()}: {direction} influences prediction"
                narrative += f" (value: {value:.2f}, impact: {impact:.3f})\n"

        # Add caveats
        narrative += "\nConsiderations:\n"
        narrative += "- Predictions are probabilistic, not deterministic\n"
        narrative += "- Model trained on historical data; future may differ\n"
        narrative += "- External factors (regulation, disruption) not captured\n"

        return narrative

    def _generate_comparison_summary(
        self,
        ranking: pd.DataFrame,
        explanations: List[Dict[str, Any]]
    ) -> str:
        """Generate summary for comparative analysis."""
        top_company = ranking.iloc[0]['company']
        top_score = ranking.iloc[0]['score']

        summary = f"Ranking Summary:\n"
        summary += f"Top performer: {top_company} (score: {top_score:.3f})\n\n"

        summary += "Rankings:\n"
        for i, row in ranking.iterrows():
            summary += f"{i+1}. {row['company']} - Score: {row['score']:.3f}\n"

        return summary

    def generate_report(
        self,
        X: pd.DataFrame,
        y_true: pd.Series = None,
        output_path: Path = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model explanation report.

        Args:
            X: Features
            y_true: True labels (optional)
            output_path: Path to save report

        Returns:
            Report dictionary
        """
        log.info("Generating explanation report")

        report = {
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_names),
            'n_samples': len(X)
        }

        # Global feature importance
        if self.shap_explainer:
            try:
                global_shap = self.shap_explainer.get_top_features(X, n=15)
                report['top_features_shap'] = global_shap.to_dict('records')
            except Exception as e:
                log.warning(f"Failed to get SHAP feature importance: {e}")

        # Permutation importance
        if y_true is not None:
            try:
                perm_imp = FeatureImportanceAnalyzer.get_permutation_importance(
                    self.model, X, y_true, n_repeats=5
                )
                report['top_features_permutation'] = perm_imp.head(15).to_dict('records')
            except Exception as e:
                log.warning(f"Failed to calculate permutation importance: {e}")

        # Save report
        if output_path is None:
            output_path = OUTPUTS_PATH / "reports" / f"explanation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

        save_json(report, output_path)

        return report
