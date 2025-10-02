"""
Full demonstration of the Company Growth Analyzer with visualizations and results.
Uses 15 companies for meaningful ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_ingestion.data_aggregator import DataAggregator
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.explainability.explanation_generator import ExplanationGenerator
from src.evaluation.evaluator import ModelEvaluator
from src.visualization.visualizer import ModelVisualizer
from src.utils.logger import log
from src.utils.helpers import save_dataframe
from config.config import DATA_PROCESSED_PATH, OUTPUTS_PATH


def main():
    """
    Run complete analysis with visualizations.
    """
    print("=" * 80)
    print("Company Growth Analyzer - Full Demo with Results")
    print("=" * 80)

    # Use 15 companies for meaningful analysis
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',  # Big Tech
        'META', 'NVDA', 'AMD', 'INTC', 'NFLX',     # Tech & Entertainment
        'DIS', 'COST', 'WMT', 'PEP', 'KO'          # Consumer & Retail
    ]

    print(f"\nAnalyzing {len(tickers)} companies: {', '.join(tickers)}")
    print("\nStep 1: Collecting data...")

    # 1. Collect data
    aggregator = DataAggregator()
    try:
        df_raw = aggregator.collect_all_data(
            tickers=tickers,
            period='1y',
            include_sentiment=True,
            include_market=True,
            save_raw=False
        )
        print(f"[OK] Collected data: {df_raw.shape[0]} companies, {df_raw.shape[1]} features")
    finally:
        aggregator.close()

    # 2. Feature engineering
    print("\nStep 2: Engineering features...")
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.engineer_features(df_raw)
    print(f"[OK] Engineered features: {df_engineered.shape[1]} total features")

    # 3. Create target variable using 6-month returns (more data available)
    if 'returns_6m' in df_engineered.columns:
        # Use 6-month returns as they're more likely to have data
        median_return = df_engineered['returns_6m'].median()
        df_engineered['success_label'] = (df_engineered['returns_6m'] > median_return).astype(int)
        print(f"[OK] Created target (6m returns > {median_return:.2%})")
    else:
        # Fallback: use market cap percentile as proxy for success
        df_engineered['success_label'] = (
            df_engineered['market_cap'] > df_engineered['market_cap'].median()
        ).astype(int)
        print("[OK] Created target (market cap based)")

    # 4. Preprocess
    print("\nStep 3: Preprocessing data...")
    preprocessor = DataPreprocessor(scaler_type='robust')
    df_processed = preprocessor.fit_transform(df_engineered, target_col='success_label')
    print(f"[OK] Preprocessed: {df_processed.shape[0]} samples, {df_processed.shape[1]} features")

    # Save processed data
    save_dataframe(df_processed, DATA_PROCESSED_PATH / "demo_processed.parquet")

    # 5. Train models with smaller CV folds
    print("\nStep 4: Training models...")
    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df_processed,
        target_col='success_label'
    )

    # Adjust CV folds for sample size
    from config import config
    original_cv = config.CV_FOLDS
    config.CV_FOLDS = min(3, len(X_train) // 3)  # Use 3-fold or less

    # Train models
    results = trainer.train_models(
        model_names=['random_forest', 'xgboost'],
        use_cv=True
    )

    config.CV_FOLDS = original_cv  # Restore

    print(f"[OK] Trained {len(results)} models\n")

    # 6. Evaluate
    print("Step 5: Evaluating models...")
    evaluations = {}

    for name, result in results.items():
        predictions = trainer.get_predictions(name)

        eval_result = ModelEvaluator.evaluate_classification(
            y_true=y_test.values,
            y_pred=predictions['test_pred'],
            y_proba=predictions.get('test_proba')
        )

        evaluations[name] = eval_result

        print(f"\n{name}:")
        print(f"  Accuracy:  {eval_result['accuracy']:.4f}")
        print(f"  Precision: {eval_result['precision']:.4f}")
        print(f"  Recall:    {eval_result['recall']:.4f}")
        print(f"  F1 Score:  {eval_result['f1_score']:.4f}")
        if 'roc_auc' in eval_result:
            print(f"  ROC-AUC:   {eval_result['roc_auc']:.4f}")

    # Model comparison
    comparison = ModelEvaluator.compare_models(evaluations)
    print(f"\nModel Comparison:")
    print(comparison.to_string(index=False))

    # 7. Generate explanations
    print("\nStep 6: Generating explanations...")
    best_name, best_model = trainer.get_best_model()
    print(f"Best model: {best_name}")

    # Initialize explainer
    explainer = ExplanationGenerator(best_model, X_train, y_train)

    # Global explanation
    report = explainer.generate_report(X_test, y_test)

    if 'top_features_shap' in report:
        print(f"\nTop 10 Features (SHAP):")
        top_features_df = pd.DataFrame(report['top_features_shap'])
        for i, row in top_features_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

    # 8. Create visualizations
    print("\nStep 7: Creating visualizations...")
    viz_dir = OUTPUTS_PATH / "visualizations" / "demo"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance plot
    if 'top_features_shap' in report:
        ModelVisualizer.plot_feature_importance(
            top_features_df,
            title=f"Top Features ({best_name})",
            top_n=15,
            save_path=viz_dir / "feature_importance.png"
        )
        print("[OK] Saved feature_importance.png")

    # Confusion matrix
    if 'confusion_matrix' in evaluations[best_name]:
        ModelVisualizer.plot_confusion_matrix(
            np.array(evaluations[best_name]['confusion_matrix']),
            class_names=["Low Performer", "High Performer"],
            title=f"Confusion Matrix ({best_name})",
            save_path=viz_dir / "confusion_matrix.png"
        )
        print("[OK] Saved confusion_matrix.png")

    # Model comparison
    ModelVisualizer.plot_model_comparison(
        comparison,
        metric='f1_score',
        title="Model Performance Comparison",
        save_path=viz_dir / "model_comparison.png"
    )
    print("[OK] Saved model_comparison.png")

    # 9. Company rankings
    print("\nStep 8: Generating company rankings...")

    # Make predictions on all data
    X_full = df_processed[trainer.feature_names]
    predictions = best_model.predict(X_full)
    probabilities = best_model.predict_proba(X_full)

    # Create ranking DataFrame
    ranking_df = pd.DataFrame({
        'company': df_raw['company_name'].values,
        'ticker': df_raw['ticker'].values,
        'prediction': predictions,
        'success_probability': probabilities[:, 1],
        'market_cap': df_raw['market_cap'].values,
        'sector': df_raw['sector'].values
    }).sort_values('success_probability', ascending=False)

    print(f"\nCompany Success Ranking:")
    print(ranking_df[['company', 'ticker', 'success_probability', 'sector']].to_string(index=False))

    # Save ranking
    save_dataframe(ranking_df, OUTPUTS_PATH / "reports" / "company_rankings.csv", format='csv')
    print(f"\n[OK] Saved company_rankings.csv")

    # Visualize ranking
    ModelVisualizer.plot_company_ranking(
        ranking_df,
        score_col='success_probability',
        name_col='company',
        title="Company Success Probability Ranking",
        top_n=15,
        save_path=viz_dir / "company_ranking.png"
    )
    print("[OK] Saved company_ranking.png")

    # 10. Individual explanations
    print("\nStep 9: Generating individual explanations...")

    # Explain top 3 and bottom 3
    top_3_idx = ranking_df.head(3).index.tolist()
    bottom_3_idx = ranking_df.tail(3).index.tolist()

    print("\n" + "=" * 80)
    print("Sample Predictions & Explanations:")
    print("=" * 80)

    for idx in top_3_idx[:2]:  # Show top 2
        company_name = ranking_df.loc[idx, 'company']

        # Find position in X_full
        data_idx = list(X_full.index).index(idx)

        explanation = explainer.explain_prediction(
            X_full,
            index=data_idx,
            company_name=company_name,
            top_n_features=3
        )

        print(f"\n{company_name} ({ranking_df.loc[idx, 'ticker']}):")
        print(f"  Prediction: {'High Performer' if explanation['prediction'] == 1 else 'Low Performer'}")
        if explanation['confidence']:
            print(f"  Confidence: {explanation['confidence']:.1%}")

        if 'shap_contributions' in explanation:
            print(f"  Top factors:")
            for contrib in explanation['shap_contributions'][:3]:
                direction = "+" if contrib['impact'] > 0 else "-"
                print(f"    {direction} {contrib['feature']}: {abs(contrib['impact']):.3f}")

    print("\n" + "=" * 80)
    print("[OK] Demo complete!")
    print("\nResults saved to:")
    print(f"  - Visualizations: {viz_dir}")
    print(f"  - Rankings: {OUTPUTS_PATH / 'reports' / 'company_rankings.csv'}")
    print(f"  - Processed data: {DATA_PROCESSED_PATH / 'demo_processed.parquet'}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        log.error(f"Error in demo: {e}", exc_info=True)
        print(f"\n[X] Error: {e}")
        print("\nCheck logs/ directory for details")
