"""
Quick start script for Company Growth Analyzer.
Demonstrates basic usage with minimal code.
"""

import pandas as pd
from pathlib import Path

from src.data_ingestion.data_aggregator import DataAggregator
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.explainability.explanation_generator import ExplanationGenerator
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import log
from config.config import DATA_PROCESSED_PATH


def main():
    """
    Run a quick analysis on sample companies.

    This demonstrates the complete pipeline:
    1. Data collection
    2. Preprocessing
    3. Model training
    4. Evaluation
    5. Explanation generation
    """

    print("=" * 80)
    print("Company Growth Analyzer - Quick Start")
    print("=" * 80)

    # Define companies to analyze (modify as needed)
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

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

    # 3. Create target variable (example: above-median 1-year return)
    if 'returns_1y' in df_engineered.columns:
        median_return = df_engineered['returns_1y'].median()
        df_engineered['success_label'] = (df_engineered['returns_1y'] > median_return).astype(int)
        print(f"[OK] Created target variable (threshold: {median_return:.2%})")
    else:
        print("[!] Warning: 'returns_1y' not available, using placeholder target")
        df_engineered['success_label'] = [0, 1] * (len(df_engineered) // 2) + [0] * (len(df_engineered) % 2)

    # 4. Preprocess
    print("\nStep 3: Preprocessing data...")
    preprocessor = DataPreprocessor(scaler_type='robust')
    df_processed = preprocessor.fit_transform(df_engineered, target_col='success_label')
    print(f"[OK] Preprocessed: {df_processed.shape[0]} samples, {df_processed.shape[1]} features")

    # 5. Train models
    print("\nStep 4: Training models...")
    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df_processed,
        target_col='success_label'
    )

    # Train with cross-validation
    results = trainer.train_models(
        model_names=['random_forest', 'xgboost'],
        use_cv=True
    )

    print(f"[OK] Trained {len(results)} models")

    # 6. Evaluate
    print("\nStep 5: Evaluating models...")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Train Score: {result['train_score']:.4f}")
        print(f"  Test Score:  {result['test_score']:.4f}")
        print(f"  CV Score:    {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")

    # 7. Generate explanations
    print("\nStep 6: Generating explanations...")
    best_name, best_model = trainer.get_best_model()

    explainer = ExplanationGenerator(best_model, X_train, y_train)

    # Global explanation
    report = explainer.generate_report(X_test, y_test)

    if 'top_features_shap' in report:
        print(f"\nTop 5 Features (SHAP):")
        for i, feature in enumerate(report['top_features_shap'][:5], 1):
            print(f"  {i}. {feature['feature']}: {feature['importance']:.4f}")

    # Individual predictions
    print("\n" + "=" * 80)
    print("Sample Predictions:")
    print("=" * 80)

    for i in range(min(3, len(X_test))):
        company_idx = X_test.index[i]
        company_name = df_raw.iloc[company_idx].get('company_name', f'Company_{i}')

        explanation = explainer.explain_prediction(
            X_test,
            index=i,
            company_name=company_name,
            top_n_features=3
        )

        print(f"\n{company_name}:")
        print(f"  Prediction: {'Success [OK]' if explanation['prediction'] == 1 else 'Risk [X]'}")
        if explanation['confidence']:
            print(f"  Confidence: {explanation['confidence']:.1%}")

        if 'shap_contributions' in explanation:
            print(f"  Top factors:")
            for contrib in explanation['shap_contributions'][:3]:
                direction = "+" if contrib['impact'] > 0 else "-"
                print(f"    {direction} {contrib['feature']}: {contrib['impact']:+.3f}")

    print("\n" + "=" * 80)
    print("[OK] Quick start complete!")
    print("\nNext steps:")
    print("  - Modify ticker list in this script")
    print("  - Explore notebooks/end_to_end_demo.ipynb for detailed analysis")
    print("  - Use CLI: python main.py --help")
    print("  - Read README.md for full documentation")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        log.error(f"Error in quickstart: {e}", exc_info=True)
        print(f"\n[X] Error: {e}")
        print("\nCheck logs/ directory for details")
