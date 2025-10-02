"""
Semiconductor Sector Analysis Demo

Analyzes 14 semiconductor companies using sector-specific factor models.

Features:
- Sector-specific comparison (semiconductors only)
- Factor-based dimensionality reduction (70+ features → 5 factors)
- Sector-relative performance targets
- Multiple ML models with cross-validation
- Factor importance analysis and company rankings
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.data_aggregator import DataAggregator
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.sector_factor_model import SectorFactorModel, compare_feature_importance_reduction
from src.preprocessing.sector_relative_target import SectorRelativeTarget
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import ModelVisualizer
from src.explainability.explanation_generator import ExplanationGenerator
from config.sector_config import get_sector_info
from src.utils.helpers import save_dataframe
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def main():
    print_section("Semiconductor Sector Analysis")

    # Get sector configuration
    sector_info = get_sector_info('semiconductors')
    companies = sector_info['companies']
    tickers = list(companies.keys())

    print(f"Analyzing {len(tickers)} semiconductor companies:")
    for ticker, info in companies.items():
        print(f"  - {ticker}: {info['name']} ({info['segment']})")

    # ==========================================
    # STEP 1: Data Collection
    # ==========================================
    print_section("Step 1: Data Collection")

    aggregator = DataAggregator()

    try:
        df = aggregator.collect_all_data(tickers=tickers, period='1y')
        print(f"[OK] Collected data: {len(df)} companies, {len(df.columns)} features")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return

    # ==========================================
    # STEP 2: Feature Engineering (Raw)
    # ==========================================
    print_section("Step 2: Feature Engineering")

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    print(f"[OK] Engineered features: {len(df.columns)} total features")

    # ==========================================
    # STEP 3: Factor Model (Dimensionality Reduction)
    # ==========================================
    print_section("Step 3: Factor Model - Dimensionality Reduction")

    print("BEFORE Factor Model:")
    print(f"  - Features: {len(df.columns) - 3}")  # Exclude ticker, company, sector
    print(f"  - Samples: {len(df)}")
    print(f"  - Ratio: {len(df) / (len(df.columns) - 3):.2f}:1 (POOR - need 10:1)")

    # Initialize factor model
    factor_model = SectorFactorModel(sector='semiconductors')

    # Show factor composition
    factor_composition = factor_model.get_factor_composition()
    print("\nFactor Model Composition:")
    print(factor_composition.to_string(index=False))

    # Compute factor scores
    factor_df = factor_model.fit_transform(df)
    print(f"\n[OK] Computed factor scores: {len(factor_df.columns) - 1} factors")

    # Compare efficiency
    comparison = compare_feature_importance_reduction(df, factor_model)
    print("\nAFTER Factor Model:")
    print(f"  - Factors: {comparison['n_factors']}")
    print(f"  - Samples: {comparison['n_samples']}")
    print(f"  - Ratio: {comparison['factor_sample_per_feature']:.2f}:1")
    print(f"  - Improvement: {comparison['efficiency_improvement']:.1f}x better")
    print(f"  - {comparison['recommendation']}")

    # ==========================================
    # STEP 4: Sector-Relative Target
    # ==========================================
    print_section("Step 4: Sector-Relative Target (vs Absolute Threshold)")

    target_generator = SectorRelativeTarget(sector='semiconductors')

    # Create sector-relative target
    target, target_meta = target_generator.create_target(df, return_column='returns_6m')
    factor_df['target'] = target

    print(f"Target Method: {target_meta['method']}")
    print(f"Threshold: {target_meta.get('threshold', 'N/A'):.4f}")
    print(f"Outperformers: {target_meta['n_outperformers']} ({target_meta['positive_class_pct']:.1%})")
    print(f"Underperformers: {target_meta['n_underperformers']}")

    # Validate target
    validation = target_generator.validate_target_distribution(target, target_meta)
    print(f"\nTarget Validation: {validation['recommendation']}")

    # ==========================================
    # STEP 5: Preprocessing (Scaling Only)
    # ==========================================
    print_section("Step 5: Preprocessing")

    # Save factor scores before preprocessing
    save_dataframe(factor_df, project_root / "data" / "processed" / "semiconductor_factors.parquet")

    # Prepare for modeling - only numeric factor columns
    factor_cols = [col for col in factor_df.columns if col not in ['ticker', 'company', 'sector', 'target']]

    preprocessor = DataPreprocessor()
    X = factor_df[factor_cols].copy()
    y = factor_df['target'].copy()

    X_processed = preprocessor.fit_transform(X)
    print(f"[OK] Preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")

    # ==========================================
    # STEP 6: Model Training
    # ==========================================
    print_section("Step 6: Model Training with Proper Sample Efficiency")

    trainer = ModelTrainer()

    # Prepare DataFrame for ModelTrainer
    X_processed_df = pd.DataFrame(X_processed, columns=factor_cols)
    X_processed_df['target'] = y.values

    # Prepare train/test split
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        X_processed_df,
        target_col='target',
        test_size=0.2
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train models (use simpler models given sample size)
    model_names = ['random_forest', 'xgboost']  # Skip logistic regression for now

    print(f"\nTraining {len(model_names)} models...")
    results = trainer.train_models(model_names=model_names)

    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Train Score: {metrics['train_score']:.4f}")
        print(f"  Test Score:  {metrics['test_score']:.4f}")
        if metrics['cv_mean'] is not None:
            print(f"  CV Score:    {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        else:
            print(f"  CV Score:    Not available")

    # Get best model
    best_name, best_model = trainer.get_best_model()
    print(f"\n[OK] Best model: {best_name}")

    # ==========================================
    # STEP 7: Factor Importance & Explainability
    # ==========================================
    print_section("Step 7: Factor Importance (Not Individual Features)")

    # Get feature importance from best model
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'factor': factor_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFactor Importance Rankings:")
        print(importances.to_string(index=False))
    else:
        print("Model does not support feature importances")

    # ==========================================
    # STEP 8: Company Rankings
    # ==========================================
    print_section("Step 8: Semiconductor Company Rankings")

    # Predict on all companies
    y_pred_proba = trainer.models[best_name].predict_proba(X_processed)[:, 1]

    # Create rankings DataFrame
    rankings = pd.DataFrame({
        'ticker': factor_df['ticker'].values,
        'company': factor_df['ticker'].map(lambda x: companies.get(x, {}).get('name', x)),
        'success_probability': y_pred_proba,
        'prediction': (y_pred_proba > 0.5).astype(int),
        'actual': y.values,
        'segment': factor_df['ticker'].map(lambda x: companies.get(x, {}).get('segment', ''))
    })

    # Sort by probability
    rankings = rankings.sort_values('success_probability', ascending=False)

    print("\nSemiconductor Company Success Rankings:")
    print(rankings.to_string(index=False))

    # Save rankings
    save_path = project_root / "outputs" / "reports" / "semiconductor_rankings.csv"
    save_dataframe(rankings, save_path)
    print(f"\n[OK] Saved rankings to {save_path}")

    # ==========================================
    # STEP 9: Factor Analysis for Top/Bottom Companies
    # ==========================================
    print_section("Step 9: Factor Analysis - Why Did Companies Rank This Way?")

    # Top company
    top_company = rankings.iloc[0]
    print(f"\nTOP PERFORMER: {top_company['company']} ({top_company['ticker']})")
    print(f"Success Probability: {top_company['success_probability']:.1%}")
    print(f"Segment: {top_company['segment']}")

    top_factors = factor_model.get_top_factors_for_company(top_company['ticker'], top_n=3)
    print("\nTop Strengths (Factors):")
    print(top_factors)

    # Bottom company
    bottom_company = rankings.iloc[-1]
    print(f"\nBOTTOM PERFORMER: {bottom_company['company']} ({bottom_company['ticker']})")
    print(f"Success Probability: {bottom_company['success_probability']:.1%}")
    print(f"Segment: {bottom_company['segment']}")

    bottom_factors = factor_model.get_top_factors_for_company(bottom_company['ticker'], top_n=3)
    print("\nTop Strengths (Factors):")
    print(bottom_factors)

    # ==========================================
    # STEP 10: Visualizations
    # ==========================================
    print_section("Step 10: Generating Visualizations")

    # Create output directory
    viz_dir = project_root / "outputs" / "visualizations" / "semiconductors"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = (y_pred_proba > 0.5).astype(int)
    y_test_pred = trainer.models[best_name].predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix ({best_name})', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_yticklabels(['Underperformer', 'Outperformer'])
    ax.set_xticklabels(['Underperformer', 'Outperformer'])
    plt.tight_layout()
    plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved confusion_matrix.png")

    # Model comparison - create simple visualization
    import matplotlib.pyplot as plt
    comparison_df = pd.DataFrame({
        'model': list(results.keys()),
        'train_score': [results[m]['train_score'] for m in results.keys()],
        'test_score': [results[m]['test_score'] for m in results.keys()],
        'cv_mean': [results[m]['cv_mean'] if results[m]['cv_mean'] is not None else 0 for m in results.keys()],
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.25

    ax.bar(x - width, comparison_df['train_score'], width, label='Train Score', alpha=0.8)
    ax.bar(x, comparison_df['test_score'], width, label='Test Score', alpha=0.8)
    ax.bar(x + width, comparison_df['cv_mean'], width, label='CV Score', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison - Semiconductor Sector', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved model_comparison.png")

    # Company ranking visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2ecc71' if p > 0.5 else '#e74c3c' for p in rankings['success_probability']]
    bars = ax.barh(range(len(rankings)), rankings['success_probability'], color=colors, alpha=0.7)

    ax.set_yticks(range(len(rankings)))
    ax.set_yticklabels(rankings['company'], fontsize=10)
    ax.set_xlabel('Success Probability', fontsize=12)
    ax.set_title('Semiconductor Company Success Probability Rankings', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add probability values
    for i, (idx, row) in enumerate(rankings.iterrows()):
        ax.text(row['success_probability'] + 0.02, i, f"{row['success_probability']:.2f}",
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(viz_dir / 'company_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved company_ranking.png")

    # ==========================================
    # Summary
    # ==========================================
    print_section("Demo Complete - Summary")

    print("Key Features:")
    print("  - Sector-specific analysis (semiconductors only)")
    print("  - Factor model reduces 74 features to 5 interpretable factors")
    print("  - Sector-relative targets for balanced classification")
    print("  - Sample-to-feature ratio: 2.8:1")
    print()
    print("Results:")
    print(f"  - Best Model: {best_name}")
    print(f"  - Test Score: {results[best_name]['test_score']:.4f}")
    if results[best_name]['cv_mean'] is not None:
        print(f"  - CV Score: {results[best_name]['cv_mean']:.4f} (+/- {results[best_name]['cv_std']:.4f})")
    print(f"  - Top Semiconductor: {top_company['company']} ({top_company['success_probability']:.1%})")
    print()
    print("Next Steps to Further Improve:")
    print("  1. Collect monthly time-series data (14 companies × 24 months = 336 samples)")
    print("  2. Implement walk-forward validation")
    print("  3. Add sector-specific features (fab utilization, R&D trends)")
    print("  4. Test on additional sectors (Cloud SaaS, Consumer Staples)")
    print()
    print(f"Outputs saved to:")
    print(f"  - Rankings: {save_path}")
    print(f"  - Visualizations: {viz_dir}")
    print(f"  - Factor Data: {project_root / 'data' / 'processed' / 'semiconductor_factors.parquet'}")


if __name__ == "__main__":
    main()
