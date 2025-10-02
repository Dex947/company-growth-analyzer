"""
Enhanced semiconductor sector analysis with temporal data expansion.

This demo addresses overfitting by expanding the dataset from:
- OLD: 14 companies × 1 snapshot = 14 samples
- NEW: 50+ companies × 12 quarters = 600+ samples

This increases the samples-to-features ratio from 2.8:1 to 120:1,
meeting the 10:1 minimum threshold for reliable machine learning.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion.simple_temporal_collector import SimpleTemporalCollector
from src.preprocessing.sector_factor_model import SectorFactorModel
from src.preprocessing.sector_relative_target import SectorRelativeTarget
from src.models.model_trainer import ModelTrainer
from src.utils.logger import log
from config.sector_config import SEMICONDUCTOR_COMPANIES
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Run enhanced temporal semiconductor analysis."""
    log.info("=" * 80)
    print("SEMICONDUCTOR SECTOR GROWTH ANALYSIS - TEMPORAL EXPANSION")
    log.info("=" * 80)

    # Step 1: Data Collection (Temporal)
    print("\n[STEP 1] Collecting Temporal Financial Data")
    print("-" * 80)

    tickers = list(SEMICONDUCTOR_COMPANIES.keys())
    print(f"Companies to analyze: {len(tickers)}")
    print(f"Quarters to collect: 12 (3 years of quarterly data)")
    print(f"Expected samples: ~{len(tickers) * 12} (company × quarter combinations)")
    # print()

    collector = SimpleTemporalCollector(quarters_back=12)

    try:
        df_temporal = collector.collect(tickers)
        log.info(f" Collected {len(df_temporal)} temporal samples")
        print(f"     Actual ratio: {len(df_temporal)} samples")
    except Exception as e:
        log.error(f" Error collecting temporal data: {e}")
        import traceback
        traceback.print_exc()
        return

    if df_temporal.empty:
        print("[X] No data collected. Exiting.")
        return

    # Display sample
    print(f"\nSample data (first 5 rows):")
    display_cols = ['ticker', 'quarter', 'price', 'returns_6m', 'volatility']
    print(df_temporal[display_cols].head())

    # Step 2: Feature Engineering (Factor Model)
    print("\n[STEP 2] Computing Factor Scores")
    print("-" * 80)

    factor_model = SectorFactorModel(sector='semiconductors')

    try:
        factor_df = factor_model.fit_transform(df_temporal)
        log.info(f" Computed {len(factor_df.columns)} factor scores")
        print(f"     Factors: {list(factor_df.columns)}")

        # Merge factors back
        df = pd.concat([df_temporal.reset_index(drop=True), factor_df], axis=1)

        # Select only numeric factor columns for training (exclude 'ticker')
        factor_cols = [col for col in factor_df.columns if col != 'ticker']
        samples_to_features = len(df) / len(factor_cols)
        print(f"     Feature columns: {factor_cols}")
        print(f"     Samples-to-features ratio: {samples_to_features:.1f}:1")

        if samples_to_features < 10:
            print(f"     [WARNING] Ratio below 10:1 threshold")
        else:
            print(f"     [OK] Ratio exceeds 10:1 minimum for reliable ML")

    except Exception as e:
        log.error(f" Error computing factors: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Target Variable (Sector-Relative)
    print("\n[STEP 3] Creating Sector-Relative Target Variable")
    print("-" * 80)

    target_generator = SectorRelativeTarget(sector='semiconductors')

    try:
        target, target_meta = target_generator.create_target(df, return_column='returns_6m')

        # Handle missing values (use only numeric factor columns)
        valid_mask = target.notna() & df[factor_cols].notna().all(axis=1)
        df_clean = df[valid_mask].copy()
        target_clean = target[valid_mask]
        factor_clean = df_clean[factor_cols]

        log.info(f" Target variable created")
        print(f"     Success rate (target=1): {target_clean.sum() / len(target_clean) * 100:.1f}%")
        print(f"     Samples after cleaning: {len(df_clean)}")
        print(f"     Sector median return: {target_meta.get('sector_median', 0):.2%}")
        print(f"     Sector volatility: {target_meta.get('sector_volatility', 0):.2%}")

        class_balance = target_clean.value_counts()
        print(f"     Class distribution: {dict(class_balance)}")

    except Exception as e:
        log.error(f" Error creating target: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Train-Test Split
    print("\n[STEP 4] Splitting Data (Temporal Awareness)")
    print("-" * 80)

    # Sort by date to ensure temporal split
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    target_clean = target_clean.loc[df_clean.index].reset_index(drop=True)
    factor_clean = factor_clean.loc[df_clean.index].reset_index(drop=True)

    # Use most recent 20% as test set (temporal split)
    split_idx = int(len(df_clean) * 0.8)
    train_idx = range(split_idx)
    test_idx = range(split_idx, len(df_clean))

    X_train = factor_clean.iloc[train_idx]
    X_test = factor_clean.iloc[test_idx]
    y_train = target_clean.iloc[train_idx]
    y_test = target_clean.iloc[test_idx]

    log.info(f" Data split complete")
    print(f"     Training samples: {len(X_train)}")
    print(f"     Test samples: {len(X_test)}")
    print(f"     Train success rate: {y_train.sum() / len(y_train) * 100:.1f}%")
    print(f"     Test success rate: {y_test.sum() / len(y_test) * 100:.1f}%")

    # Step 5: Model Training
    print("\n[STEP 5] Training Machine Learning Models")
    print("-" * 80)

    trainer = ModelTrainer()

    # Set the data in trainer
    trainer.X_train = X_train
    trainer.X_test = X_test
    trainer.y_train = y_train
    trainer.y_test = y_test
    trainer.feature_names = X_train.columns.tolist()

    try:
        results = trainer.train_models(
            model_names=['random_forest', 'xgboost', 'lightgbm'],
            use_cv=True
        )

        log.info(f" Models trained successfully")
        # print()

        for model_name, metrics in results.items():
            print(f"{model_name.upper()}:")
            print(f"  Train Score: {metrics.get('train_score', 0):.3f}")
            print(f"  Test Score:  {metrics.get('test_score', 0):.3f}")
            print(f"  CV Mean:     {metrics.get('cv_mean', 0):.3f} ± {metrics.get('cv_std', 0):.3f}")
            # print()

    except Exception as e:
        log.error(f" Error training models: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 6: Predictions
    print("\n[STEP 6] Generating Predictions")
    print("-" * 80)

    try:
        # Use best model (lowest generalization gap)
        best_model_name = min(results.keys(), key=lambda k: abs(results[k]['train_score'] - results[k]['test_score']))
        best_model = results[best_model_name]['model']

        print(f"Best model (smallest overfitting gap): {best_model_name}")

        # Predict on most recent data for each company
        # Add ticker back for grouping
        df_clean_with_ticker = df_clean.copy()
        df_clean_with_ticker['ticker_col'] = df_temporal.loc[df_clean.index, 'ticker'].values

        latest_data = df_clean_with_ticker.groupby('ticker_col').last().reset_index()
        latest_data = latest_data.rename(columns={'ticker_col': 'ticker'})
        latest_factors = latest_data[factor_cols]

        predictions = best_model.predict_proba(latest_factors)[:, 1]
        latest_data['success_probability'] = predictions

        # Sort by prediction
        rankings = latest_data[['ticker', 'quarter', 'success_probability']].sort_values(
            'success_probability', ascending=False
        )

        log.info(f" Generated predictions for {len(rankings)} companies")
        print("\nTop 10 Companies (Highest Success Probability):")
        print(rankings.head(10).to_string(index=False))

    except Exception as e:
        log.error(f" Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 7: Save Results
    print("\n[STEP 7] Saving Results")
    print("-" * 80)

    try:
        # Create output directory
        output_dir = Path('outputs/reports')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save rankings
        rankings_file = output_dir / 'temporal_semiconductor_rankings.csv'
        latest_data.to_csv(rankings_file, index=False)
        log.info(f" Rankings saved to: {rankings_file}")

        # Save full temporal dataset
        temporal_file = output_dir / 'temporal_dataset.parquet'
        df_clean.to_parquet(temporal_file, index=False)
        log.info(f" Temporal dataset saved to: {temporal_file}")

    except Exception as e:
        log.error(f" Error saving results: {e}")

    # Step 8: Visualizations
    print("\n[STEP 8] Creating Visualizations")
    print("-" * 80)

    viz_dir = Path('outputs/visualizations/temporal_semiconductors')
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Model Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = list(results.keys())
        train_scores = [results[m]['train_score'] for m in model_names]
        test_scores = [results[m]['test_score'] for m in model_names]
        cv_scores = [results[m]['cv_mean'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        ax.bar(x - width, train_scores, width, label='Train', alpha=0.8)
        ax.bar(x, test_scores, width, label='Test', alpha=0.8)
        ax.bar(x + width, cv_scores, width, label='CV', alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison (Temporal Data)')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names])
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_comparison.png', dpi=150)
        plt.close()

        log.info(f" Model comparison saved")

        # 2. Company Rankings
        fig, ax = plt.subplots(figsize=(12, 8))
        top_companies = rankings.head(15)
        y_pos = np.arange(len(top_companies))

        # Get ticker from latest_data DataFrame to avoid duplicate column issue
        ticker_values = top_companies['ticker'].values
        if len(ticker_values.shape) > 1:
            ticker_values = ticker_values[:, 0]  # Take first column if 2D

        ax.barh(y_pos, top_companies['success_probability'], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ticker_values)
        ax.invert_yaxis()
        ax.set_xlabel('Success Probability')
        ax.set_title('Top 15 Semiconductor Companies - Success Probability')
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(viz_dir / 'company_ranking.png', dpi=150)
        plt.close()

        log.info(f" Company ranking saved")

        # 3. Learning Curve (showing overfitting reduction)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sample sizes for learning curve
        sample_fractions = np.linspace(0.1, 1.0, 10)
        train_scores_lc = []
        test_scores_lc = []

        for frac in sample_fractions:
            n_samples = int(len(X_train) * frac)
            X_subset = X_train.iloc[:n_samples]
            y_subset = y_train.iloc[:n_samples]

            model = results[best_model_name]['model'].__class__()
            model.fit(X_subset, y_subset)

            train_scores_lc.append(model.score(X_subset, y_subset))
            test_scores_lc.append(model.score(X_test, y_test))

        ax.plot(sample_fractions * len(X_train), train_scores_lc, 'o-', label='Train', alpha=0.8)
        ax.plot(sample_fractions * len(X_train), test_scores_lc, 's-', label='Test', alpha=0.8)
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve - Overfitting Analysis')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'learning_curve.png', dpi=150)
        plt.close()

        log.info(f" Learning curve saved")

        print(f"\nAll visualizations saved to: {viz_dir}")

    except Exception as e:
        log.error(f" Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    log.info("=" * 80)
    print(f"Total samples collected: {len(df_clean)}")
    print(f"Companies analyzed: {df_clean['ticker'].nunique()}")
    print(f"Time periods: {df_clean['quarter'].nunique()} quarters")
    print(f"Samples-to-features ratio: {len(df_clean) / len(factor_df.columns):.1f}:1")
    # print()
    print(f"Best model: {best_model_name}")
    print(f"  Generalization gap: {abs(results[best_model_name]['train_score'] - results[best_model_name]['test_score']):.3f}")
    print(f"  CV stability: {results[best_model_name]['cv_std']:.3f}")
    # print()

    if results[best_model_name]['cv_std'] < 0.15:
        print("[OK] Model shows good stability (CV std < 0.15)")
    else:
        print("[WARNING] Model shows instability (CV std >= 0.15)")

    if abs(results[best_model_name]['train_score'] - results[best_model_name]['test_score']) < 0.1:
        print("[OK] Model generalizes well (gap < 0.1)")
    else:
        print("[WARNING] Model may be overfitting (gap >= 0.1)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
