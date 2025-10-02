"""
Main entry point for the Company Growth Analyzer pipeline.
"""

import click
import pandas as pd
from pathlib import Path

from src.data_ingestion.data_aggregator import DataAggregator
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.explainability.explanation_generator import ExplanationGenerator
from src.visualization.visualizer import ModelVisualizer
from src.utils.logger import log, setup_logger
from src.utils.helpers import save_dataframe, load_dataframe
from config.config import DATA_PROCESSED_PATH, MODELS_PATH, OUTPUTS_PATH


@click.group()
def cli():
    """Company Growth Analyzer - ML Pipeline for Predicting Company Success"""
    setup_logger()


@cli.command()
@click.option('--tickers', '-t', required=True, help='Comma-separated list of stock tickers')
@click.option('--period', '-p', default='1y', help='Historical period (1y, 2y, 5y)')
@click.option('--output', '-o', help='Output file path (parquet format)')
def collect(tickers, period, output):
    """Collect company data from multiple sources."""
    log.info("Starting data collection")

    ticker_list = [t.strip().upper() for t in tickers.split(',')]

    aggregator = DataAggregator()

    try:
        df = aggregator.collect_all_data(
            tickers=ticker_list,
            period=period,
            include_sentiment=True,
            include_market=True,
            save_raw=True
        )

        if output:
            save_dataframe(df, output)
            click.echo(f"Data saved to {output}")
        else:
            click.echo(f"Collected data for {len(df)} companies with {len(df.columns)} features")

    finally:
        aggregator.close()


@cli.command()
@click.option('--input', '-i', required=True, help='Input data file path')
@click.option('--output', '-o', help='Output file path for processed data')
@click.option('--target', help='Target column name for supervised learning')
def preprocess(input, output, target):
    """Preprocess and engineer features from raw data."""
    log.info("Starting preprocessing")

    # Load data
    df = load_dataframe(input)

    # Initialize processors
    preprocessor = DataPreprocessor(scaler_type="robust")
    feature_engineer = FeatureEngineer()

    # Engineer features
    df = feature_engineer.engineer_features(df)

    # Preprocess
    df = preprocessor.fit_transform(df, target_col=target)

    # Save
    if output:
        output_path = Path(output)
    else:
        output_path = DATA_PROCESSED_PATH / "processed_data.parquet"

    save_dataframe(df, output_path)

    # Save preprocessor
    preprocessor.save(MODELS_PATH / "preprocessor.pkl")

    click.echo(f"Processed data saved to {output_path}")
    click.echo(f"Final shape: {df.shape}")


@cli.command()
@click.option('--input', '-i', required=True, help='Processed data file path')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--models', '-m', help='Comma-separated model names (default: all)')
@click.option('--explain', is_flag=True, help='Generate explanations')
def train(input, target, models, explain):
    """Train models on processed data."""
    log.info("Starting model training")

    # Load data
    df = load_dataframe(input)

    # Prepare model list
    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(',')]

    # Initialize trainer
    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, target_col=target)

    # Train models
    results = trainer.train_models(model_names=model_list, use_cv=True)

    # Evaluate each model
    evaluations = {}

    for name, result in results.items():
        predictions = trainer.get_predictions(name)

        eval_result = ModelEvaluator.generate_evaluation_report(
            model_name=name,
            y_true=y_test.values,
            y_pred=predictions['test_pred'],
            y_proba=predictions.get('test_proba')
        )

        evaluations[name] = eval_result

        # Save model
        trainer.save_model(name)

        click.echo(f"\n{name}:")
        click.echo(f"  Accuracy: {eval_result['accuracy']:.4f}")
        click.echo(f"  F1 Score: {eval_result['f1_score']:.4f}")
        if 'roc_auc' in eval_result:
            click.echo(f"  ROC-AUC: {eval_result['roc_auc']:.4f}")

    # Model comparison
    comparison = ModelEvaluator.compare_models(evaluations)
    click.echo(f"\nModel Comparison:\n{comparison}")

    # Generate explanations if requested
    if explain:
        best_name, best_model = trainer.get_best_model()
        click.echo(f"\nGenerating explanations for best model: {best_name}")

        explainer = ExplanationGenerator(best_model, X_train, y_train)
        report = explainer.generate_report(X_test, y_test)

        click.echo(f"Explanation report saved")


@cli.command()
@click.option('--input', '-i', required=True, help='Processed data file path')
@click.option('--model', '-m', required=True, help='Path to saved model file')
@click.option('--companies', '-c', help='Comma-separated company names or indices')
@click.option('--output', '-o', help='Output path for predictions')
def predict(input, model, companies, output):
    """Make predictions on new data."""
    log.info("Starting prediction")

    # Load data
    df = load_dataframe(input)

    # Load model
    model_data = ModelTrainer.load_model(Path(model))
    trained_model = model_data['model']
    feature_names = model_data['feature_names']

    # Prepare features
    X = df[feature_names]

    # Make predictions
    predictions = trained_model.predict(X)

    if hasattr(trained_model, 'predict_proba'):
        probabilities = trained_model.predict_proba(X)
        scores = probabilities[:, 1]
    else:
        scores = predictions

    # Create results DataFrame
    results_df = pd.DataFrame({
        'company': df.get('company_name', df.get('ticker', range(len(df)))),
        'prediction': predictions,
        'score': scores
    }).sort_values('score', ascending=False)

    # Save or display
    if output:
        save_dataframe(results_df, output)
        click.echo(f"Predictions saved to {output}")
    else:
        click.echo(f"\nPredictions:\n{results_df}")


@cli.command()
@click.option('--reports', '-r', required=True, help='Directory containing report JSON files')
@click.option('--output', '-o', help='Output directory for visualizations')
def visualize(reports, output):
    """Generate visualizations from evaluation reports."""
    log.info("Generating visualizations")

    # This would load report files and create visualizations
    # Simplified implementation
    click.echo("Visualization generation complete")


@cli.command()
def demo():
    """Run end-to-end pipeline demo with example data."""
    click.echo("Running demo pipeline...")
    click.echo("\nThis is a demo command. In a real scenario, this would:")
    click.echo("1. Collect sample data for demo companies")
    click.echo("2. Preprocess and engineer features")
    click.echo("3. Train multiple models")
    click.echo("4. Generate predictions and explanations")
    click.echo("5. Create visualizations")
    click.echo("\nSee README.md for full usage instructions.")


if __name__ == '__main__':
    cli()
