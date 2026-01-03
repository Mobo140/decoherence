"""Generate final report with comparison table of all models."""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logging


def load_metrics_from_file(metrics_path: Path) -> Dict:
    """Load metrics from JSON file.
    
    Args:
        metrics_path: Path to metrics JSON file
        
    Returns:
        Dictionary with metrics
    """
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_cv_results(cv_results_path: Path) -> Dict:
    """Load CV results from JSON file.
    
    Args:
        cv_results_path: Path to CV results JSON file
        
    Returns:
        Dictionary with CV results
    """
    with open(cv_results_path, 'r') as f:
        return json.load(f)


def format_metric_value(mean: float, std: float, precision: int = 4) -> str:
    """Format metric as mean ± std.
    
    Args:
        mean: Mean value
        std: Standard deviation
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def generate_metrics_table(
    results: Dict,
    output_dir: Path,
    primary_metric: str = 'mae'
) -> pd.DataFrame:
    """Generate metrics comparison table.
    
    Args:
        results: Dictionary with results for each model
        output_dir: Output directory
        primary_metric: Primary metric for model selection
        
    Returns:
        DataFrame with metrics
    """
    rows = []
    
    for model_type, model_results in results.items():
        row = {'Model': model_type.upper()}
        
        # Check if we have CV results
        if 'cv' in model_results and 'aggregated_metrics' in model_results['cv']:
            agg_metrics = model_results['cv']['aggregated_metrics']
            
            for metric_name in ['mae', 'rmse', 'r2', 'coverage']:
                if metric_name in agg_metrics:
                    mean = agg_metrics[metric_name]['mean']
                    std = agg_metrics[metric_name]['std']
                    row[metric_name.upper()] = format_metric_value(mean, std)
                else:
                    row[metric_name.upper()] = 'N/A'
        
        # Check if we have holdout results
        elif 'holdout' in model_results:
            # Try to load metrics from holdout experiment
            # This would need to be implemented based on actual structure
            for metric_name in ['mae', 'rmse', 'r2', 'coverage']:
                row[metric_name.upper()] = 'N/A'  # Placeholder
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_dir / 'metrics_summary.csv'
    df.to_csv(csv_path, index=False)
    
    return df


def generate_latex_table(df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table from DataFrame.
    
    Args:
        df: DataFrame with metrics
        output_path: Path to save LaTeX file
    """
    # Start LaTeX table
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}",
        "\\toprule"
    ]
    
    # Header
    header = " & ".join(df.columns)
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    
    # Rows
    for _, row in df.iterrows():
        row_str = " & ".join(str(val) for val in row.values)
        lines.append(row_str + " \\\\")
    
    # End table
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Сравнение моделей по метрикам (среднее $\\pm$ стандартное отклонение по фолдам кросс-валидации).}",
        "\\label{tab:model_comparison}",
        "\\end{table}"
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_markdown_report(
    results: Dict,
    config_path: str,
    output_path: Path,
    primary_metric: str = 'mae'
):
    """Generate markdown report.
    
    Args:
        results: Dictionary with results
        config_path: Path to config file
        output_path: Path to save markdown report
        primary_metric: Primary metric for model selection
    """
    lines = [
        "# Отчёт о сравнении моделей",
        "",
        f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Конфигурация:** {config_path}",
        "",
        "## Методы валидации",
        "",
        "### Кросс-валидация",
        "",
        "Использована K-fold кросс-валидация для оценки моделей.",
        "",
        "## Результаты",
        ""
    ]
    
    # Find best model
    best_model = None
    best_score = float('inf')
    
    for model_type, model_results in results.items():
        if 'cv' in model_results and 'aggregated_metrics' in model_results['cv']:
            agg_metrics = model_results['cv']['aggregated_metrics']
            if primary_metric in agg_metrics:
                score = agg_metrics[primary_metric]['mean']
                if score < best_score:
                    best_score = score
                    best_model = model_type
    
    if best_model:
        lines.extend([
            f"**Лучшая модель:** {best_model.upper()} (по метрике {primary_metric.upper()}: {best_score:.6f})",
            ""
        ])
    
    # Add table
    lines.extend([
        "## Таблица метрик",
        "",
        "| Модель | MAE | RMSE | R² | Coverage |",
        "|--------|-----|------|----|----------|"
    ])
    
    for model_type, model_results in results.items():
        if 'cv' in model_results and 'aggregated_metrics' in model_results['cv']:
            agg_metrics = model_results['cv']['aggregated_metrics']
            mae = format_metric_value(
                agg_metrics.get('mae', {}).get('mean', 0),
                agg_metrics.get('mae', {}).get('std', 0)
            )
            rmse = format_metric_value(
                agg_metrics.get('rmse', {}).get('mean', 0),
                agg_metrics.get('rmse', {}).get('std', 0)
            )
            r2 = format_metric_value(
                agg_metrics.get('r2', {}).get('mean', 0),
                agg_metrics.get('r2', {}).get('std', 0)
            )
            coverage = format_metric_value(
                agg_metrics.get('coverage', {}).get('mean', 0),
                agg_metrics.get('coverage', {}).get('std', 0)
            )
            lines.append(f"| {model_type.upper()} | {mae} | {rmse} | {r2} | {coverage} |")
    
    lines.extend([
        "",
        "## Финальная оценка на тестовой выборке",
        "",
        "Тестовая выборка использовалась только один раз для финальной оценки выбранной модели.",
        ""
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    """Main function for generating report."""
    parser = argparse.ArgumentParser(description='Generate comparison report')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Directory with experiment results')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output directory for reports')
    parser.add_argument('--primary-metric', type=str, default='mae',
                        choices=['mae', 'rmse', 'r2'],
                        help='Primary metric for model selection')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"Generating report from: {args.run_dir}")
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        # Try to provide helpful error message
        # Check both experiments and runs/experiments
        possible_dirs = []
        if Path('experiments').exists():
            # Check if experiments/ has model subdirs (old structure) or exp_* (new structure)
            exp_dir = Path('experiments')
            for item in exp_dir.iterdir():
                if item.is_dir():
                    if item.name in ['mlp', 'gru', 'lstm', 'transformer', 'two_stage']:
                        # Old structure: experiments/<model>/
                        possible_dirs.append(exp_dir)
                        break
                    elif item.name.startswith('exp_'):
                        # New structure: experiments/exp_*/
                        possible_dirs.append(item)
        
        if Path('runs/experiments').exists():
            possible_dirs.extend(list(Path('runs/experiments').glob('exp_*')))
        
        error_msg = f"Run directory does not exist: {run_dir}"
        if possible_dirs:
            error_msg += f"\n\nAvailable experiments:\n"
            for d in sorted(set(possible_dirs))[:10]:  # Show up to 10
                error_msg += f"  - {d}\n"
            error_msg += "\nUse one of these paths"
        else:
            error_msg += "\n\nNo experiments found. Please run experiments first:\n"
            error_msg += "  python3 -m src.train.run_suite --config configs/experiments/E1.yaml --models mlp --method cv"
        raise ValueError(error_msg)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load experiment results
    # Support two structures:
    # 1. experiments/<model>/ (old structure from first TZ)
    # 2. experiments/exp_*/<model>/ (new structure)
    
    # Try to load experiment config
    exp_config_path = run_dir / 'experiment_config.json'
    if exp_config_path.exists():
        with open(exp_config_path, 'r') as f:
            exp_config = json.load(f)
        models = exp_config.get('models', [])
    else:
        # Try to infer models from directory structure
        models = []
        
        # Check if this is experiments/ directly (old structure)
        if run_dir.name == 'experiments':
            # Old structure: experiments/<model>/
            for subdir in run_dir.iterdir():
                if subdir.is_dir() and subdir.name in ['mlp', 'gru', 'lstm', 'transformer', 'two_stage']:
                    models.append(subdir.name)
        else:
            # New structure: experiments/exp_*/<model>/
            for subdir in run_dir.iterdir():
                if subdir.is_dir() and subdir.name in ['mlp', 'gru', 'lstm', 'transformer', 'two_stage']:
                    models.append(subdir.name)
    
    if not models:
        raise ValueError("No models found in run directory")
    
    logger.info(f"Found models: {models}")
    
    # Load results for each model
    results = {}
    
    for model_type in models:
        # Support both structures
        if run_dir.name == 'experiments':
            # Old structure: experiments/<model>/
            model_dir = run_dir / model_type
        else:
            # New structure: experiments/exp_*/<model>/
            model_dir = run_dir / model_type
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            continue
        
        model_results = {}
        
        # Try to load CV results
        # Check both possible locations:
        # 1. model_dir/cv_results.json (new structure after fix)
        # 2. model_dir/model_type/cv_results.json (old structure with double nesting)
        cv_results_path = model_dir / 'cv_results.json'
        if not cv_results_path.exists():
            # Try old structure with double nesting
            cv_results_path = model_dir / model_type / 'cv_results.json'
        
        if cv_results_path.exists():
            model_results['cv'] = load_cv_results(cv_results_path)
            logger.info(f"Loaded CV results for {model_type} from {cv_results_path}")
        else:
            # Try to load from subdirectories (for CV folds structure)
            # Check if there are fold directories
            fold_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')]
            if fold_dirs:
                logger.info(f"Found {len(fold_dirs)} fold directories for {model_type}, but cv_results.json not found")
                logger.warning(f"CV results not yet aggregated for {model_type}. Run CV first or wait for completion.")
        
        # Try to load holdout results (check for metrics.json or similar)
        metrics_path = model_dir / 'metrics.json'
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                if 'metrics' in metrics_data:
                    model_results['holdout'] = {
                        'metrics': metrics_data['metrics'],
                        'checkpoint': metrics_data.get('checkpoint', '')
                    }
                    logger.info(f"Loaded holdout results for {model_type}")
            except Exception as e:
                logger.warning(f"Could not load holdout results from {metrics_path}: {e}")
        
        if model_results:
            results[model_type] = model_results
    
    if not results:
        # Provide more helpful error message
        error_msg = f"No results found in run directory: {run_dir}\n\n"
        error_msg += f"Found model directories: {models}\n"
        error_msg += f"Expected structure:\n"
        error_msg += f"  {run_dir}/<model>/cv_results.json (for CV)\n"
        error_msg += f"  {run_dir}/<model>/metrics.json (for holdout)\n"
        error_msg += f"\nTo generate results, run:\n"
        error_msg += f"  python3 -m src.train.run_suite --config configs/experiments/E1.yaml \\\n"
        error_msg += f"      --models {' '.join(models[:3])} --method cv --n-folds 5\n"
        error_msg += f"\nOr for holdout validation:\n"
        error_msg += f"  python3 -m src.train.train --config configs/experiments/E1.yaml --model <model_type>\n"
        raise ValueError(error_msg)
    
    # Generate metrics table
    df = generate_metrics_table(results, output_dir, args.primary_metric)
    logger.info(f"Metrics table saved to {output_dir / 'metrics_summary.csv'}")
    
    # Generate LaTeX table
    latex_path = output_dir / 'metrics_summary.tex'
    generate_latex_table(df, latex_path)
    logger.info(f"LaTeX table saved to {latex_path}")
    
    # Generate markdown report
    config_path = args.config or (run_dir / 'experiment_config.json')
    markdown_path = output_dir / 'report.md'
    generate_markdown_report(results, str(config_path), markdown_path, args.primary_metric)
    logger.info(f"Markdown report saved to {markdown_path}")
    
    logger.info(f"\nReport generation completed!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
