#!/usr/bin/env python3
"""
Main entry point for ArXiv Trends Analysis.
Run the complete pipeline from command line.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ArxivTrendsPipeline
from utils.logging_config import setup_logging
from utils.helpers import load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ArXiv Trends Analysis Pipeline - Categorizing Scientific Trends',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python main.py
  
  # Run with custom config file
  python main.py --config my_config.yaml
  
  # Run with specific sample size
  python main.py --sample-size 10000
  
  # Run with verbose logging
  python main.py --verbose
  
  # Run without downloading data (use existing)
  python main.py --no-download
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/config.yaml',
        help='Path to configuration YAML file (default: src/config/config.yaml)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of papers to sample (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        help='Number of clusters (overrides automatic detection)'
    )
    
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip automatic data download, use existing data'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal logging output (WARNING level only)'
    )
    
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization generation (faster execution)'
    )
    
    return parser.parse_args()


def override_config(config, args):
    """
    Override config with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Parsed arguments
        
    Returns:
        Updated configuration
    """
    if args.sample_size:
        config['data']['sample_size'] = args.sample_size
    
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
        config['output']['plots_dir'] = f"{args.output_dir}/plots/"
        config['output']['data_dir'] = f"{args.output_dir}/data/"
        config['output']['reports_dir'] = f"{args.output_dir}/reports/"
    
    if args.n_clusters:
        config['clustering']['kmeans']['n_clusters'] = args.n_clusters
    
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    elif args.quiet:
        config['logging']['level'] = 'WARNING'
    
    return config


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'data': {'data_dir': 'data/', 'sample_size': 50000, 'recent_years': 3, 'min_abstract_length': 100, 'random_seed': 42},
            'preprocessing': {'max_features': 5000, 'ngram_range': [1, 2], 'min_df': 5, 'max_df': 0.8, 'use_lemmatization': True, 'custom_stopwords': []},
            'dimensionality_reduction': {
                'pca': {'n_components': 50, 'random_state': 42},
                'umap': {'n_components': 2, 'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'cosine', 'random_state': 42}
            },
            'clustering': {
                'kmeans': {'n_clusters': None, 'cluster_range': [4, 12], 'max_iter': 300, 'n_init': 10, 'random_state': 42},
                'optimal_k_method': 'silhouette'
            },
            'keywords': {'top_n': 20, 'min_term_freq': 3},
            'visualization': {'figure_size': [12, 8], 'dpi': 300, 'style': 'seaborn-v0_8-darkgrid', 'save_format': 'png'},
            'output': {'output_dir': 'output/', 'plots_dir': 'output/plots/', 'data_dir': 'output/data/', 'reports_dir': 'output/reports/', 'save_cluster_assignments': True, 'save_keywords': True, 'save_metrics': True, 'export_latex_data': True},
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 'log_file': 'output/pipeline.log', 'console_output': True},
            'pipeline': {'skip_if_exists': False, 'save_intermediate': True, 'verbose': True}
        }
    
    # Override with command line arguments
    config = override_config(config, args)
    
    # Setup logging
    logger = setup_logging(
        log_level=config['logging']['level'],
        log_file=config['logging']['log_file'],
        log_format=config['logging']['format'],
        console_output=config['logging']['console_output']
    )
    
    logger.info("ArXiv Trends Analysis - Categorizing Trends in Science")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Sample size: {config['data']['sample_size']}")
    logger.info(f"Output directory: {config['output']['output_dir']}")
    
    try:
        # Initialize and run pipeline
        pipeline = ArxivTrendsPipeline(config)
        
        # Skip visualization if requested
        if args.skip_viz:
            logger.info("Skipping visualization generation (--skip-viz flag)")
            # We'll modify this in the pipeline run
        
        # Run the full pipeline
        results = pipeline.run_full_pipeline()
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total papers analyzed: {results.get('n_papers', 0):,}")
        logger.info(f"Number of clusters: {results.get('n_clusters', 0)}")
        logger.info(f"Silhouette score: {results.get('clustering_metrics', {}).get('silhouette_score', 0):.3f}")
        logger.info(f"Output directory: {config['output']['output_dir']}")
        logger.info("=" * 80)
        
        logger.info("\nAnalysis completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Review visualizations in: output/plots/")
        logger.info("2. Check cluster assignments in: output/data/cluster_assignments.csv")
        logger.info("3. Review keywords in: output/data/cluster_keywords.json")
        logger.info("4. Generate LaTeX report with: python src/report_generator.py")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n\nPipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
