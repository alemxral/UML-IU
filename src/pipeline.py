"""
Main pipeline for ArXiv Trends Analysis.
Orchestrates the entire workflow from data loading to report generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json

from data_loader import ArxivDataLoader
from preprocessing import TextPreprocessor, FeatureExtractor, extract_top_keywords
from clustering import ClusterAnalyzer, DimensionalityReducer
from visualization import Visualizer
from utils.helpers import ensure_dir, save_json

logger = logging.getLogger("arxiv_trends")


class ArxivTrendsPipeline:
    """Main pipeline for analyzing ArXiv trends."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_loader = ArxivDataLoader(data_dir=config['data']['data_dir'])
        self.text_preprocessor = None
        self.vectorizer = None
        self.dim_reducer = None
        self.clustering_analyzer = None
        self.visualizer = None
        
        # Data containers
        self.df = None
        self.df_processed = None
        self.tfidf_matrix = None
        self.reduced_features = None
        self.reduced_2d = None
        self.cluster_labels = None
        self.clustering_metrics = {}
        self.keywords = {}
        
        # Setup output directories
        self._setup_output_dirs()
        
    def _setup_output_dirs(self):
        """Create output directory structure."""
        ensure_dir(self.config['output']['output_dir'])
        ensure_dir(self.config['output']['plots_dir'])
        ensure_dir(self.config['output']['data_dir'])
        ensure_dir(self.config['output']['reports_dir'])
        
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 80)
        logger.info("STARTING ARXIV TRENDS ANALYSIS PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load data
        self.df = self.load_data()
        if self.df is None or len(self.df) == 0:
            logger.error("Failed to load data. Aborting pipeline.")
            return {}
        
        # Step 2: Exploratory Data Analysis
        eda_stats = self.run_eda()
        
        # Step 3: Preprocess text
        self.df_processed = self.preprocess_text()
        
        # Step 4: Vectorization
        self.tfidf_matrix = self.vectorize_text()
        
        # Step 5: Dimensionality Reduction
        self.reduced_features, self.reduced_2d = self.reduce_dimensions()
        
        # Step 6: Clustering
        self.cluster_labels, self.clustering_metrics = self.perform_clustering()
        
        # Step 7: Extract keywords
        self.keywords = self.extract_keywords()
        
        # Step 8: Analyze clusters
        cluster_analysis = self.analyze_clusters()
        
        # Step 9: Visualizations
        self.generate_visualizations()
        
        # Step 10: Export results
        results = self.export_results(eda_stats, cluster_analysis)
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return results
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare ArXiv dataset."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 80)
        
        # FORCE SMALL SAMPLE - NO KAGGLE DOWNLOAD
        sample_size = min(self.config['data']['sample_size'], 5000)
        logger.info(f"Using sample size: {sample_size} papers")
        logger.info("Creating synthetic dataset (no download required)")
        
        # Load using sample creation only
        df = self.data_loader.load_from_json(
            file_path=None,  # Will create sample dataset
            sample_size=sample_size,
            recent_years=self.config['data']['recent_years'],
            min_abstract_length=self.config['data']['min_abstract_length']
        )
        
        if df is None or len(df) == 0:
            raise ValueError("Failed to create sample dataset")
        
        # Prepare dataset
        df = self.data_loader.prepare_dataset(df)
        
        logger.info(f"✅ Created {len(df)} sample papers")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def run_eda(self) -> Dict[str, Any]:
        """Run exploratory data analysis."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 80)
        
        stats = self.data_loader.get_dataset_statistics(self.df)
        logger.info(f"Total papers: {stats['total_papers']:,}")
        logger.info(f"Unique categories: {stats['unique_categories']}")
        logger.info(f"Year range: {stats['year_range']}")
        logger.info(f"Avg abstract length: {stats['avg_abstract_length']:.0f} chars")
        
        # Initialize visualizer
        self.visualizer = Visualizer(
            output_dir=self.config['output']['plots_dir'],
            style=self.config['visualization']['style'],
            figsize=tuple(self.config['visualization']['figure_size']),
            dpi=self.config['visualization']['dpi']
        )
        
        # Generate EDA plots
        if 'categories' in self.df.columns:
            self.visualizer.plot_category_distribution(self.df, top_n=15)
        
        if 'year' in self.df.columns and 'categories' in self.df.columns:
            self.visualizer.plot_temporal_trends(self.df, top_categories=10)
        
        return stats
    
    def preprocess_text(self) -> pd.DataFrame:
        """Preprocess text data."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: TEXT PREPROCESSING")
        logger.info("=" * 80)
        
        self.text_preprocessor = TextPreprocessor(
            custom_stopwords=self.config['preprocessing'].get('custom_stopwords', []),
            use_lemmatization=self.config['preprocessing'].get('use_lemmatization', True)
        )
        
        # Download NLTK data
        self.text_preprocessor.download_nltk_data()
        
        # Preprocess abstracts
        logger.info("Cleaning and preprocessing abstracts...")
        df_processed = self.df.copy()
        df_processed['abstract_clean'] = self.text_preprocessor.preprocess_corpus(
            df_processed['abstract'].tolist()
        )
        
        # Remove empty abstracts after preprocessing
        df_processed = df_processed[df_processed['abstract_clean'].str.len() > 0].copy()
        logger.info(f"Papers after preprocessing: {len(df_processed)}")
        
        return df_processed
    
    def vectorize_text(self) -> np.ndarray:
        """Convert text to TF-IDF features."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: FEATURE EXTRACTION (TF-IDF)")
        logger.info("=" * 80)
        
        self.vectorizer = TFIDFVectorizer(
            max_features=self.config['preprocessing']['max_features'],
            ngram_range=tuple(self.config['preprocessing']['ngram_range']),
            min_df=self.config['preprocessing']['min_df'],
            max_df=self.config['preprocessing']['max_df']
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(
            self.df_processed['abstract_clean'].tolist()
        )
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Number of features: {len(self.vectorizer.get_feature_names())}")
        
        return tfidf_matrix
    
    def reduce_dimensions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce dimensionality for clustering and visualization."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: DIMENSIONALITY REDUCTION")
        logger.info("=" * 80)
        
        self.dim_reducer = DimensionalityReducer()
        
        # PCA for clustering
        logger.info("Applying PCA...")
        pca_features = self.dim_reducer.fit_transform_pca(
            self.tfidf_matrix,
            n_components=self.config['dimensionality_reduction']['pca']['n_components'],
            random_state=self.config['dimensionality_reduction']['pca']['random_state']
        )
        logger.info(f"PCA output shape: {pca_features.shape}")
        
        # UMAP for visualization
        logger.info("Applying UMAP for 2D visualization...")
        umap_features = self.dim_reducer.fit_transform_umap(
            pca_features,
            n_components=self.config['dimensionality_reduction']['umap']['n_components'],
            n_neighbors=self.config['dimensionality_reduction']['umap']['n_neighbors'],
            min_dist=self.config['dimensionality_reduction']['umap']['min_dist'],
            metric=self.config['dimensionality_reduction']['umap']['metric'],
            random_state=self.config['dimensionality_reduction']['umap']['random_state']
        )
        logger.info(f"UMAP output shape: {umap_features.shape}")
        
        return pca_features, umap_features
    
    def perform_clustering(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform clustering analysis."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: CLUSTERING")
        logger.info("=" * 80)
        
        self.clustering_analyzer = ClusterAnalyzer()
        
        # Determine optimal number of clusters
        n_clusters = self.config['clustering']['kmeans'].get('n_clusters')
        
        if n_clusters is None:
            logger.info("Determining optimal number of clusters...")
            cluster_range = self.config['clustering']['kmeans']['cluster_range']
            optimal_results = self.clustering_analyzer.find_optimal_k(
                self.reduced_features,
                k_range=range(cluster_range[0], cluster_range[1] + 1)
            )
            
            # Plot analysis
            self.visualizer.plot_elbow_silhouette_analysis(optimal_results)
            
            # Select optimal k based on method
            method = self.config['clustering']['optimal_k_method']
            if method == 'silhouette':
                n_clusters = optimal_results['silhouette_scores'].index(
                    max(optimal_results['silhouette_scores'])
                ) + cluster_range[0]
            else:
                n_clusters = 8  # Default
                
            logger.info(f"Optimal number of clusters: {n_clusters}")
        
        # Perform K-Means clustering
        logger.info(f"Performing K-Means clustering with k={n_clusters}...")
        labels, metrics = self.clustering_analyzer.kmeans_clustering(
            self.reduced_features,
            n_clusters=n_clusters,
            max_iter=self.config['clustering']['kmeans']['max_iter'],
            n_init=self.config['clustering']['kmeans']['n_init'],
            random_state=self.config['clustering']['kmeans']['random_state']
        )
        
        logger.info(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
        logger.info(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.3f}")
        logger.info(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.1f}")
        
        # Add cluster labels to dataframe
        self.df_processed['cluster'] = labels
        
        return labels, metrics
    
    def extract_keywords(self) -> Dict[int, list]:
        """Extract keywords for each cluster."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: KEYWORD EXTRACTION")
        logger.info("=" * 80)
        
        keywords_dict = extract_cluster_keywords(
            self.tfidf_matrix,
            self.cluster_labels,
            self.vectorizer.get_feature_names(),
            top_n=self.config['keywords']['top_n']
        )
        
        for cluster_id, keywords in keywords_dict.items():
            logger.info(f"Cluster {cluster_id}: {', '.join(keywords[:10])}")
        
        return keywords_dict
    
    def analyze_clusters(self) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: CLUSTER ANALYSIS")
        logger.info("=" * 80)
        
        cluster_stats = self.clustering_analyzer.get_cluster_statistics(
            self.df_processed,
            'cluster'
        )
        
        analysis = {
            'cluster_stats': cluster_stats,
            'cluster_sizes': self.df_processed['cluster'].value_counts().to_dict(),
            'keywords': self.keywords
        }
        
        # Analyze categories by cluster
        if 'categories' in self.df_processed.columns:
            category_cluster = pd.crosstab(
                self.df_processed['cluster'],
                self.df_processed['categories'],
                normalize='index'
            )
            analysis['category_distribution'] = category_cluster.to_dict()
        
        # Temporal analysis
        if 'year' in self.df_processed.columns:
            temporal_stats = self.df_processed.groupby('cluster')['year'].agg([
                'min', 'max', 'mean', 'count'
            ]).to_dict()
            analysis['temporal_stats'] = temporal_stats
        
        logger.info(f"Analyzed {len(cluster_stats)} clusters")
        
        return analysis
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: VISUALIZATION GENERATION")
        logger.info("=" * 80)
        
        # Cluster visualization
        self.visualizer.plot_clusters_2d(
            self.reduced_2d,
            self.cluster_labels,
            title="ArXiv Papers Clustering (UMAP 2D)"
        )
        
        # Cluster sizes
        self.visualizer.plot_cluster_sizes(self.df_processed, 'cluster')
        
        # Word clouds
        for cluster_id, keywords in self.keywords.items():
            self.visualizer.plot_wordcloud(
                keywords,
                title=f"Cluster {cluster_id} Keywords",
                filename=f"wordcloud_cluster_{cluster_id}.png"
            )
        
        # Keyword heatmap
        self.visualizer.plot_keyword_heatmap(self.keywords, top_n=15)
        
        # Category by cluster
        if 'categories' in self.df_processed.columns:
            self.visualizer.plot_category_by_cluster(self.df_processed, 'cluster', top_n=10)
        
        logger.info("All visualizations generated successfully")
    
    def export_results(self, eda_stats: Dict[str, Any], 
                      cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Export all results to files."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 10: EXPORTING RESULTS")
        logger.info("=" * 80)
        
        output_data_dir = Path(self.config['output']['data_dir'])
        
        # Export cluster assignments
        if self.config['output']['save_cluster_assignments']:
            assignments_path = output_data_dir / 'cluster_assignments.csv'
            self.df_processed[['id', 'title', 'cluster', 'categories', 'year']].to_csv(
                assignments_path, index=False
            )
            logger.info(f"Cluster assignments saved to {assignments_path}")
        
        # Export keywords
        if self.config['output']['save_keywords']:
            keywords_path = output_data_dir / 'cluster_keywords.json'
            save_json(self.keywords, str(keywords_path))
            logger.info(f"Keywords saved to {keywords_path}")
        
        # Export metrics
        if self.config['output']['save_metrics']:
            all_metrics = {
                'eda_stats': eda_stats,
                'clustering_metrics': self.clustering_metrics,
                'cluster_analysis': {
                    'cluster_sizes': cluster_analysis['cluster_sizes'],
                    'cluster_stats': cluster_analysis['cluster_stats']
                }
            }
            metrics_path = output_data_dir / 'analysis_metrics.json'
            save_json(all_metrics, str(metrics_path))
            logger.info(f"Metrics saved to {metrics_path}")
        
        # Export data for LaTeX report
        if self.config['output']['export_latex_data']:
            latex_data = self._prepare_latex_data(eda_stats, cluster_analysis)
            latex_path = output_data_dir / 'latex_report_data.json'
            save_json(latex_data, str(latex_path))
            logger.info(f"LaTeX report data saved to {latex_path}")
        
        logger.info("All results exported successfully")
        
        return {
            'eda_stats': eda_stats,
            'clustering_metrics': self.clustering_metrics,
            'cluster_analysis': cluster_analysis,
            'n_papers': len(self.df_processed),
            'n_clusters': len(self.keywords)
        }
    
    def _prepare_latex_data(self, eda_stats: Dict[str, Any], 
                           cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data structure for LaTeX report generation."""
        latex_data = {
            'dataset': {
                'total_papers': eda_stats['total_papers'],
                'unique_categories': eda_stats['unique_categories'],
                'year_range': eda_stats['year_range'],
                'avg_abstract_length': eda_stats['avg_abstract_length']
            },
            'clustering': {
                'n_clusters': len(self.keywords),
                'silhouette_score': self.clustering_metrics.get('silhouette_score', 0),
                'davies_bouldin_index': self.clustering_metrics.get('davies_bouldin_index', 0),
                'calinski_harabasz_index': self.clustering_metrics.get('calinski_harabasz_index', 0)
            },
            'clusters': []
        }
        
        # Add cluster details
        for cluster_id in sorted(self.keywords.keys()):
            cluster_size = cluster_analysis['cluster_sizes'].get(cluster_id, 0)
            cluster_pct = (cluster_size / eda_stats['total_papers']) * 100
            
            cluster_info = {
                'id': cluster_id,
                'size': cluster_size,
                'percentage': cluster_pct,
                'keywords': self.keywords[cluster_id][:20],
                'top_keywords': ', '.join(self.keywords[cluster_id][:5])
            }
            
            latex_data['clusters'].append(cluster_info)
        
        return latex_data
