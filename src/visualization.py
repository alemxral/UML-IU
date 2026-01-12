"""
Visualization module for scientific paper analysis.
Provides plotting functions for EDA and clustering results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Comprehensive visualization for paper analysis."""
    
    def __init__(self, output_dir='../output'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        
    def plot_category_distribution(self, df: pd.DataFrame, top_n=20, 
                                   save_path=None):
        """
        Plot distribution of paper categories.
        
        Args:
            df: DataFrame with 'categories' column
            top_n: Number of top categories to show
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Count categories
        category_counts = df['categories'].value_counts().head(top_n)
        
        # Create bar plot
        category_counts.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Number of Papers', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        ax.set_title(f'Top {top_n} ArXiv Categories by Paper Count', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(category_counts):
            ax.text(v + 10, i, str(v), va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_temporal_trends(self, df: pd.DataFrame, save_path=None):
        """
        Plot temporal trends in publications.
        
        Args:
            df: DataFrame with 'year' column
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Count papers per year
        year_counts = df['year'].value_counts().sort_index()
        
        # Create line plot
        ax.plot(year_counts.index, year_counts.values, 
               marker='o', linewidth=2, markersize=6, color='darkblue')
        ax.fill_between(year_counts.index, year_counts.values, 
                        alpha=0.3, color='skyblue')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title('Publication Trends Over Time', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_elbow_analysis(self, metrics_dict: Dict, save_path=None):
        """
        Plot elbow curve and silhouette scores for K selection.
        
        Args:
            metrics_dict: Dictionary with clustering metrics
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        k_values = metrics_dict['k_values']
        
        # Elbow curve (Inertia)
        axes[0, 0].plot(k_values, metrics_dict['inertia'], 
                       marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (K)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method - Inertia')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette Score
        axes[0, 1].plot(k_values, metrics_dict['silhouette'], 
                       marker='s', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Number of Clusters (K)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score by K (Higher is Better)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Davies-Bouldin Index
        axes[1, 0].plot(k_values, metrics_dict['davies_bouldin'], 
                       marker='^', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Number of Clusters (K)')
        axes[1, 0].set_ylabel('Davies-Bouldin Index')
        axes[1, 0].set_title('Davies-Bouldin Index by K (Lower is Better)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz Score
        axes[1, 1].plot(k_values, metrics_dict['calinski_harabasz'], 
                       marker='d', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('Number of Clusters (K)')
        axes[1, 1].set_ylabel('Calinski-Harabasz Score')
        axes[1, 1].set_title('Calinski-Harabasz Score by K (Higher is Better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_cluster_distribution(self, labels: np.ndarray, 
                                  method_name='K-Means', save_path=None):
        """
        Plot distribution of samples across clusters.
        
        Args:
            labels: Cluster labels
            method_name: Name of clustering method
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Count samples per cluster
        unique, counts = np.unique(labels, return_counts=True)
        
        # Create bar plot
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))
        bars = ax.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title(f'Cluster Size Distribution - {method_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_clusters_2d(self, embedding: np.ndarray, labels: np.ndarray, 
                        method_name='UMAP', save_path=None):
        """
        Plot 2D visualization of clusters.
        
        Args:
            embedding: 2D embedding coordinates
            labels: Cluster labels
            method_name: Name of dimensionality reduction method
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get unique labels
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Create color map
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f'Cluster {label}' if label != -1 else 'Noise'
            
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[colors[i]], label=label_name,
                      alpha=0.6, s=30, edgecolors='none')
        
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=12)
        ax.set_title(f'2D Cluster Visualization using {method_name}', 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        if n_clusters <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     borderaxespad=0., frameon=True, fancybox=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_wordcloud(self, keywords: List[tuple], cluster_id: int, 
                      save_path=None):
        """
        Generate word cloud for a cluster.
        
        Args:
            keywords: List of (word, score) tuples
            cluster_id: Cluster identifier
            save_path: Path to save figure
        """
        # Create frequency dictionary
        word_freq = {word: score for word, score in keywords}
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Cluster {cluster_id} - Key Terms', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_cluster_keywords_heatmap(self, cluster_keywords: Dict, 
                                     top_n=15, save_path=None):
        """
        Plot heatmap of top keywords across clusters.
        
        Args:
            cluster_keywords: Dictionary mapping cluster ID to keywords
            top_n: Number of top keywords per cluster
            save_path: Path to save figure
        """
        # Prepare data
        all_keywords = set()
        for keywords in cluster_keywords.values():
            all_keywords.update([kw for kw, _ in keywords[:top_n]])
        
        # Create matrix
        keyword_list = sorted(list(all_keywords))
        cluster_ids = sorted(cluster_keywords.keys())
        
        matrix = np.zeros((len(keyword_list), len(cluster_ids)))
        
        for j, cluster_id in enumerate(cluster_ids):
            keyword_dict = {kw: score for kw, score in cluster_keywords[cluster_id]}
            for i, keyword in enumerate(keyword_list):
                matrix[i, j] = keyword_dict.get(keyword, 0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, max(10, len(keyword_list) * 0.3)))
        
        sns.heatmap(matrix, 
                   xticklabels=[f'C{i}' for i in cluster_ids],
                   yticklabels=keyword_list,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'TF-IDF Score'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Keywords', fontsize=12)
        ax.set_title('Top Keywords Heatmap Across Clusters', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_category_by_cluster(self, df: pd.DataFrame, labels: np.ndarray, 
                                top_categories=10, save_path=None):
        """
        Plot category distribution within each cluster.
        
        Args:
            df: DataFrame with 'categories' column
            labels: Cluster labels
            top_categories: Number of top categories to show
            save_path: Path to save figure
        """
        # Add cluster labels to dataframe
        df_temp = df.copy()
        df_temp['cluster'] = labels
        
        # Get top categories overall
        top_cats = df_temp['categories'].value_counts().head(top_categories).index
        
        # Count category occurrences per cluster
        cluster_cat_counts = pd.crosstab(
            df_temp['cluster'], 
            df_temp['categories']
        )[top_cats]
        
        # Normalize by cluster size
        cluster_cat_pct = cluster_cat_counts.div(
            cluster_cat_counts.sum(axis=1), axis=0
        ) * 100
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        cluster_cat_pct.plot(kind='bar', stacked=True, ax=ax, 
                            colormap='tab20')
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title('Category Distribution Across Clusters', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Category', bbox_to_anchor=(1.05, 1), 
                 loc='upper left', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
