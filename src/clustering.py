"""
Clustering module for scientific paper analysis.
Implements K-Means, DBSCAN, and cluster evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import umap
import warnings
warnings.filterwarnings('ignore')


class ClusterAnalyzer:
    """Comprehensive clustering analysis for scientific papers."""
    
    def __init__(self):
        """Initialize the cluster analyzer."""
        self.models = {}
        self.labels = {}
        self.metrics = {}
        
    def find_optimal_k(self, X: np.ndarray, k_range=range(2, 15), 
                      method='elbow') -> Tuple[int, Dict]:
        """
        Find optimal number of clusters using elbow method or silhouette analysis.
        
        Args:
            X: Feature matrix
            k_range: Range of k values to test
            method: 'elbow', 'silhouette', or 'both'
            
        Returns:
            Tuple of (optimal k, metrics dictionary)
        """
        inertias = []
        silhouettes = []
        davies_bouldin = []
        calinski_harabasz = []
        
        print(f"Testing K values from {min(k_range)} to {max(k_range)}...")
        
        for k in k_range:
            # Fit K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            davies_bouldin.append(davies_bouldin_score(X, labels))
            calinski_harabasz.append(calinski_harabasz_score(X, labels))
            
            print(f"K={k}: Silhouette={silhouettes[-1]:.3f}, "
                  f"Davies-Bouldin={davies_bouldin[-1]:.3f}")
        
        metrics_dict = {
            'k_values': list(k_range),
            'inertia': inertias,
            'silhouette': silhouettes,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz
        }
        
        # Find optimal k based on silhouette score (higher is better)
        optimal_k = list(k_range)[np.argmax(silhouettes)]
        
        print(f"\nOptimal K based on Silhouette Score: {optimal_k}")
        
        return optimal_k, metrics_dict
    
    def kmeans_clustering(self, X: np.ndarray, n_clusters=8, 
                         random_state=42) -> Tuple[np.ndarray, KMeans]:
        """
        Perform K-Means clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        print(f"Performing K-Means clustering with {n_clusters} clusters...")
        
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        labels = model.fit_predict(X)
        
        # Store model and labels
        self.models['kmeans'] = model
        self.labels['kmeans'] = labels
        
        # Calculate metrics
        self.metrics['kmeans'] = {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'inertia': model.inertia_
        }
        
        print(f"K-Means Metrics:")
        print(f"  Silhouette Score: {self.metrics['kmeans']['silhouette']:.3f}")
        print(f"  Davies-Bouldin Index: {self.metrics['kmeans']['davies_bouldin']:.3f}")
        print(f"  Calinski-Harabasz Score: {self.metrics['kmeans']['calinski_harabasz']:.1f}")
        
        return labels, model
    
    def dbscan_clustering(self, X: np.ndarray, eps=0.5, 
                         min_samples=5) -> Tuple[np.ndarray, DBSCAN]:
        """
        Perform DBSCAN clustering.
        
        Args:
            X: Feature matrix
            eps: Maximum distance between two samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        print(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X)
        
        # Store model and labels
        self.models['dbscan'] = model
        self.labels['dbscan'] = labels
        
        # Count clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        # Calculate metrics (excluding noise points)
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0:
                self.metrics['dbscan'] = {
                    'silhouette': silhouette_score(X[mask], labels[mask]) if len(set(labels[mask])) > 1 else 0,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': n_noise / len(labels)
                }
                print(f"  Silhouette Score (excluding noise): {self.metrics['dbscan']['silhouette']:.3f}")
        
        return labels, model
    
    def hierarchical_clustering(self, X: np.ndarray, n_clusters=8, 
                               linkage='ward') -> Tuple[np.ndarray, AgglomerativeClustering]:
        """
        Perform Hierarchical/Agglomerative clustering.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average')
            
        Returns:
            Tuple of (cluster labels, fitted model)
        """
        print(f"Performing Hierarchical clustering with {n_clusters} clusters...")
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = model.fit_predict(X)
        
        # Store model and labels
        self.models['hierarchical'] = model
        self.labels['hierarchical'] = labels
        
        # Calculate metrics
        self.metrics['hierarchical'] = {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels)
        }
        
        print(f"Hierarchical Metrics:")
        print(f"  Silhouette Score: {self.metrics['hierarchical']['silhouette']:.3f}")
        
        return labels, model
    
    def get_cluster_statistics(self, labels: np.ndarray, 
                              df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate statistics for each cluster.
        
        Args:
            labels: Cluster labels
            df: Optional dataframe with additional features
            
        Returns:
            DataFrame with cluster statistics
        """
        cluster_ids, counts = np.unique(labels, return_counts=True)
        
        stats = pd.DataFrame({
            'cluster_id': cluster_ids,
            'size': counts,
            'percentage': counts / len(labels) * 100
        })
        
        # Add category distribution if dataframe provided
        if df is not None and 'categories' in df.columns:
            category_dist = []
            for cluster_id in cluster_ids:
                mask = labels == cluster_id
                top_cat = df[mask]['categories'].mode()
                category_dist.append(top_cat.iloc[0] if len(top_cat) > 0 else 'Unknown')
            stats['dominant_category'] = category_dist
        
        return stats.sort_values('size', ascending=False)


class DimensionalityReducer:
    """Dimensionality reduction for visualization."""
    
    def __init__(self):
        """Initialize the reducer."""
        self.models = {}
        self.embeddings = {}
        
    def reduce_pca(self, X: np.ndarray, n_components=50) -> np.ndarray:
        """
        Reduce dimensions using PCA.
        
        Args:
            X: Feature matrix
            n_components: Number of components
            
        Returns:
            Reduced feature matrix
        """
        print(f"Reducing to {n_components} dimensions with PCA...")
        
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)
        
        self.models['pca'] = pca
        self.embeddings['pca'] = X_reduced
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained_var:.2%}")
        
        return X_reduced
    
    def reduce_umap(self, X: np.ndarray, n_components=2, n_neighbors=15, 
                   min_dist=0.1, metric='cosine') -> np.ndarray:
        """
        Reduce dimensions using UMAP for visualization.
        
        Args:
            X: Feature matrix
            n_components: Number of dimensions (typically 2 or 3)
            n_neighbors: Number of neighbors to consider
            min_dist: Minimum distance between points
            metric: Distance metric
            
        Returns:
            UMAP embedding
        """
        print(f"Reducing to {n_components}D with UMAP...")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        
        embedding = reducer.fit_transform(X)
        
        self.models['umap'] = reducer
        self.embeddings['umap'] = embedding
        
        print("UMAP reduction complete")
        
        return embedding
