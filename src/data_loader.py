"""
Data loader module for ArXiv dataset.
Handles downloading and preparing the dataset for analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class ArxivDataLoader:
    """Load and prepare ArXiv dataset."""
    
    def __init__(self, data_dir='../data'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_from_json(self, file_path: str, 
                      sample_size: Optional[int] = None,
                      recent_years: Optional[int] = None) -> pd.DataFrame:
        """
        Load ArXiv data from JSON file.
        
        Args:
            file_path: Path to JSON file
            sample_size: Number of records to load (None for all)
            recent_years: Only load papers from last N years
            
        Returns:
            DataFrame with paper metadata
        """
        print(f"Loading data from {file_path}...")
        
        papers = []
        count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if sample_size and count >= sample_size:
                        break
                    
                    try:
                        paper = json.loads(line)
                        papers.append(paper)
                        count += 1
                        
                        if count % 10000 == 0:
                            print(f"Loaded {count} papers...")
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("\nTo obtain the ArXiv dataset:")
            print("1. Download from Kaggle: https://www.kaggle.com/Cornell-University/arxiv")
            print("2. Or use the provided SharePoint link")
            print("3. Extract to the data/ directory")
            return pd.DataFrame()
        
        print(f"Total papers loaded: {len(papers)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        # Filter by recent years if specified
        if recent_years and 'versions' in df.columns:
            df = self._filter_recent_papers(df, recent_years)
        
        return df
    
    def _filter_recent_papers(self, df: pd.DataFrame, 
                             recent_years: int) -> pd.DataFrame:
        """
        Filter papers from recent years.
        
        Args:
            df: DataFrame with papers
            recent_years: Number of recent years to keep
            
        Returns:
            Filtered DataFrame
        """
        print(f"Filtering papers from last {recent_years} years...")
        
        # Extract year from versions
        if 'versions' in df.columns:
            df['year'] = df['versions'].apply(
                lambda x: int(x[0]['created'].split(',')[-1].strip()) 
                if isinstance(x, list) and len(x) > 0 else None
            )
        
        current_year = pd.Timestamp.now().year
        cutoff_year = current_year - recent_years
        
        df_filtered = df[df['year'] >= cutoff_year].copy()
        print(f"Papers after filtering: {len(df_filtered)}")
        
        return df_filtered
    
    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean the dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Preparing dataset...")
        
        df_clean = df.copy()
        
        # Extract primary category
        if 'categories' in df_clean.columns:
            df_clean['categories'] = df_clean['categories'].apply(
                lambda x: x.split()[0] if isinstance(x, str) and x else 'Unknown'
            )
        
        # Extract year if not already done
        if 'year' not in df_clean.columns and 'versions' in df_clean.columns:
            df_clean['year'] = df_clean['versions'].apply(
                lambda x: int(x[0]['created'].split(',')[-1].strip()) 
                if isinstance(x, list) and len(x) > 0 else None
            )
        
        # Clean abstract
        if 'abstract' in df_clean.columns:
            df_clean['abstract'] = df_clean['abstract'].fillna('')
            # Remove leading/trailing whitespace and newlines
            df_clean['abstract'] = df_clean['abstract'].str.replace('\n', ' ')
            df_clean['abstract'] = df_clean['abstract'].str.strip()
        
        # Clean title
        if 'title' in df_clean.columns:
            df_clean['title'] = df_clean['title'].fillna('')
            df_clean['title'] = df_clean['title'].str.replace('\n', ' ')
            df_clean['title'] = df_clean['title'].str.strip()
        
        # Extract number of authors
        if 'authors_parsed' in df_clean.columns:
            df_clean['num_authors'] = df_clean['authors_parsed'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Remove papers without abstracts
        df_clean = df_clean[df_clean['abstract'].str.len() > 50].copy()
        
        print(f"Dataset prepared: {len(df_clean)} papers")
        
        # Display basic info
        print("\nDataset Info:")
        print(f"Columns: {list(df_clean.columns)}")
        print(f"Date range: {df_clean['year'].min()} - {df_clean['year'].max()}")
        print(f"Unique categories: {df_clean['categories'].nunique()}")
        
        return df_clean
    
    def create_sample_dataset(self, output_path: str, n_samples=50000):
        """
        Create a sample dataset for demonstration purposes.
        
        Args:
            output_path: Path to save sample data
            n_samples: Number of samples to create
        """
        print(f"Creating sample dataset with {n_samples} papers...")
        
        # Simulate ArXiv-like data
        categories = ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'physics.data-an',
                     'stat.ML', 'math.OC', 'q-bio.GN', 'astro-ph.GA', 'cond-mat.str-el']
        
        sample_abstracts = [
            "This paper presents a novel deep learning approach for image classification...",
            "We propose a new algorithm for reinforcement learning in complex environments...",
            "Our study investigates the properties of quantum materials using advanced simulations...",
            "We analyze large-scale genomic data to identify disease-associated variants...",
            "This work develops mathematical optimization techniques for network design...",
        ]
        
        data = {
            'id': [f'paper_{i}' for i in range(n_samples)],
            'title': [f'Research Paper {i}' for i in range(n_samples)],
            'abstract': np.random.choice(sample_abstracts, n_samples),
            'categories': np.random.choice(categories, n_samples),
            'year': np.random.randint(2018, 2024, n_samples),
            'num_authors': np.random.randint(1, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"Sample dataset saved to {output_path}")
        
        return df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate basic statistics about the dataset.
        
        Args:
            df: DataFrame with papers
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_papers': len(df),
            'unique_categories': df['categories'].nunique() if 'categories' in df.columns else 0,
            'year_range': (df['year'].min(), df['year'].max()) if 'year' in df.columns else (None, None),
            'avg_abstract_length': df['abstract'].str.len().mean() if 'abstract' in df.columns else 0,
            'avg_authors': df['num_authors'].mean() if 'num_authors' in df.columns else 0,
        }
        
        return stats


def download_instructions():
    """Print instructions for downloading the ArXiv dataset."""
    
    instructions = """
    ===== ArXiv Dataset Download Instructions =====
    
    Option 1: Kaggle API (Recommended)
    ----------------------------------
    1. Install Kaggle CLI: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to https://www.kaggle.com/account
       - Click "Create New API Token"
       - Place kaggle.json in ~/.kaggle/
    3. Download dataset:
       kaggle datasets download -d Cornell-University/arxiv
    4. Extract to data/ directory
    
    Option 2: Manual Download
    ------------------------
    1. Visit: https://www.kaggle.com/Cornell-University/arxiv
    2. Click "Download" button
    3. Extract arxiv-metadata-oai-snapshot.json to data/ directory
    
    Option 3: SharePoint Link
    ------------------------
    1. Visit the provided SharePoint link
    2. Download the dataset
    3. Extract to data/ directory
    
    Note: The full dataset is ~3.5GB and contains 2M+ papers.
    You may want to work with a subset for faster processing.
    """
    
    print(instructions)


if __name__ == "__main__":
    download_instructions()
