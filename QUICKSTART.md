# Quick Start Guide - ArXiv Trends Analysis

## Project Overview
Complete data science case study for categorizing trends in science using the ArXiv dataset through clustering and dimensionality reduction techniques.

## Files Created

### Source Code (`src/`)
- **data_loader.py**: Download and prepare ArXiv dataset
- **preprocessing.py**: Text cleaning, tokenization, TF-IDF feature extraction
- **clustering.py**: K-Means, DBSCAN, cluster evaluation, UMAP/PCA
- **visualization.py**: All plotting functions for EDA and results
- **__init__.py**: Package initialization

### Analysis (`notebooks/`)
- **arxiv_trends_analysis.ipynb**: Complete analysis pipeline with 17 sections

### Presentation
- **presentation.tex**: Comprehensive LaTeX beamer presentation (40+ slides)

### Configuration
- **requirements.txt**: All Python dependencies
- **README.md**: Project documentation

## Getting Started

### Step 1: Install Dependencies

#### Option A: Using pip (Standard)

```bash
# Install Python packages
pip install -r requirements.txt

# Download NLP model
python -m spacy download en_core_web_sm
``` 

#### Option B: Using pipenv (Recommended for isolated environments)

```bash
# Install pipenv if you don't have it
pip install pipenv

# Create virtual environment and install dependencies from requirements.txt
pipenv install -r requirements.txt

# Activate the pipenv environment
pipenv shell

# Download NLP model (inside pipenv shell)
python -m spacy download en_core_web_sm

# Alternative: Run commands without activating shell
pipenv run python -m spacy download en_core_web_sm
pipenv run jupyter notebook
```

### Step 2: Obtain Data

**Option A - Kaggle (Recommended)**
```bash
# Configure Kaggle API
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d Cornell-University/arxiv
# Extract to data/ directory
```

**Option B - Manual Download**
1. Visit: https://www.kaggle.com/Cornell-University/arxiv
2. Download arxiv-metadata-oai-snapshot.json
3. Place in `data/` directory

**Option C - Sample Data (for testing)**
The notebook will automatically create sample data if the full dataset is not available.

### Step 3: Run Analysis

```bash
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook arxiv_trends_analysis.ipynb
```

Then execute cells sequentially (Shift+Enter).

### Step 4: Generate Presentation

```bash
# Compile LaTeX presentation
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for references

# Output: presentation.pdf
```

## Key Configuration Parameters

In the Jupyter notebook, you can adjust:

```python
SAMPLE_SIZE = 50000  # Number of papers to analyze
RECENT_YEARS = 3     # Focus on last N years
N_CLUSTERS = 8       # Number of clusters (or use optimal_k)
```

## Expected Outputs

### In `output/` directory:
- `category_distribution.png` - Top categories bar chart
- `temporal_trends.png` - Publication trends over time
- `elbow_analysis.png` - Cluster optimization metrics
- `cluster_distribution_kmeans.png` - Cluster sizes
- `clusters_2d_kmeans.png` - 2D UMAP visualization
- `wordcloud_cluster_*.png` - Word clouds for each cluster
- `keywords_heatmap.png` - Keyword importance heatmap
- `category_by_cluster.png` - Category distribution
- `cluster_trends_overtime.png` - Temporal evolution
- `cluster_assignments.csv` - Paper-to-cluster mappings
- `cluster_keywords.csv` - Top keywords per cluster

## Analysis Pipeline Summary

1. **Data Acquisition** (Section 2-3)
   - Load ArXiv JSON data
   - Sample and filter by recency
   - Clean and prepare dataset

2. **Exploratory Analysis** (Section 4)
   - Category distribution
   - Temporal trends
   - Abstract statistics

3. **Preprocessing** (Section 5)
   - Text cleaning (LaTeX, URLs, special chars)
   - Tokenization and stopword removal
   - Lemmatization

4. **Feature Engineering** (Section 6)
   - TF-IDF vectorization
   - 5000 features, unigrams + bigrams

5. **Dimensionality Reduction** (Section 7)
   - PCA: 5000 → 50 dimensions
   - UMAP: 50 → 2 dimensions (visualization)

6. **Cluster Optimization** (Section 8)
   - Elbow method
   - Silhouette score
   - Davies-Bouldin index

7. **Clustering** (Section 9-10)
   - K-Means clustering
   - DBSCAN (alternative)
   - Cluster statistics

8. **Interpretation** (Section 11-14)
   - Extract top keywords
   - Generate word clouds
   - Label clusters
   - Sample validation

9. **Trend Analysis** (Section 15)
   - Growth rates
   - Temporal evolution
   - Category analysis

10. **Export Results** (Section 16)
    - Save cluster assignments
    - Export keywords
    - Generate reports

## Presentation Structure

The LaTeX presentation includes:

1. **Introduction** - Business context and research questions
2. **Methodology** - Pipeline, preprocessing, algorithms
3. **EDA** - Dataset overview and distributions
4. **Results** - Clustering performance and visualizations
5. **Cluster Interpretation** - Detailed analysis of each cluster
6. **Trend Analysis** - Growth patterns and evolution
7. **Recommendations** - Cooperation priorities and strategy
8. **Critical Assessment** - Limitations and quality evaluation
9. **Future Work** - Potential improvements
10. **Conclusion** - Key findings and next steps
11. **Appendix** - Technical details and samples

## Customization Tips

### Adjust Preprocessing
Edit `src/preprocessing.py`:
- Modify `scientific_stopwords` set
- Change lemmatization settings
- Adjust TF-IDF parameters

### Change Clustering
Edit notebook Section 9:
- Try different K values
- Adjust DBSCAN eps/min_samples
- Experiment with hierarchical clustering

### Customize Visualizations
Edit `src/visualization.py`:
- Change color schemes
- Modify plot sizes
- Add new plot types

### Update Presentation
Edit `presentation.tex`:
- Replace cluster names based on your results
- Update statistics with actual values
- Add/remove slides as needed
- Include actual figure paths

## Troubleshooting

### "File not found" error
- Ensure data is in `data/` directory
- Or let notebook create sample data

### Memory errors
- Reduce `SAMPLE_SIZE`
- Use PCA earlier in pipeline
- Process in batches

### Import errors
- Run `pip install -r requirements.txt`
- Check Python version (3.8+)

### NLTK download errors
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### LaTeX compilation errors
- Install TeX distribution (MiKTeX, TeX Live)
- Ensure Beamer package available
- Check for missing LaTeX packages

## Performance Notes

- **50K papers**: ~30 minutes processing time, ~8GB RAM
- **100K papers**: ~60 minutes, ~16GB RAM
- **Full dataset (2M+)**: Use HPC cluster or cloud

## Next Steps After Running

1. **Review Results**
   - Examine cluster visualizations
   - Validate keyword interpretations
   - Check sample papers

2. **Refine Analysis**
   - Adjust preprocessing based on results
   - Try different cluster numbers
   - Experiment with parameters

3. **Prepare Presentation**
   - Update LaTeX with actual results
   - Generate PDF
   - Practice delivery

4. **Document Findings**
   - Write executive summary
   - Compile recommendations
   - Prepare for stakeholder review

## Contact & Support

For questions or issues:
- Review README.md
- Check code comments
- Consult docstrings in source files

## License

This case study is for educational and research purposes.
ArXiv data subject to their terms of use.

---

**Last Updated**: December 29, 2025
**Version**: 1.0
