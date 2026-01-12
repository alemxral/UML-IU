#!/usr/bin/env python3
"""
LaTeX Report Generator for ArXiv Trends Analysis.
Automatically populates case_study.tex with actual analysis results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.helpers import load_json
from utils.logging_config import setup_logging

logger = logging.getLogger("arxiv_trends")


class LaTeXReportGenerator:
    """Generate LaTeX report from analysis results."""
    
    def __init__(self, latex_data_path: str, template_path: str, output_path: str):
        """
        Initialize report generator.
        
        Args:
            latex_data_path: Path to JSON file with analysis results
            template_path: Path to LaTeX template file
            output_path: Path to save generated LaTeX file
        """
        self.latex_data_path = Path(latex_data_path)
        self.template_path = Path(template_path)
        self.output_path = Path(output_path)
        self.data = None
        
    def load_data(self):
        """Load analysis results from JSON."""
        logger.info(f"Loading analysis data from {self.latex_data_path}")
        self.data = load_json(str(self.latex_data_path))
        logger.info("Data loaded successfully")
        
    def generate_cluster_descriptions(self) -> str:
        """
        Generate LaTeX code for cluster descriptions.
        
        Returns:
            LaTeX formatted cluster descriptions
        """
        latex_code = []
        
        for cluster in self.data['clusters']:
            cluster_id = cluster['id']
            size = cluster['size']
            pct = cluster['percentage']
            keywords = cluster['top_keywords']
            
            # Generate a descriptive name based on keywords
            # This is simplified - ideally would use domain knowledge
            cluster_name = f"Cluster {cluster_id}"
            
            latex_code.append(
                f"\\textbf{{Cluster {cluster_id}}} ({size:,} papers, {pct:.1f}\\%): "
                f"Primary keywords: {keywords}.\n"
            )
        
        return "\n".join(latex_code)
    
    def generate_metrics_table(self) -> str:
        """
        Generate LaTeX table with clustering metrics.
        
        Returns:
            LaTeX formatted metrics table
        """
        dataset = self.data['dataset']
        clustering = self.data['clustering']
        
        latex_code = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Clustering Performance Metrics}",
            "\\begin{tabular}{lr}",
            "\\hline",
            "\\textbf{Metric} & \\textbf{Value} \\\\",
            "\\hline",
            f"Total Papers & {dataset['total_papers']:,} \\\\",
            f"Number of Clusters & {clustering['n_clusters']} \\\\",
            f"Silhouette Score & {clustering['silhouette_score']:.3f} \\\\",
            f"Davies-Bouldin Index & {clustering['davies_bouldin_index']:.3f} \\\\",
            f"Calinski-Harabasz Index & {clustering['calinski_harabasz_index']:.1f} \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:metrics}",
            "\\end{table}"
        ]
        
        return "\n".join(latex_code)
    
    def generate_cluster_sizes_table(self) -> str:
        """
        Generate LaTeX table with cluster sizes.
        
        Returns:
            LaTeX formatted cluster sizes table
        """
        latex_code = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Cluster Sizes and Percentages}",
            "\\begin{tabular}{lrr}",
            "\\hline",
            "\\textbf{Cluster} & \\textbf{Papers} & \\textbf{Percentage} \\\\",
            "\\hline"
        ]
        
        for cluster in self.data['clusters']:
            latex_code.append(
                f"Cluster {cluster['id']} & {cluster['size']:,} & {cluster['percentage']:.1f}\\% \\\\"
            )
        
        latex_code.extend([
            "\\hline",
            "\\end{tabular}",
            "\\label{tab:cluster_sizes}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_code)
    
    def generate_keywords_list(self, cluster_id: int, n_keywords: int = 20) -> str:
        """
        Generate LaTeX formatted keyword list for a cluster.
        
        Args:
            cluster_id: Cluster identifier
            n_keywords: Number of keywords to include
            
        Returns:
            LaTeX formatted keyword list
        """
        cluster = next((c for c in self.data['clusters'] if c['id'] == cluster_id), None)
        if not cluster:
            return ""
        
        keywords = cluster['keywords'][:n_keywords]
        return ", ".join(keywords)
    
    def generate_report(self):
        """Generate the complete LaTeX report."""
        logger.info(f"Generating LaTeX report from template: {self.template_path}")
        
        # Load template
        if not self.template_path.exists():
            logger.error(f"Template file not found: {self.template_path}")
            logger.info("Using existing case_study.tex without modifications")
            return
        
        with open(self.template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Replace placeholders with actual data
        # Note: This is a simplified version. The actual implementation would need
        # to handle more complex replacements based on the template structure.
        
        replacements = {
            '{{TOTAL_PAPERS}}': f"{self.data['dataset']['total_papers']:,}",
            '{{N_CLUSTERS}}': str(self.data['clustering']['n_clusters']),
            '{{SILHOUETTE_SCORE}}': f"{self.data['clustering']['silhouette_score']:.3f}",
            '{{DAVIES_BOULDIN}}': f"{self.data['clustering']['davies_bouldin_index']:.3f}",
            '{{CALINSKI_HARABASZ}}': f"{self.data['clustering']['calinski_harabasz_index']:.1f}",
        }
        
        # Apply replacements
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        
        # Save generated report
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(template)
        
        logger.info(f"LaTeX report generated: {self.output_path}")
        logger.info("\nTo compile the report:")
        logger.info(f"  cd {self.output_path.parent}")
        logger.info(f"  pdflatex {self.output_path.name}")
        logger.info(f"  pdflatex {self.output_path.name}  # Run twice for TOC")
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS RESULTS SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nDataset:")
        logger.info(f"  Total Papers: {self.data['dataset']['total_papers']:,}")
        logger.info(f"  Unique Categories: {self.data['dataset']['unique_categories']}")
        logger.info(f"  Year Range: {self.data['dataset']['year_range']}")
        
        logger.info(f"\nClustering:")
        logger.info(f"  Number of Clusters: {self.data['clustering']['n_clusters']}")
        logger.info(f"  Silhouette Score: {self.data['clustering']['silhouette_score']:.3f}")
        logger.info(f"  Davies-Bouldin Index: {self.data['clustering']['davies_bouldin_index']:.3f}")
        
        logger.info(f"\nCluster Details:")
        for cluster in self.data['clusters']:
            logger.info(f"  Cluster {cluster['id']}: {cluster['size']:,} papers ({cluster['percentage']:.1f}%)")
            logger.info(f"    Keywords: {cluster['top_keywords']}")
        
        logger.info("=" * 80 + "\n")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate LaTeX report from ArXiv trends analysis results'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='output/data/latex_report_data.json',
        help='Path to analysis results JSON file'
    )
    
    parser.add_argument(
        '--template',
        type=str,
        default='case_study.tex',
        help='Path to LaTeX template file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='case_study_generated.tex',
        help='Path to save generated LaTeX file'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary, do not generate report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(log_level=log_level, console_output=True)
    
    try:
        # Initialize generator
        generator = LaTeXReportGenerator(
            latex_data_path=args.data,
            template_path=args.template,
            output_path=args.output
        )
        
        # Load data
        generator.load_data()
        
        # Print summary
        generator.print_summary()
        
        # Generate report unless summary-only
        if not args.summary_only:
            generator.generate_report()
        else:
            logger.info("Summary-only mode: skipping report generation")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("\nMake sure to run the analysis pipeline first:")
        logger.error("  python src/main.py")
        return 1
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
