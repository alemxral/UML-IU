# LaTeX Presentation Compilation Instructions

## Prerequisites

You need a LaTeX distribution installed on your system.

### Windows
Download and install **MiKTeX** or **TeX Live**:
- MiKTeX: https://miktex.org/download
- TeX Live: https://www.tug.org/texlive/

### macOS
Install **MacTeX**:
```bash
brew install --cask mactex
```

### Linux
Install **TeX Live**:
```bash
sudo apt-get install texlive-full  # Ubuntu/Debian
sudo yum install texlive           # Fedora/RHEL
```

## Compilation Methods

### Method 1: Command Line (Recommended)

```bash
# Navigate to project directory
cd c:\Users\pc\Downloads\rs

# Compile (run twice for proper references)
pdflatex presentation.tex
pdflatex presentation.tex

# Output: presentation.pdf
```

### Method 2: PowerShell

```powershell
# Navigate to project directory
cd C:\Users\pc\Downloads\rs

# Compile
pdflatex presentation.tex
pdflatex presentation.tex
```

### Method 3: LaTeX Editor

**Using TeXworks (comes with MiKTeX):**
1. Open `presentation.tex` in TeXworks
2. Select "pdfLaTeX" from dropdown
3. Click green "Typeset" button
4. Run twice for proper references

**Using Overleaf (Online):**
1. Go to https://www.overleaf.com
2. Create new project → Upload Project
3. Upload `presentation.tex`
4. Create `output/` folder in project
5. Upload your generated PNG images to `output/`
6. Click "Recompile"

### Method 4: VS Code with LaTeX Workshop

```bash
# Install LaTeX Workshop extension
code --install-extension James-Yu.latex-workshop

# Open presentation.tex and press Ctrl+Alt+B to build
```

## Required LaTeX Packages

The presentation uses these packages (auto-installed by MiKTeX):
- beamer (presentation class)
- graphicx (images)
- booktabs (tables)
- amsmath (math)
- tikz (diagrams)
- xcolor (colors)
- hyperref (links)

## Before Compiling

### 1. Generate Analysis Results
Run the Jupyter notebook first to create visualizations in `output/`:
```bash
cd notebooks
jupyter notebook arxiv_trends_analysis.ipynb
```

### 2. Verify Image Files
Make sure these files exist in `output/` directory:
- category_distribution.png
- temporal_trends.png
- elbow_analysis.png
- cluster_distribution_kmeans.png
- clusters_2d_kmeans.png
- clusters_2d_dbscan.png
- wordcloud_cluster_*.png
- keywords_heatmap.png
- category_by_cluster.png
- cluster_trends_overtime.png

### 3. Update Placeholder Content
Edit `presentation.tex` and replace:
- **Line ~338**: Dataset statistics with actual values
- **Line ~478**: Clustering metrics with actual values
- **Line ~545-630**: Cluster descriptions with actual findings
- **Line ~690**: Cluster summary table with actual data

## Compilation Output

After successful compilation, you'll have:
- `presentation.pdf` - Main output (40+ slides)
- `presentation.aux` - Auxiliary file
- `presentation.log` - Compilation log
- `presentation.nav` - Navigation file
- `presentation.out` - Outline file
- `presentation.snm` - ?
- `presentation.toc` - Table of contents

You only need `presentation.pdf` for presenting.

## Common Issues

### Issue: "File not found" for images
**Solution**: 
- Ensure images are in `output/` folder
- Check file names match exactly (case-sensitive on Linux/Mac)
- Use forward slashes in paths: `output/file.png`

### Issue: Missing packages
**Solution**:
- MiKTeX will auto-install on first run
- Or manually: `miktex packages install <package-name>`

### Issue: Compilation errors
**Solution**:
- Check `presentation.log` for specific error
- Run `pdflatex presentation.tex` in terminal to see errors
- Ensure all `\end{...}` match their `\begin{...}`

### Issue: References/links broken
**Solution**:
- Compile twice: `pdflatex presentation.tex` (run 2x)
- This updates cross-references

### Issue: Beamer theme not found
**Solution**:
- Install beamer package: `miktex packages install beamer`

## Customization Tips

### Change Theme
```latex
\usetheme{Madrid}      % Try: Copenhagen, Berlin, Singapore
\usecolortheme{default} % Try: dolphin, beaver, crane
```

### Adjust Aspect Ratio
```latex
\documentclass[aspectratio=169]{beamer}  % 16:9 widescreen
% Options: 43 (4:3), 169 (16:9), 1610 (16:10)
```

### Remove Navigation Symbols
Already done:
```latex
\setbeamertemplate{navigation symbols}{}
```

### Add Logo
```latex
\logo{\includegraphics[height=1cm]{company-logo.png}}
```

### Change Font Size
```latex
\documentclass[aspectratio=169,12pt]{beamer}  % Default: 11pt
```

## Viewing the PDF

### Windows
- Double-click `presentation.pdf`
- Opens in default PDF viewer (Adobe Reader, Edge, etc.)

### Present Mode
- Press F5 in Adobe Reader for full-screen
- Use arrow keys to navigate
- Press Esc to exit

## File Cleanup

To remove auxiliary files after compilation:
```bash
# Windows PowerShell
Remove-Item presentation.aux, presentation.log, presentation.nav, presentation.out, presentation.snm, presentation.toc

# Linux/Mac
rm presentation.aux presentation.log presentation.nav presentation.out presentation.snm presentation.toc
```

Or keep only the PDF:
```bash
# Windows
del presentation.* 
# Then restore presentation.tex and presentation.pdf from backup/git
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pdflatex presentation.tex` | Compile to PDF |
| `pdflatex presentation.tex` (2nd time) | Update references |
| `F5` in Adobe Reader | Full-screen presentation mode |
| `Ctrl+L` in most viewers | Full-screen |
| Arrow keys | Navigate slides |

## Troubleshooting Command

If compilation fails, try verbose mode:
```bash
pdflatex -interaction=nonstopmode presentation.tex
```

This shows all errors without stopping.

## Alternative: Export from Jupyter

If LaTeX is too complex, you can present directly from Jupyter:
```bash
# Install RISE (presentation extension)
pip install RISE
jupyter nbextension install rise --py --sys-prefix
jupyter nbextension enable rise --py --sys-prefix

# Open notebook and click "Enter/Exit RISE Slideshow" button
```

However, the LaTeX presentation is more professional and customizable.

---

**Need Help?**
- Check `presentation.log` for detailed error messages
- Search for specific error message online
- Ask on TeX Stack Exchange: https://tex.stackexchange.com
