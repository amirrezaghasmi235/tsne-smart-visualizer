# Smart t-SNE Visualizer for EDA 📊

A robust, production-ready Python module designed to simplify Dimensionality Reduction and spatial data exploration during **Exploratory Data Analysis (EDA)**. Just like pandas `.describe()` provides quick statistical summaries, this tool acts as a graphical `.describe()` for discovering hidden geometric structures, clusters, and non-linear relationships in complex datasets.

## ✨ Features
- **Smart Column Filtering:** Automatically detects and isolates non-numeric columns (like IDs, strings, or dates) so the pipeline never crashes.
- **On-Demand Scaling:** Optional built-in `StandardScaler` that automatically scales numerical data to ensure distance-based metrics work flawlessly.
- **Dynamic 2D/3D Plotting:** Seamlessly switches between **Seaborn** (for fast 2D scatter plots) and **Plotly** (for interactive, rotatable 3D web visualizations).
- **Graceful Fallbacks:** If `plotly` isn't installed, the module smoothly falls back to a static `matplotlib` 3D engine to guarantee continuity.
- **Automatic HTML Export:** Saves interactive 3D plots as standalone `.html` files, making it easy to share insights in reports or web browsers.

## 🚀 Quick Start

### 1. Installation
Make sure you have the required dependencies installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly