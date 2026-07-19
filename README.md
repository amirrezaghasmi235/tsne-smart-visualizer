# Smart t-SNE Visualizer for EDA 📊

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

An enterprise-grade, object-oriented Python module designed to optimize, automate, and accelerate high-dimensional spatial data exploration during **Exploratory Data Analysis (EDA)**. Just as pandas `.describe()` provides a statistical profile of numbers, `TSNEVisualizer` acts as a topological wrapper that maps and exposes the underlying non-linear geometry, manifold structures, and clustering boundaries of your datasets.

---

## 🧠 Mathematical Philosophy & The EDA Problem
In high-dimensional spaces (e.g., datasets with 50+ features), standard statistical metrics and linear projections (like PCA) often obscure the true relationship between classes due to the *Curse of Dimensionality*. 

While **t-Distributed Stochastic Neighbor Embedding (t-SNE)** excels at capturing local similarities by converting Euclidean distances into conditional probabilities, implementing it directly in raw EDA workflows poses significant operational hurdles:
1.  **Distance Distortion:** Features with larger mathematical scales dominate the distance matrix unless carefully normalized.
2.  **Data Incompatibility:** Categorical text columns, indexing strings, or datetime signatures immediately trigger system crashes inside distance-based estimators.
3.  **Visual Overlap:** 2D projections of complex manifolds frequently lead to visual crowding, necessitating dynamic 3D spatial separation.

`TSNEVisualizer` acts as an automated, defensive pipeline wrapper that sanitizes data structures, applies mathematical scale parity, reduces dimensions, and deploys context-aware rendering engines under a unified API.

---

## 🛠️ Architecture & Pipeline Flow
The module isolates the raw inputs and handles complex validation stages abstractly behind the scenes:


```

[ Raw Heterogeneous DataFrame ]
│
▼
┌──────────────────────────────┐
│  Automated Feature Sieve     │ ──> Isolates non-numeric data (IDs, dates, text strings)
└──────────────────────────────┘
│
▼ (Numeric values only)
┌──────────────────────────────┐
│   Standard Parity Scaling   │ ──> Applies μ=0, σ=1 via Sklearn StandardScaler
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│     t-SNE Core Manifold      │ ──> Computes conditional probabilities and lowers space
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│   Adaptive Visual Engine     │ ──> Evaluates target space bounds
└──────────────────────────────┘
│              │
▼ (2D Space)   ▼ (3D Space)
[Seaborn Render]   [Plotly Interactive HTML Export]

```

---

## ✨ Enterprise Features
*   **Defensive Type Analysis:** Dynamically maps input structures, executing column parsing to isolate string flags, primary keys, and categorical metrics from the mathematical calculation.
*   **Scale Normalization Parity:** Integrates optional, state-tracked `StandardScaler` transformations to ensure mathematical parity across varied scale features.
*   **Multi-Dimensional Visual Engine:** Intuitively switches graphics pipelines based on dimensional configuration, instantly adapting from static matrix charts to interactive webs.
*   **Plotly-to-Matplotlib Fallback:** Implements a try-except environment wrapper that automatically deploys a native static 3D `mpl_toolkits.mplot3d` graph if runtime dependencies like Plotly are absent.
*   **Asynchronous Report Exporting:** Generates lightweight, portable, responsive standalone `.html` representations for 3D outputs, allowing seamless deployment into dashboards or web reports.

---

## 🚀 Installation

Ensure you deploy the package dependencies listed in your environment:

```bash
pip install -r requirements.txt

```

*Note: Core libraries include `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, and `plotly`.*

---

## 📖 Complete API Reference

### `TSNEVisualizer(__init__)`

Initializes the dimensional reduction wrapping layer.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n_components` | `int` | `2` | Number of dimensions to compress down to. Supported options: `2` or `3`. |
| `perplexity` | `float/int` | `30` | The balance parameter representing the number of effective nearest neighbors (typically between `5` and `50`). |
| `random_state` | `int` | `42` | Deterministic random seed ensuring reproducibility across multiple executions. |

### `fit_transform(X, scale_numeric=True)`

Executes automated pre-processing pipelines and performs manifold projection.

* **Inputs:**
* `X` (*DataFrame or Array-like*): High-dimensional input feature space.
* `scale_numeric` (*bool*): Toggles automated Z-score scaling.


* **Returns:** `np.ndarray` of shape `(n_samples, n_components)`.

### `plot(y=None, figsize=(8, 6), palette="bright", alpha=0.8)`

Dynamically matches dimensional outputs and renders statistical distributions.

* **Inputs:**
* `y` (*Array-like, optional*): Target labels or class matrices used for conditional hue mapping.
* `figsize` (*tuple*): Spatial size mapping for localized static rendering.
* `palette` (*str*): Categorical hue maps (e.g., `"bright"`, `"tab10"`, `"viridis"`).
* `alpha` (*float*): Point blending opacity bounds (`0.0` to `1.0`).



---

💻 Code Implementation Examples

🟢 Use Case A: Accelerated 2D Exploratory Analysis

Optimized for swift feature-space separation reviews and validation checks.

```python
import pandas as pd
from sklearn.datasets import load_digits
from tsne_visualizer import TSNEVisualizer

# 1. Load high-dimensional structural data (64-pixel feature space)
digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

# Add unstructured metadata columns to test the visualizer's robustness
X['record_hash'] = [f"HASH_{i}" for i in range(len(X))]
X['system_flag'] = "Verified"

# 2. Instantiate and process the pipeline for 2D space
visualizer_2d = TSNEVisualizer(n_components=2, perplexity=30, random_state=42)
X_transformed_2d = visualizer_2d.fit_transform(X, scale_numeric=True)

# 3. Render categorical classification maps
visualizer_2d.plot(y=y, palette="tab10", alpha=0.7)

```

### 🔵 Use Case B: Interactive 3D Topological Exploration

Deploys rich HTML graphics to explore complex, overlapping manifolds.

```python
# 1. Instantiate and process pipeline for 3D space
visualizer_3d = TSNEVisualizer(n_components=3, perplexity=35, random_state=42)
X_transformed_3d = visualizer_3d.fit_transform(X, scale_numeric=True)

# 2. Render and output web-based HTML report
visualizer_3d.plot(y=y, alpha=0.8)

```

---

## 📊 Technical Comparison Matrix

| Operational Vector | 2D Environment (`n_components=2`) | 3D Environment (`n_components=3`) |
| --- | --- | --- |
| **Graphics Engine** | Seaborn / Matplotlib Core | Plotly Express Engine |
| **User Interaction** | Static Vector Image | Axis Rotation, Zoom, Pan, Datapoint Hover |
| **Output Type** | Direct Inline Block | Standalone Executable Web Page (`.html`) |
| **Primary Use Case** | Linear Separation Diagnostics | Manifold Intersection and Depth Auditing |
| **System Resiliency** | Standard Exception Handling | Native `mplot3d` Secondary Fallback Engine |

---

## 💡 Interpreting EDA Outputs (The "Ball of Noise" Warning)

If your dataset projects as an isotropic, unclustered spherical mass, the module has done its job perfectly. It reveals that the selected high-dimensional features currently lack geometric or topological class boundaries. This acts as an immediate signal to the engineer to shift focus toward **Feature Engineering**, data sanitization, or hyperparameter adjustments (`perplexity`, `learning_rate`) before wasting resources on complex machine learning model training.

---

Developed with 💻 by [amirrezaghasmi235](https://github.com/amirrezaghasmi235)

