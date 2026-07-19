### 2. How to Use

#### 🟢 Option A: 2D Visualization (Fast & Static)
```python
import pandas as pd
from sklearn.datasets import load_digits
from tsne_visualizer import TSNEVisualizer

# Load sample dataset (Handwritten Digits with 64 features)
digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

# Initialize the visualizer for 2D analysis
visualizer_2d = TSNEVisualizer(n_components=2, perplexity=30, random_state=42)

# Fit and transform
X_embedded_2d = visualizer_2d.fit_transform(X, scale_numeric=True)

# Generate a 2D Seaborn scatter plot
visualizer_2d.plot(y=y)