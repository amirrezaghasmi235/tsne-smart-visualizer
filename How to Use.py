import pandas as pd
from sklearn.datasets import load_digits
from tsne_visualizer import TSNEVisualizer

# Load 64-dimensional Handwritten Digits dataset
digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

# Add string columns to test smart defensive filtering
X['user_id'] = [f"ID_{i}" for i in range(len(X))]
X['status'] = "Active"

# Initialize and transform into 2D space
visualizer_2d = TSNEVisualizer(n_components=2, perplexity=30, random_state=42)
X_embedded_2d = visualizer_2d.fit_transform(X, scale_numeric=True)

# Plot using Seaborn engine
visualizer_2d.plot(y=y, palette="tab10", alpha=0.7)
#__________________
# Initialize and transform into 3D space
visualizer_3d = TSNEVisualizer(n_components=3, perplexity=35, random_state=42)
X_embedded_3d = visualizer_3d.fit_transform(X, scale_numeric=True)

# Plot using Plotly engine (Saves a standalone 'tsne_3d_result.html')
visualizer_3d.plot(y=y, alpha=0.8)