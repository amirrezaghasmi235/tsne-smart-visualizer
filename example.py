import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from tsne_visualizer import TSNEVisualizer
import  matplotlib.pyplot as plt
from sklearn.datasets import load_digits
# ۱. شبیه‌سازی یک دیتابیس پیچیده برای EDA
# ساخت ۱۰۰۰ نمونه داده با ۵۰ ویژگی (ستون) که به ۴ کلاس مختلف تعلق دارند
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# ۲. استفاده از همان ماژول هوشمندی که خودت ساختی (حالت ۲بعدی)
visualizer_digits = TSNEVisualizer(
    n_components=2, perplexity=30, random_state=42
)

# ۳. پردازش داده‌ها
visualizer_digits.fit_transform(X_digits, scale_numeric=True)

# ۴. رسم نمودار
visualizer_digits.plot(y=y_digits, alpha=0.7)
amirrezaghasmi235