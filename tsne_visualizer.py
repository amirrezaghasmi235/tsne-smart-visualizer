import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class TSNEVisualizer:

    def __init__(self, n_components=2, perplexity=30, random_state=42):
        self.n_components = n_components
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.X_tsne = None

    def fit_transform(self, X, scale_numeric=True):
        """اعمال الگوریتم روی دیتافریم ورودی و ذخیره نتایج"""
        
        if isinstance(X, pd.DataFrame):
            # ۱. فقط ستون‌های عددی را برای الگوریتم انتخاب می‌کنیم
            numeric_cols = X.select_dtypes(include=['number']).columns
            X_numeric = X[numeric_cols].copy()
            
            # ۲. اگر کاربر استانداردسازی خواست، روی ستون‌های عددی اعمال می‌کنیم
            if scale_numeric:
                X_final = self.scaler.fit_transform(X_numeric)
            else:
                X_final = X_numeric.values
        else:
            # اگر از اول آرایه نامپای بود، فرض می‌کنیم تماماً عددی است
            X_final = self.scaler.fit_transform(X) if scale_numeric else X

        # ۳. حالا با خیال راحت داده‌های ۱۰۰٪ عددی را به t-SNE می‌دهیم
        self.X_tsne = self.tsne.fit_transform(X_final)
        return self.X_tsne

    def plot(self, y=None, figsize=(8, 6), palette="bright", alpha=0.8):
        """رسم هوشمند نمودار بر اساس ۲ بعدی یا ۳ بعدی بودن خروجی t-SNE"""
        if self.X_tsne is None:
            raise ValueError(
                "ابتدا باید متد fit_transform را روی داده‌ها اجرا کنید!"
            )

        # آماده‌سازی لیبل‌ها
        labels = np.array(y) if y is not None else None

        # حالت اول: رسم نمودار ۲ بعدی (مانند قبل با Seaborn)
        if self.n_components == 2:
            plot_df = pd.DataFrame(self.X_tsne, columns=["t-SNE 1", "t-SNE 2"])
            if labels is not None:
                plot_df["Label"] = labels
            hue_param = "Label" if labels is not None else None

            plt.figure(figsize=figsize)
            sns.scatterplot(
                data=plot_df,
                x="t-SNE 1",
                y="t-SNE 2",
                hue=hue_param,
                palette=palette,
                alpha=alpha,
            )
            plt.title(
                "2D t-SNE Visualization of Complex Data", fontsize=14, pad=15
            )
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

        # حالت دوم: رسم نمودار ۳ بعدی
        elif self.n_components == 3:
            # ساخت دیتافریم برای پلات ۳بعدی
            plot_df = pd.DataFrame(
                self.X_tsne, columns=["t-SNE 1", "t-SNE 2", "t-SNE 3"]
            )
            if labels is not None:
                # تبدیل به استرینگ برای اینکه چارت به شکل دسته‌بندی شده رنگ‌آمیزی شود
                plot_df["Label"] = labels.astype(str)

            try:
                # تلاش برای رسم تعاملی با Plotly
                import plotly.express as px

                fig = px.scatter_3d(
                    plot_df,
                    x="t-SNE 1",
                    y="t-SNE 2",
                    z="t-SNE 3",
                    color="Label" if labels is not None else None,
                    opacity=alpha,
                    title="Interactive 3D t-SNE Visualization",
                    labels={"color": "Class"},
                )
                # تنظیمات ظاهری مارکرها
                fig.update_traces(marker=dict(size=5))
                fig.write_html("tsne_3d_result.html")
                print("نمودار با موفقیت در فایل tsne_3d_result.html ذخیره شد. آن را باز کنید.")
            except ImportError:
                # سیستم کمکی: اگر کاربر پلاتلی نداشت، با Matplotlib ثابت رسم میکنیم
                print(
                    "نکته: کتابخانه plotly نصب نبود. نمودار به صورت ۳بعدی ثابت (Matplotlib) رسم می‌شود."
                )
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")

                # ساخت دیکشنری رنگ برای تفکیک کلاس‌ها در مت‌پلات‌لیب
                if labels is not None:
                    unique_labels = np.unique(labels)
                    colors = plt.cm.get_cmap("tab10", len(unique_labels))
                    color_dict = {
                        lbl: colors(i) for i, lbl in enumerate(unique_labels)
                    }
                    c_vector = [color_dict[lbl] for lbl in labels]
                else:
                    c_vector = "blue"

                ax.scatter(
                    plot_df["t-SNE 1"],
                    plot_df["t-SNE 2"],
                    plot_df["t-SNE 3"],
                    c=c_vector,
                    alpha=alpha,
                    s=20,
                )

                ax.set_title("3D t-SNE Visualization (Static)")
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                ax.set_zlabel("t-SNE 3")
                plt.show()

        else:
            raise ValueError(
                "تعداد ابعاد (n_components) باید فقط ۲ یا ۳ باشد تا قابل رسم باشد."
            )