import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from preprocessing import load_data

CLASS_COLORS = {
    "HC": "tab:green",
    "Asthma": "tab:orange",
    "COPD": "tab:red",
    "Infected": "tab:purple",
}

def plot_2d(X_reduced, y, ratio, title, fname):
    plt.figure(figsize=(7, 6))
    for cls, color in CLASS_COLORS.items():
        mask = y == cls
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                    c=color, label=cls, alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.xlabel(f"PC1 ({ratio[0]*100:.1f}% дисперсии)")
    plt.ylabel(f"PC2 ({ratio[1]*100:.1f}% дисперсии)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_3d(X_reduced, y, ratio, title, fname):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for cls, color in CLASS_COLORS.items():
        mask = y == cls
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                   c=color, label=cls, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ratio[2]*100:.1f}%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def main():
    X, y, feature_names = load_data("Exasens.csv")

    pca2 = PCA(n_components=2)
    X2 = pca2.fit_transform(X)
    plot_2d(X2, y, pca2.explained_variance_ratio_,
            "PCA (sklearn), 2 компоненты — Exasens", "sklearn_pca_2d.png")

    pca3 = PCA(n_components=3)
    X3 = pca3.fit_transform(X)
    plot_3d(X3, y, pca3.explained_variance_ratio_,
            "PCA (sklearn), 3 компоненты — Exasens", "sklearn_pca_3d.png")

    pca_full = PCA(n_components=X.shape[1])
    pca_full.fit(X)
    all_ratio = pca_full.explained_variance_ratio_
    loss2 = 1 - all_ratio[:2].sum()
    loss3 = 1 - all_ratio[:3].sum()

    # MSE реконструкции
    X2_back = pca2.inverse_transform(X2)
    X3_back = pca3.inverse_transform(X3)
    mse2 = np.mean((X - X2_back) ** 2)
    mse3 = np.mean((X - X3_back) ** 2)

    print("Признаки:", feature_names)
    print("Собственные значения (explained_variance_, все компоненты):")
    for i, (v, r) in enumerate(zip(pca_full.explained_variance_, all_ratio), 1):
        print(f"  λ{i} = {v:.4f}  (доля дисперсии = {r*100:.2f}%)")
    print()
    print(f"[k=2] объяснённая дисперсия = {(1-loss2)*100:.2f}%  |  "
          f"потеря = {loss2*100:.2f}%  |  MSE реконструкции = {mse2:.4f}")
    print(f"[k=3] объяснённая дисперсия = {(1-loss3)*100:.2f}%  |  "
          f"потеря = {loss3*100:.2f}%  |  MSE реконструкции = {mse3:.4f}")

if __name__ == "__main__":
    main()