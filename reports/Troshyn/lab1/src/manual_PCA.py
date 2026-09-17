import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_data

CLASS_COLORS = {
    "HC": "tab:green",
    "Astma": "tab:orange",
    "COPD": "tab:red",
    "Infected": "tab:purple",
}

def pca_manual(X: np.ndarray, n_components: int):
    n_samples = X.shape[0]

    cov = (X.T @ X) / (n_samples - 1) #матрица ковариации признаков
    eigvals, eigvecs = np.linalg.eig(cov) #собственные значения, собственные вектора
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    #сортировка по убыванию собственных значений
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
 
    components = eigvecs[:, :n_components]
    X_reduced = X @ components
 
    return X_reduced, eigvals, components

def explained_variance_report(eigvals: np.ndarray, k: int):
    total = eigvals.sum()
    explained = eigvals[:k].sum()
    ratio_per_component = eigvals / total
    loss = 1 - explained / total
    return ratio_per_component, loss

def reconstruction_error(X, components, X_reduced):
    #MSE между исходными (стандартизированными) данными и их реконструкцией из k главных компонент: X_approx = X_reduced @ components^T.
    X_approx = X_reduced @ components.T
    mse = np.mean((X - X_approx) ** 2)
    return mse

def plot_2d(X_reduced, y, eigvals, title, fname):
    plt.figure(figsize=(7, 6))
    for cls, color in CLASS_COLORS.items():
        mask = y == cls
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                    c=color, label=cls, alpha=0.7, edgecolors="k", linewidths=0.3)
    var_ratio = eigvals / eigvals.sum()
    plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}% дисперсии)")
    plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}% дисперсии)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
 
 
def plot_3d(X_reduced, y, eigvals, title, fname):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for cls, color in CLASS_COLORS.items():
        mask = y == cls
        ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                   c=color, label=cls, alpha=0.7, edgecolors="k", linewidths=0.3)
    var_ratio = eigvals / eigvals.sum()
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def main():
    X, y, feature_names = load_data("Exasens.csv")

    X2, eigvals, comp2 = pca_manual(X, 2)
    ratio, loss2 = explained_variance_report(eigvals, 2)
    mse2 = reconstruction_error(X, comp2, X2)
    plot_2d(X2, y, eigvals, "PCA (numpy.linalg.eig), 2 компоненты — Exasens", "manual_pca_2d.png")

    X3, eigvals3, comp3 = pca_manual(X, 3)
    _, loss3 = explained_variance_report(eigvals3, 3)
    mse3 = reconstruction_error(X, comp3, X3)
    plot_3d(X3, y, eigvals3, "PCA (numpy.linalg.eig), 3 компоненты — Exasens", "manual_pca_3d.png")

    print("Признаки:", feature_names)
    print("Собственные значения:")
    for i, v in enumerate(eigvals, 1):
        print(f"  λ{i} = {v:.4f}  (доля дисперсии = {ratio[i-1]*100:.2f}%)")
    print()
    print(f"[k=2] объяснённая дисперсия = {(1-loss2)*100:.2f}%  |  "
          f"потеря = {loss2*100:.2f}%  |  MSE реконструкции = {mse2:.4f}")
    print(f"[k=3] объяснённая дисперсия = {(1-loss3)*100:.2f}%  |  "
          f"потеря = {loss3*100:.2f}%  |  MSE реконструкции = {mse3:.4f}")

if __name__ == "__main__":
    main()
