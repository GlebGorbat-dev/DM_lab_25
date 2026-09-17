import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # для 3D-графиков
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CSV_PATH = r'D:\Учёба\4 курс\ИАД\1\hcvdat0.csv'

df = pd.read_csv(CSV_PATH, index_col=0)

print("Исходный размер данных:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nТипы данных:")
print(df.dtypes)
print("\nКолонки:", list(df.columns))

target = df['Category'].copy()

X = df.drop(columns=['Category']).copy()

X['Sex'] = X['Sex'].map({'m': 0, 'f': 1})

print("\nПропуски до обработки:")
print(X.isna().sum())

X = X.fillna(X.median(numeric_only=True))

print("\nПропуски после обработки:")
print(X.isna().sum().sum(), "пропусков осталось")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nРазмер данных после предобработки:", X_scaled.shape)

# Ручной PCA
def pca_manual(data, n_components=2):
    data_centered = data - data.mean(axis=0)
    
    cov_matrix = np.cov(data_centered.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    components = eigenvectors[:, :n_components]
    
    data_reduced = data_centered.dot(components)
    
    return data_reduced, eigenvalues, eigenvectors

X_manual_2, eigenvalues, eigenvectors = pca_manual(X_scaled, n_components=2)

X_manual_3 = (X_scaled - X_scaled.mean(axis=0)).dot(eigenvectors[:, :3])

print("\nРучной PCA выполнен.")
print("Первые 5 собственных значений:", np.round(eigenvalues[:5], 4))

# PCA через sklearn
pca_sklearn_2 = PCA(n_components=2)
X_sklearn_2 = pca_sklearn_2.fit_transform(X_scaled)

pca_sklearn_3 = PCA(n_components=3)
X_sklearn_3 = pca_sklearn_3.fit_transform(X_scaled)

print("\nsklearn PCA выполнен.")
print("Объяснённая дисперсия (2 компоненты):", 
      np.round(pca_sklearn_2.explained_variance_ratio_, 4))
print("Сумма объяснённой дисперсии (2 компоненты):", 
      round(pca_sklearn_2.explained_variance_ratio_.sum(), 4))

# Визуализация
classes = target.unique()
colors = ['blue', 'green', 'red', 'orange', 'purple']
markers = ['o', 's', '^', 'D', 'v']

# Визуализация 2 компонент (ручной PCA)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, cls in enumerate(classes):
    mask = (target == cls).values
    axes[0].scatter(X_manual_2[mask, 0], X_manual_2[mask, 1],
                    c=colors[i % len(colors)], marker=markers[i % len(markers)],
                    label=cls, alpha=0.7, edgecolors='k', linewidths=0.3)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('Ручной PCA (numpy.linalg.eig) — 2 компоненты')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Визуализация 2 компонент (sklearn PCA)
for i, cls in enumerate(classes):
    mask = (target == cls).values
    axes[1].scatter(X_sklearn_2[mask, 0], X_sklearn_2[mask, 1],
                    c=colors[i % len(colors)], marker=markers[i % len(markers)],
                    label=cls, alpha=0.7, edgecolors='k', linewidths=0.3)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('sklearn PCA — 2 компоненты')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_2d.png', dpi=150)
plt.show()

# Визуализация 3 компонент (3D)
fig = plt.figure(figsize=(16, 7))

# 3D для ручного PCA
ax1 = fig.add_subplot(121, projection='3d')
for i, cls in enumerate(classes):
    mask = (target == cls).values
    ax1.scatter(X_manual_3[mask, 0], X_manual_3[mask, 1], X_manual_3[mask, 2],
                c=colors[i % len(colors)], marker=markers[i % len(markers)],
                label=cls, alpha=0.7, edgecolors='k', linewidths=0.3)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.set_zlabel('PC3')
ax1.set_title('Ручной PCA — 3 компоненты')
ax1.legend(fontsize=7)

# 3D для sklearn PCA
ax2 = fig.add_subplot(122, projection='3d')
for i, cls in enumerate(classes):
    mask = (target == cls).values
    ax2.scatter(X_sklearn_3[mask, 0], X_sklearn_3[mask, 1], X_sklearn_3[mask, 2],
                c=colors[i % len(colors)], marker=markers[i % len(markers)],
                label=cls, alpha=0.7, edgecolors='k', linewidths=0.3)
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.set_zlabel('PC3')
ax2.set_title('sklearn PCA — 3 компоненты')
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig('pca_3d.png', dpi=150)
plt.show()

# Подсчёт потерь информативности
full_info = eigenvalues.sum()

# Потери при 2 компонентах
reduced_info_2 = eigenvalues[:2].sum()
loss_2 = (1 - reduced_info_2 / full_info) * 100

# Потери при 3 компонентах
reduced_info_3 = eigenvalues[:3].sum()
loss_3 = (1 - reduced_info_3 / full_info) * 100

print("Потери информативности")
print(f"Полная информация (сумма всех собств. значений): {full_info:.4f}")
print(f"Информация в 2 компонентах: {reduced_info_2:.4f}")
print(f"Информация в 3 компонентах: {reduced_info_3:.4f}")
print(f"\nПотери при 2 компонентах: {loss_2:.2f}%")
print(f"Потери при 3 компонентах: {loss_3:.2f}%")

# Проверка через sklearn (должно совпасть с ручным расчётом)
print("\nПроверка через sklearn:")
print(f"Потери при 2 компонентах: {(1 - pca_sklearn_2.explained_variance_ratio_.sum())*100:.2f}%")
print(f"Потери при 3 компонентах: {(1 - pca_sklearn_3.explained_variance_ratio_.sum())*100:.2f}%")

# Доля объяснённой дисперсии по каждой компоненте
print("\nДоля объяснённой дисперсии по компонентам:")
for i, var in enumerate(pca_sklearn_3.explained_variance_ratio_, 1):
    print(f"  PC{i}: {var*100:.2f}%")