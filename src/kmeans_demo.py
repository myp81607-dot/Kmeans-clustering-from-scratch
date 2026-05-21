import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Part 1: K-means from scratch
def initialize_centroids(X, k, random_state=42):
    rng = np.random.RandomState(random_state)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].copy()

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            centroids.append(np.zeros(X.shape[1]))
        else:
            centroids.append(cluster_points.mean(axis=0))
    return np.array(centroids)

def compute_inertia(X, labels, centroids):
    inertia = 0.0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        inertia += np.sum((cluster_points - centroids[i]) ** 2)
    return inertia


def kmeans_from_scratch(X, k, max_iter=100, tol=1e-4, random_state=42):
    centroids = initialize_centroids(X, k, random_state=random_state)
    for it in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            return labels, centroids, it + 1
    return labels, centroids, max_iter

# Part 2: Compare with scikit-learn
X1, _ = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=42)

scaler1 = StandardScaler()
X1_std = scaler1.fit_transform(X1)
labels_scratch, centroids_scratch, n_iter_scratch = kmeans_from_scratch(X1_std, k=3, max_iter=100, tol=1e-4, random_state=42)
kmeans_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_sklearn = kmeans_sklearn.fit_predict(X1_std)
centroids_sklearn = kmeans_sklearn.cluster_centers_

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X1_std[:, 0], X1_std[:, 1], c=labels_scratch, cmap='viridis', s=30, label='Samples')
axes[0].scatter(centroids_scratch[:, 0], centroids_scratch[:, 1],c='red', marker='X', s=200, label='Centroids')
axes[0].set_title('Part 2: From Scratch')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].legend()

axes[1].scatter(X1_std[:, 0], X1_std[:, 1], c=labels_sklearn, cmap='viridis', s=30, label='Samples')
axes[1].scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1],c='red', marker='X', s=200, label='Centroids')
axes[1].set_title('Part 2: scikit-learn')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].legend()

plt.tight_layout()
plt.savefig('part2_compare_kmeans.png', dpi=200)
plt.show()

print("Part 2")
print("Scratch iterations:", n_iter_scratch)
print("Scratch centroids:\n", centroids_scratch)
print("Sklearn centroids:\n", centroids_sklearn)

analysis_part2 = (
    "Both methods optimize the same K-means objective, so their clustering results "
    "are expected to be nearly identical up to label permutation and centroid ordering. "
    "With the same standardized data and random seed, the two plots should look very similar, "
    "and any small difference is usually caused by initialization details or stopping criteria.")
print("\nAnalysis Part 2:\n", analysis_part2)

# Part 3: Optimal number of clusters
X2, _ = make_blobs(
    n_samples=500,
    centers=5,
    n_features=2,
    random_state=42)

scaler2 = StandardScaler()
X2_std = scaler2.fit_transform(X2)

# Elbow method
ks_inertia = range(1, 11)
inertias = []

for k in ks_inertia:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X2_std)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(ks_inertia), inertias, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.tight_layout()
plt.savefig('part3_elbow.png', dpi=200)
plt.show()

# Silhouette scores
ks_sil = range(2, 11)
sil_scores = []

for k in ks_sil:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X2_std)
    sil_scores.append(silhouette_score(X2_std, labels))

plt.figure(figsize=(6, 4))
plt.plot(list(ks_sil), sil_scores, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Scores')
plt.tight_layout()
plt.savefig('part3_silhouette_scores.png', dpi=200)
plt.show()

# Silhouette plots for k=3, 5, 7
for k in [3, 5, 7]:
    fig, ax = plt.subplots(figsize=(7, 5))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X2_std)
    sample_sil_values = silhouette_samples(X2_std, cluster_labels)
    y_lower = 10
    for i in range(k):
        ith_cluster_sil_values = sample_sil_values[cluster_labels == i]
        ith_cluster_sil_values.sort()
        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    avg_score = silhouette_score(X2_std, cluster_labels)
    ax.axvline(x=avg_score, color="red", linestyle="--")
    ax.set_title(f'Silhouette Plot for k={k}')
    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster label')
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'part3_silhouette_k_{k}.png', dpi=200)
    plt.show()

# Final choice
optimal_k = 5
km_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_opt = km_opt.fit_predict(X2_std)
centroids_opt = km_opt.cluster_centers_
plt.figure(figsize=(6, 5))
plt.scatter(X2_std[:, 0], X2_std[:, 1], c=labels_opt, cmap='viridis', s=30, label='Samples')
plt.scatter(centroids_opt[:, 0], centroids_opt[:, 1],c='red', marker='X', s=220, label='Centroids')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'Optimal K-means Clustering (k={optimal_k})')
plt.legend()
plt.tight_layout()
plt.savefig('part3_optimal_kmeans.png', dpi=200)
plt.show()
analysis_part3 = (
    "The elbow curve shows a clear bend around k=5, after which the inertia decreases more slowly. "
    "The silhouette scores also favor k=5 or a nearby value, and the silhouette plots for k=3, 5, and 7 "
    "show that k=5 gives the cleanest separation without merging true groups or splitting them unnecessarily. "
    "Therefore, k=5 is the most reasonable choice."
)

print("Analysis Part 3:\n", analysis_part3)
