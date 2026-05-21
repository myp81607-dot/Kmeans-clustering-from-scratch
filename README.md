# Kmeans-clustering-from-scratch
Python implementation of K-means from scratch with scikit-learn comparison, elbow method, silhouette analysis, and visualization.
# K-means Clustering from Scratch

This project implements K-means clustering from scratch in Python and compares it with scikit-learn's `KMeans` implementation. It also uses the elbow method and silhouette analysis to select a reasonable number of clusters.

## Project Overview

K-means is a classic unsupervised learning algorithm for clustering unlabeled data.  
The goal of this project is to understand the full K-means workflow from both algorithmic and experimental perspectives.

In this project, I implemented the core K-means algorithm using NumPy, compared the results with scikit-learn, and evaluated different choices of cluster number using inertia and silhouette scores.

## What I Implemented

- K-means from scratch using NumPy
- Random centroid initialization
- Euclidean-distance-based cluster assignment
- Centroid update by cluster mean
- Convergence checking based on centroid movement
- Comparison with scikit-learn `KMeans`
- Elbow method for selecting the number of clusters
- Silhouette score analysis
- Silhouette plots for different values of k
- Final clustering visualization with selected `k = 5`

## Repository Structure

```text
kmeans-clustering-from-scratch/
├── README.md
├── requirements.txt
├── src/
│   └── kmeans_demo.py
├── figures/
│   ├── part2_compare_kmeans.png
│   ├── part3_elbow.png
│   ├── part3_silhouette_scores.png
│   ├── part3_silhouette_k_3.png
│   ├── part3_silhouette_k_5.png
│   ├── part3_silhouette_k_7.png
│   └── part3_optimal_kmeans.png
└── reports/
    └── Kmeans_Report.pdf
