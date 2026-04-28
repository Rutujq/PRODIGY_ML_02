import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# -----------------------------
# Select features
# -----------------------------
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("elbow.png")   # SAVE IMAGE
plt.show()

# -----------------------------
# Apply KMeans
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# -----------------------------
# 3D Visualization (like your image)
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X.iloc[:, 0],  # Age
    X.iloc[:, 1],  # Income
    X.iloc[:, 2],  # Spending
    c=y_kmeans
)

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income")
ax.set_zlabel("Spending Score")

plt.title("Customer Segmentation (3D)")
plt.savefig("clusters_3d.png")   # SAVE IMAGE
plt.show()