import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("1) iris.csv")

X = data[["sepal_length", "petal_length"]]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
data["Cluster"] = kmeans.labels_

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(
    data["sepal_length"],
    data["petal_length"],
    c=data["Cluster"]
)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker="X",
    s=200
)
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.show()