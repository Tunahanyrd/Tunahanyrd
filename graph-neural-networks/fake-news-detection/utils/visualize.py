import matplotlib.pyplot as plt
import umap

def plot_umap(embeddings, labels, desc=" "):
    reduced = umap.UMAP(n_neighbors=30, min_dist=0.1).fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="coolwarm", s=10)
    plt.title(f"GAT Encoder Embedding {desc}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.show()  