---
reading_minutes: 6
objectives:
  - "Describe t-SNE as a neighbour-preserving 2D embedding — what it shows (local structure) and what it deliberately distorts (cluster sizes and inter-cluster distances)."
  - "Pick a `perplexity` appropriate to dataset size and read `kl_divergence_` after fitting to spot unconverged runs."
  - "Run `TSNE(n_components=2, perplexity=..., random_state=...).fit_transform(X)` on scaled data and visualise the embedding."
  - "Avoid common over-interpretations and remember that sklearn's `TSNE` has no `transform`, so new points cannot be embedded without refitting."
---

# t-SNE (t-Distributed Stochastic Neighbor Embedding)

**After this lesson:** you can explain the core ideas in “t-SNE (t-Distributed Stochastic Neighbor Embedding)” and reproduce the examples here in your own notebook or environment.

## Overview

Focus on **t-SNE** mechanics and interpretation—neighbor preservation, not cluster sizes or distances between clusters.

## Helpful video

StatQuest overview of t-SNE.

<iframe width="560" height="315" src="https://www.youtube.com/embed/NEaUSP4YerM" title="StatQuest: t-SNE, Clearly Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Quick Reference

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/t-sne-1.mmd" %}

t-SNE is ideal when:
- You need to visualize high-dimensional data
- Preserving local relationships matters
- You want to identify clusters visually
- Data has complex non-linear structure

```python
from sklearn.manifold import TSNE

# Basic usage
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

# Visualize
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
```

For the complete tutorial, see [t-SNE and UMAP Guide](tsne-umap.md).

## Gotchas

- **Distances between clusters are not interpretable** — t-SNE optimizes local neighborhood preservation, so two clusters sitting close together in a t-SNE plot does not mean they are similar in the original high-dimensional space.
- **No `transform` for new points** — sklearn's `TSNE` cannot embed unseen data; the entire dataset must be re-embedded together. If you need to embed new points, use UMAP which supports `transform`.
- **Perplexity interacts with dataset size** — perplexity must be less than `n_samples`; on small datasets (fewer than ~50 points), even `perplexity=30` is too large and the algorithm will silently produce poor embeddings.
- **Low `n_iter` can produce an unconverged embedding** — the default 1000 iterations is usually enough, but on large datasets the KL divergence may still be decreasing. Monitor `kl_divergence_` after fitting to confirm convergence.
- **t-SNE amplifies cluster appearance** — local neighborhood optimization tends to produce tight, separated blobs even when the underlying data has no clear cluster structure, making it easy to over-interpret faint groupings.
