# Title Page

**New Submission to IEEE Transactions on Artificial Intelligence**

---

## Title

PCA-Matryoshka: Enabling Effective Dimension Reduction for Non-Matryoshka Embedding Models with Applications to Vector Database Compression

## Author

Andrew H. Bond, Senior Member, IEEE

## Affiliation

Department of Computer Engineering
San Jose State University
San Jose, CA 95192, USA

## Contact

Email: andrew.bond@sjsu.edu

## Abstract

Matryoshka Representation Learning enables embedding models to produce vectors whose leading dimensions form useful lower-dimensional representations, allowing simple truncation for compression. However, this property requires specialized multi-scale training losses and is absent from the majority of deployed embedding models. We introduce PCA-Matryoshka, a training-free technique that applies a principal component analysis (PCA) rotation to any embedding model's output, reordering dimensions by explained variance so that truncation becomes effective without retraining. On BGE-M3 embeddings (1024 dimensions, not trained with Matryoshka loss), naive truncation to 256 dimensions yields a cosine similarity of 0.467 against the full-dimensional representation -- effectively unusable. PCA-Matryoshka truncation to 256 dimensions achieves 0.974 cosine similarity, a 109% improvement. Combined with 3-bit scalar quantization, PCA-Matryoshka delivers 27x compression at 0.979 cosine similarity and 76.4% recall@10 on a 2.4-million-vector cross-civilizational ethics corpus. We provide a comprehensive empirical comparison against product quantization, binary embeddings, scalar quantization, and native Matryoshka truncation across 15 compression configurations.

## Keywords

Embedding compression, dimensionality reduction, Matryoshka representations, vector databases, principal component analysis, quantization

---

**Corresponding Author:**
Andrew H. Bond
Department of Computer Engineering
San Jose State University
San Jose, CA 95192, USA
Email: andrew.bond@sjsu.edu
IEEE Member Number: 01054659
ORCID: 0009-0003-2599-6158
