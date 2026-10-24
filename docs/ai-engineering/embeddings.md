---
title: Embeddings & Vector Search
description: How embedding models are trained, similarity measures, bi-encoders vs cross-encoders, choosing an embedding model, approximate nearest-neighbor indexes (HNSW, IVF, PQ), vector databases, and memory math.
act: 4
priority: Critical
tags:
  - AI Engineer
  - Agentic AI Engineer
  - ML Engineer
---

# Embeddings & Vector Search

!!! abstract "TL;DR"
    An embedding model maps text (or images) to vectors so that similar meaning means nearby vectors. Models are trained **contrastively**: pull matching pairs together, push others apart. **Bi-encoders** embed queries and documents separately (fast, indexable); **cross-encoders** read them together (accurate, slow), so they rerank a shortlist. Searching millions of vectors needs **approximate nearest-neighbor** indexes: **HNSW** (graph, high recall, memory-heavy), **IVF** (clusters), **PQ** (compression). Choose models by measuring recall on your own data, and plan for re-embedding when you change models.

!!! evolution "Where this fits"
    **Before:** [word2vec](../deep-learning/sequence-models.md), [BERT](../llms/pretraining.md), [contrastive learning](../deep-learning/cnn-vision.md), [kNN](../classical-ml/knn-nb-svm.md). **This page:** semantic search, the retrieval half of RAG. **Next:** [RAG](rag.md).

## What an embedding is

A fixed-length vector (typically 256–4,096 numbers) summarizing a piece of content. Keyword search fails on "cheap flights" vs "low-cost airfare"; their embeddings sit close together.

**How modern text embedders are trained:**

1. Start from a pretrained encoder (or a decoder LLM adapted for embedding).
2. **Contrastive training** on pairs that should match: question–answer, query–clicked document, title–body, paraphrases, often hundreds of millions of weakly supervised pairs, then curated high-quality pairs.
3. **InfoNCE loss with in-batch negatives:** each query should score its own positive higher than every other document in the batch. Larger batches mean more negatives and better embeddings.
4. **Hard negatives:** documents that look relevant but aren't, mined with BM25 or an earlier model, sharpen the boundary.
5. **Instruction-tuned embedders** take a task prefix ("query: …", "Represent this question for retrieving answers: …"); using the wrong prefix lowers quality.
6. **Matryoshka representation learning:** train so the first $k$ dimensions are a good embedding on their own, allowing truncation (for example 1,024 → 256) to save memory.

| Representation | How | Strength | Weakness |
|---|---|---|---|
| **Dense** (bi-encoder) | One vector per text | Semantic matching, paraphrase | Weak on exact terms, IDs, rare names |
| **Sparse lexical** (BM25) | Term weights over the vocabulary | Exact matches, codes, names | No semantics |
| **Learned sparse** (SPLADE) | A model predicts weights for vocabulary terms, including expansions | Semantics with an inverted index | More complex |
| **Late interaction** (ColBERT) | One vector per token; score sums the best token matches | Fine-grained relevance | Much more storage |

## Similarity measures

| Measure | Formula | Notes |
|---|---|---|
| **Cosine similarity** | $\frac{a \cdot b}{\|a\|\|b\|}$ | Compares direction only; the usual default |
| **Dot product** | $a \cdot b$ | Same ranking as cosine if vectors are normalized; some models are trained for raw dot product |
| **Euclidean (L2) distance** | $\|a - b\|$ | For unit vectors, $\|a - b\|^2 = 2 - 2\cos\theta$, so it ranks like cosine |

Use the measure the model was trained with, and normalize consistently at indexing and query time.

## Bi-encoders vs cross-encoders

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | Query and document embedded **separately** | Query and document **concatenated**, read together |
| Output | Two vectors; score = similarity | One relevance score |
| Precompute documents? | Yes: index once | No: run for every pair at query time |
| Speed | Milliseconds over millions of documents (with ANN) | Tens of milliseconds per pair |
| Accuracy | Good | Better: full attention between query and document |
| Role | First-stage retrieval | **Reranking** the top 20–100 candidates |

The standard pattern: retrieve broadly with a bi-encoder (plus BM25), then rerank the shortlist with a cross-encoder or an LLM.

## Choosing an embedding model

1. **Shortlist** by the MTEB leaderboard's retrieval scores, language coverage, maximum input length, dimension, licence and hosting (API vs open weights).
2. **Evaluate on your own data:** build 100–500 real queries with known relevant documents and measure recall@k and MRR. Leaderboard rank often doesn't transfer to your domain.
3. **Weigh cost and latency:** embedding the corpus once and every query thereafter; dimension drives storage and search cost.
4. **Consider fine-tuning** the embedder on your domain's query–document pairs if recall is the bottleneck.
5. **Plan for change:** embeddings from different models aren't comparable, so switching models means re-embedding everything. Version the model alongside the index.

## Approximate nearest-neighbor search

Exact search compares the query with every vector: $O(nd)$ per query. Fine for thousands of vectors, too slow for hundreds of millions. ANN indexes trade a little recall for orders of magnitude in speed.

| Index | Idea | Key parameters | Trade-offs |
|---|---|---|---|
| **Flat (exact)** | Brute force | — | Perfect recall; slow at scale; a good baseline |
| **IVF** (inverted file) | Cluster vectors with k-means; search only the `nprobe` nearest clusters | `nlist`, `nprobe` | Fast, compact; recall drops for points near cluster edges; needs training |
| **HNSW** | A multi-layer proximity graph; greedy search from sparse top layers down to the dense bottom layer | `M` (links per node), `efConstruction`, `efSearch` | Excellent recall and latency; high memory; slower inserts; deletions are awkward |
| **PQ** (product quantization) | Split each vector into $m$ sub-vectors and replace each by the ID of its nearest centroid (1 byte each) | $m$, codebook size | Huge memory savings; lower precision (rerank with full vectors) |
| **IVF-PQ** | IVF clusters plus PQ codes | All of the above | Billion-scale on modest memory |
| **DiskANN / ScaNN** | Disk-resident graphs; anisotropic quantization | Library-specific | Very large scale, cost-efficient |

**Tuning:** raising `efSearch` (HNSW) or `nprobe` (IVF) increases recall and latency. Measure **recall@k against exact search** on sample queries, then choose the fastest setting meeting your recall target.

**Filtering:** "documents from this tenant, after 2024" combines metadata filters with vector search. **Post-filtering** (search, then filter) can return too few results; **pre-filtering** or filter-aware indexes keep recall. Multi-tenant systems often use a namespace or index per tenant.

??? example "Memory math for 10 million vectors"
    At 1,024 dimensions in float32: $10^7 \times 1{,}024 \times 4$ bytes ≈ **41 GB** (plus HNSW graph links, often 10–30% more).

    - float16 or int8: about 20 GB or 10 GB.
    - Matryoshka truncation to 256 dimensions in float32: about 10 GB.
    - PQ with 64 one-byte codes per vector: about **640 MB**, with full vectors kept on disk for reranking.
    - Binary quantization (one bit per dimension): 128 bytes per vector, about 1.3 GB, usually followed by rescoring.

## Vector databases

| Option | When it fits |
|---|---|
| **FAISS, hnswlib** (libraries) | You manage storage and serving yourself; research and batch jobs |
| **pgvector** (Postgres) | You already run Postgres; up to tens of millions of vectors; transactions and joins with metadata |
| **Elasticsearch / OpenSearch** | You need BM25 and vectors together (hybrid search) with mature operations |
| **Dedicated vector databases** (Pinecone, Weaviate, Milvus, Qdrant, Chroma) | Large scale, managed scaling, built-in filtering, hybrid search, multi-tenancy |

Choose on scale, filtering needs, hybrid search, operations burden, latency, cost and data residency, not on hype. Many systems never need more than Postgres.

## Other uses of embeddings

Clustering and topic discovery, deduplication, classification features (embeddings plus logistic regression is a strong baseline), recommendation (two-tower retrieval), semantic caching of LLM responses, anomaly detection, and multimodal search with CLIP-style image–text embeddings.

## Interview questions

??? question "Bi-encoder vs cross-encoder: what's the difference and when do you use each?"
    A bi-encoder embeds query and document independently, so documents are embedded once and searched with a vector index in milliseconds; it's the first-stage retriever. A cross-encoder feeds query and document together through one model, so every query token attends to every document token, giving more accurate relevance scores but requiring a forward pass per pair; it's used to rerank a shortlist of 20–100 candidates. Together: fast recall, then precise ordering.

??? question "How are embedding models trained?"
    Contrastively: given pairs that should match (query and relevant passage, question and answer, paraphrases), the model learns to give each anchor a higher similarity to its positive than to negatives, using an InfoNCE loss over in-batch negatives plus mined hard negatives. Training usually goes from large, noisy weakly supervised pairs to smaller, high-quality labeled sets; many models also use task instructions as prefixes and Matryoshka objectives so embeddings can be truncated.

??? question "Why is cosine similarity the usual choice?"
    It compares direction and ignores vector length, which often reflects things like text length or term frequency rather than meaning. Most text embedding models are trained with cosine or normalized dot-product similarity. If vectors are normalized to unit length, cosine, dot product and Euclidean distance all produce the same ranking, so you can normalize once and use the fastest operation.

??? question "How does HNSW work?"
    It builds a hierarchy of proximity graphs: the bottom layer contains all vectors, each linked to about M near neighbors, and each higher layer contains an exponentially smaller random subset. A search starts at the top layer, greedily walks toward the query, drops to the next layer from the best node found, and repeats, keeping a candidate list of size efSearch at the bottom. It reaches high recall with roughly logarithmic search cost, but stores many links per vector (memory-heavy) and handles deletions poorly.

??? question "IVF vs HNSW: trade-offs."
    IVF clusters vectors with k-means and searches the closest few clusters: compact, fast to build, pairs naturally with PQ compression for billion-scale data, but recall suffers for queries near cluster boundaries unless nprobe is raised, and it needs a training step. HNSW gives better recall–latency trade-offs at moderate scale and needs no training, but uses more memory and is slower to build and update. Use HNSW for high-recall, in-memory workloads; IVF-PQ when memory is the constraint.

??? question "What is product quantization?"
    A compression scheme: split each vector into m sub-vectors, run k-means (typically 256 centroids) separately on each sub-space, and store each sub-vector as the one-byte ID of its nearest centroid, so a vector becomes m bytes. Distances are computed approximately with precomputed lookup tables of query-to-centroid distances. It cuts memory by 10–100×; to recover precision, rerank the top candidates with full-precision vectors.

??? question "How do you choose an embedding model?"
    Shortlist by retrieval benchmarks (MTEB), languages, maximum sequence length, dimension, license and hosting, then evaluate on a test set from your own domain: real queries labeled with relevant documents, measuring recall@k and MRR, plus latency and cost. Check that it handles your content (code, tables, jargon, multilingual). If none is good enough, fine-tune one on domain pairs. Version it, because switching models means re-embedding the corpus.

??? question "Your dense retrieval misses queries with product codes and names. Why, and what do you do?"
    Dense embeddings capture meaning but compress away exact surface details, so rare tokens like SKUs, error codes and names often don't match well. Add **lexical retrieval** (BM25) and merge results (hybrid search with reciprocal rank fusion), use metadata filters for structured attributes, or use learned sparse retrieval. A cross-encoder reranker then orders the merged candidates.

??? question "How much memory do 100 million 768-dimensional float32 embeddings need, and how would you reduce it?"
    $10^8 \times 768 \times 4$ bytes ≈ 307 GB, plus index overhead. Reduce it with float16 or int8 (2–4× smaller), Matryoshka truncation to fewer dimensions, product quantization (for example 64 bytes per vector, about 6.4 GB) with full vectors on disk for reranking, binary quantization with rescoring, or disk-based indexes such as DiskANN. Shard across machines if needed.

??? question "Pre-filtering vs post-filtering in vector search."
    Post-filtering runs the ANN search and then drops results that fail metadata conditions; if the filter is selective, few or none of the top-k survive. Pre-filtering applies the filter first and searches only matching vectors, which is accurate but can be slow if done by brute force. Modern vector databases implement filter-aware search inside the index. For strict separation (tenants, permissions), partition indexes or namespaces instead.

??? question "Why can't you mix embeddings from two different models in one index?"
    Each model defines its own vector space; dimensions don't mean the same thing across models (even with equal sizes), so similarities between vectors from different models are meaningless. Upgrading the model requires re-embedding the whole corpus, usually by building a new index in parallel, evaluating it, and switching over. Store the model version with each vector.

!!! pitfall "Common mistakes"
    - Picking an embedding model from a leaderboard without testing on your data.
    - Forgetting query/document prefixes for instruction-tuned embedders.
    - Using dense retrieval alone for code, IDs and names.
    - Post-filtering with selective filters and getting empty results.
    - Not planning for re-embedding when the model changes.

## Revision checklist

- [ ] Explain contrastive training with in-batch and hard negatives
- [ ] Compare dense, sparse, learned-sparse and late-interaction retrieval
- [ ] Compare bi-encoders and cross-encoders and where each sits in a pipeline
- [ ] Explain HNSW, IVF and PQ, their parameters and trade-offs
- [ ] Do vector-index memory math and name four ways to shrink it
- [ ] Choose between pgvector, a search engine and a vector database

## Go deeper

- [Sentence-BERT (2019)](https://arxiv.org/abs/1908.10084)
- [HNSW (2016)](https://arxiv.org/abs/1603.09320)
- [FAISS documentation](https://github.com/facebookresearch/faiss/wiki)
- [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [Matryoshka Representation Learning (2022)](https://arxiv.org/abs/2205.13147)
