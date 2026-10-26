# Enterprise AI & Retrieval-Augmented Generation Handbook

## 1. Introduction to Enterprise Search
Modern enterprise applications require generative models to be grounded in authoritative corporate data. Traditional dense-only retrieval systems face severe limitations:
- Dense Vector Collapse: Out-of-vocabulary terms, specific error codes, serial numbers, and custom jargon suffer from poor embedding alignment.
- Hallucination Propagation: When context documents contain ambiguous or tangentially related passages, standard LLMs hallucinate plausible-sounding falsehoods.

## 2. Hybrid Retrieval Architecture
To overcome these limitations, Nexus adopts a two-pronged retrieval strategy:
1. Dense Semantic Indexing: Captures conceptual intent, paraphrase variations, and cross-lingual meaning.
2. Sparse Lexical Indexing (BM25): Ensures exact token matching, preserving precision on critical technical identifiers.

### Reciprocal Rank Fusion (RRF)
Results from both dense and sparse indices are fused using Reciprocal Rank Fusion:
RRF_Score(d) = sum( weight_i / (k + rank_i(d)) )
where k is traditionally set to 60 to prevent individual top-ranked outliers from dominating the scoring distribution.

## 3. Corrective RAG (CRAG) Agent Loop
Rather than naively feeding retrieved chunks to the generator, Nexus executes an iterative self-correction loop:
- Step A (Query Reformulation): The user prompt is enriched with synonyms and query expansions.
- Step B (Relevance Verification): Retrieved candidate chunks are graded by a lightweight filter. Chunks scoring below the threshold are pruned.
- Step C (Grounded Generation): Constrained prompts ensure claims reference specific chunk IDs.
- Step D (Factuality Auditing): A post-generation verification step computes token alignment between the answer assertions and source context.
