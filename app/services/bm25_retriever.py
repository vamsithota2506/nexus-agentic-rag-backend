import math
import re
from typing import List, Dict, Tuple
from collections import Counter


class BM25Retriever:
    """
    Okapi BM25 Sparse Search Engine with Robertson-Spärck Jones IDF
    Provides fast, exact-match keyword retrieval complementing dense semantic embeddings.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_term_freqs: Dict[str, Counter] = {}
        self.inverted_index: Dict[str, List[str]] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z0-9_-]+\b", text.lower())

    def index_documents(self, documents: List[Tuple[str, str]]):
        """
        Index documents where each element is a tuple: (chunk_id, content)
        """
        self.corpus_size = len(documents)
        total_len = 0
        self.doc_lengths.clear()
        self.doc_term_freqs.clear()
        self.inverted_index.clear()
        self.idf.clear()

        doc_frequency = Counter()

        for chunk_id, content in documents:
            tokens = self._tokenize(content)
            doc_len = len(tokens)
            self.doc_lengths[chunk_id] = doc_len
            total_len += doc_len

            term_counts = Counter(tokens)
            self.doc_term_freqs[chunk_id] = term_counts

            unique_terms = set(tokens)
            for term in unique_terms:
                doc_frequency[term] += 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(chunk_id)

        self.avg_doc_len = (total_len / self.corpus_size) if self.corpus_size > 0 else 0.0

        # Calculate Robertson-Spärck Jones Inverse Document Frequency (IDF)
        for term, df in doc_frequency.items():
            # Standard smoothed BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1.0)
            idf_score = math.log(((self.corpus_size - df + 0.5) / (df + 0.5)) + 1.0)
            self.idf[term] = max(idf_score, 0.01)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns list of tuples: (chunk_id, bm25_score) sorted descending by relevance.
        """
        if self.corpus_size == 0 or self.avg_doc_len == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            term_idf = self.idf.get(token, 0.0)
            matching_chunks = self.inverted_index[token]

            for chunk_id in matching_chunks:
                tf = self.doc_term_freqs[chunk_id][token]
                doc_len = self.doc_lengths[chunk_id]

                # BM25 term weighting formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = term_idf * (numerator / denominator)

                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:top_k]
