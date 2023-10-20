from collections import defaultdict
from math import log
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from ..base.search_engine import SearchEngine
from ..base.matrix_search import MatrixSearch
from ..base.dict_search import DictSearch


class BM25Matrices(MatrixSearch):
    """
    Реализация бм25 индекса через матрицы
    """
    def __init__(self, corpus: pd.DataFrame, k: float = 2, b: float = 0.75):
        super().__init__(corpus)
        self.k = k
        self.b = b

        doc_term_count, self._vocabulary = self.extract_features(self.text)
        doc_term_count = doc_term_count.toarray()
        self._index = self.compute_bm25(doc_term_count, self.k, self.b)

    @staticmethod
    def compute_bm25(doc_term_count: np.ndarray, k: float, b: float) -> np.ndarray:
        """
        Подсчитывает значения бм25 для всего корпуса
        Args:
            doc_term_count: матрица документ-терм с количеством употреблений слова в документне (n_docs, vocab_size)
            k:
            b:

        Returns: подсчитанные бм25 значения в виде матрицы размером (n_docs, vocab_size)

        """
        N = doc_term_count.shape[0]

        tf = doc_term_count.astype(float)
        doc_lens = np.sum(doc_term_count, axis=1).reshape(-1, 1)  # (n_docs, 1)
        non_empty_docs = (doc_lens != 0)[:, 0]  # чтобы не делить на ноль для пустых документов
        tf[non_empty_docs, :] = tf[non_empty_docs, :] / doc_lens[non_empty_docs, :]

        df = np.count_nonzero(tf, axis=0)  # (vocab_size,)
        idf = np.log(N) - np.log(df)  # (vocab_size, )

        docs_len = tf.sum(axis=1)  # (n_docs, )
        avg_len = docs_len.mean()

        numerator = tf * (k + 1)  # (n_docs, vocab_size)
        denom_summand = k * (1 - b + b * docs_len / avg_len).reshape(-1, 1)  # (n_docs, 1)
        denominator = tf + denom_summand  # (n_docs, vocab_size)

        bm25_index = idf * numerator / denominator  # (n_docs, vocab_size)
        return bm25_index


class BM25Dict(DictSearch):
    """
    Класс для поиска при помощи словаря bm25
    """
    def __init__(self, corpus: pd.DataFrame, k: float = 2, b: float = 0.75):
        super().__init__(corpus)
        self.k = k
        self.b = b

        self.N = len(self.doc_idx)
        self.docs_len = {self.doc_idx[i]: max(len(text.split()), 1) for i, text in enumerate(self.text)}
        self.avg_len = sum(self.docs_len.values()) / self.N

        self._tf = self._compute_tf()
        self._idf = self._compute_idf()
        self._index = self.compute_bm25()

    def _compute_tf(self) -> Dict[str, Dict[int, float]]:
        tf = defaultdict(lambda: defaultdict(int))
        for i, doc in enumerate(self.text):
            for w in doc.split(' '):
                tf[w][self.doc_idx[i]] += (1 / self.docs_len[self.doc_idx[i]])
        return tf

    def _compute_idf(self) -> Dict[str, float]:
        idfs = {}
        for word in self._tf:
            word_df = len(self._tf[word].keys())
            idf = log(self.N) - log(word_df)
            idfs[word] = idf
        return idfs

    @staticmethod
    def bm25_formula(tf: float, idf: float, k: float, b: float, doc_len: int, avg_len: float):
        return idf * (tf * (k + 1)) / (tf + k * (1 - b + b * doc_len / avg_len))

    def compute_bm25(self):
        bm25 = defaultdict(lambda: defaultdict(float))
        for w in self._tf:
            for doc in self._tf[w]:
                bm25_value = self.bm25_formula(
                    tf=self._tf[w][doc],
                    idf=self._idf[w],
                    k=self.k,
                    b=self.b,
                    doc_len=self.docs_len[doc],
                    avg_len=self.avg_len
                )
                bm25[w][doc] = bm25_value
        return bm25


class BM25Search(SearchEngine):
    """Класс для поиска при помощи bm25, реализованного в библиотеке rank_bm25"""
    def __init__(self, corpus: pd.DataFrame):
        super().__init__(corpus)
        self.doc_idx = self.doc_idx.tolist()
        tokenized_corpus = [doc.split(" ") for doc in self.text]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        tokenized_query = lemmatized_query.split(" ")
        return self._bm25.get_top_n(tokenized_query, self.doc_idx, top_n)
