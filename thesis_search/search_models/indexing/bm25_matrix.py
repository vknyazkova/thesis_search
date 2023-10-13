import numpy as np
import pandas as pd

from ..base.matrix_search import MatrixSearch


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
