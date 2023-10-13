from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from .search_engine import SearchEngine


class MatrixSearch(SearchEngine):
    """
    Базовая имлементация для поиска через матрицы
    """
    def __init__(self, corpus: pd.DataFrame):
        super().__init__(corpus)
        self._index = None
        self._vocabulary = None

    @property
    def index(self):
        """
        Матрица значения метрики для каждой пары (документ, слово)
        # shape = (n_docs, vocab_size)
        """
        return self._index

    @index.setter
    def index(self, index_matrix: csr_array):
        self._index = index_matrix

    @staticmethod
    def extract_features(docs: Iterable[str]) -> Tuple[csr_array, Dict[str, int]]:
        """
        Получает вектор признаков для каждого документа, где признак - сколько раз слово
        встретилось в документе
        Args:
            docs: список документов в корпусе

        Returns: index, vocabulary
            index: частотный индекс для документов в формате csr_array
            vocabulary: словарь соответствий меджу словом и индексом соответствующего признанка

        """
        indptr = [0]  # куммулятивная сумма ненулевых элементов для каждой строки
        indices = []  # индексы столбцов, где хранятся ненулевые элементы (столбцы - термины)
        vocabulary = {}  # {слово: индекс признака}
        data = []
        for doc in docs:
            for w in doc.split():
                feature_index = vocabulary.setdefault(w, len(vocabulary))
                indices.append(feature_index)
                data.append(1)
            indptr.append(len(indices))
        index = csr_array((data, indices, indptr), dtype=int)
        return index, vocabulary

    def vectorize_query(self, query: str) -> np.ndarray:
        """
        Представляет запрос в виде вектора размером (размер_словаря, 1),
        где на позициях слов из запроса стоят 1, а в остальных местах нули
        Args:
            query: лемматизировнный запрос в виде строки

        Returns: вектор запроса

        """
        query_idx = []
        for w in query.split():
            if self._vocabulary.get(w):
                query_idx.append(self._vocabulary[w])
        vector = np.zeros((self._index.shape[1], 1))
        vector[query_idx] = 1
        return vector

    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        query_vector = self.vectorize_query(lemmatized_query)
        scores = self._index @ query_vector
        rank = (-scores).argsort(axis=0)[:, 0]
        return self.doc_idx[rank][:top_n].tolist()
