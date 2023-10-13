import os
from abc import abstractmethod
from typing import Iterable, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import csr_array

from .search_engine import SearchEngine


class EmbeddingSearch(SearchEngine):
    loaded = defaultdict(dict)  # информация о загруженных моделях

    def __init__(self,
                 corpus: pd.DataFrame,
                 model_name: str,
                 similarity_metric: str,):
        """

        Args:
            corpus: pandas dataframe with two columns: text ids and texts themselves
            model_name: string with model name (ex.:  'cc.ru.300.bin')
            similarity_metric: either 'cosine' or 'dot-prod'
        """
        super().__init__(corpus)
        self.model_name = model_name
        self.similarity = similarity_metric

    @abstractmethod
    def register_model(self):
        """
        Регистрирует модель в loaded, чтобы повторно не загружать уже существующую модель
        """
        ...

    def save_index(self, index_path: Union[str, os.PathLike]):
        """
        Saves precomputed index to index_path
        Args:
            index_path: path to index file

        Returns:

        """
        np.savetxt(index_path, self.index)

    @staticmethod
    def load_index(index_path: Union[str, os.PathLike]) -> np.ndarray:
        """
        Loads precomputed index from file
        Args:
            index_path: path to index file

        Returns: numpy ndarray with shape (n_docs, emb_size)

        """
        return np.loadtxt(index_path)

    @property
    def index(self):
        """
        Матрица с векторизованными документами
        # shape = (n_docs, emb_size)
        """
        return self.loaded[self.model_name]['index']

    @index.setter
    def index(self, index_matrix: csr_array):
        self.loaded[self.model_name]['index'] = index_matrix

    @property
    def model(self):
        return self.loaded[self.model_name]['model']

    @property
    def similarity(self):
        return self._similarity

    @similarity.setter
    def similarity(self, value: str):
        if value == 'dot-prod':
            self._similarity = self.dot_prod_similarity
        elif value == 'cosine':
            self._similarity = self.cosine_similarity
        else:
            raise ValueError('Wrong similarity metric')

    @staticmethod
    def cosine_similarity(documents: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        """
        Считает косинусную близость между документами и запросом
        Args:
            documents: матрица размером (n_docs, emb_size)
            query_vector: вектор запроса размером (emb_size, 1)

        Returns: вектор со значением косинусных близостей размером (n_docs, 1)

        """
        return (documents @ query_vector) / (
                norm(documents, axis=1, keepdims=True) * norm(query_vector, keepdims=True)
        )

    @staticmethod
    def dot_prod_similarity(documents: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        """
        Считает скалярное произведение между документами и запросом
        Args:
            documents: матриуа размером (n_docs, emb_size)
            query_vector: вектор запроса размером (emb_size, 1)

        Returns: вектор со значением скалярного произведения размером (n_docs, 1)

        """
        return documents @ query_vector

    @abstractmethod
    def compute_index(self) -> np.ndarray:
        """
        Indexes corpus documents
        Returns: numpy ndarray with shape (n_docs, emb_size)

        """
        ...

    @abstractmethod
    def vectorize(self, text: str) -> np.ndarray:
        """
        Метод векторизующии текста
        Args:
            text: текст в виде строки

        Returns: вектор размером (emb_size, 1)

        """
        ...

    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        query_vector = self.vectorize(lemmatized_query)
        scores = self.similarity(self.index, query_vector)
        rank = (-scores).argsort(axis=0)[:, 0]
        return self.doc_idx[rank][:top_n].tolist()
