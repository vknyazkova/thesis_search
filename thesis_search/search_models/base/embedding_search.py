import os
from abc import abstractmethod
from typing import Iterable, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import csr_array

from .search_engine import SearchEngine


class EmbeddingSearch(SearchEngine):
    """
    Search in the index based on document embeddings
    Attributes:
        loaded: dictionary of loaded vector_models attributes {model_name: {attr: val}}
        model_name: name of pre-trained vector-model
        similarity: metric that is used to compute relevance od the document to the query

    """
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
        Adds model to loaded

        """
        ...

    def save_index(self, index_path: Union[str, os.PathLike]):
        """
        Saves precomputed index to index_path
        Args:
            index_path: path to index file
        """

        np.savetxt(index_path, self.index)

    @staticmethod
    def load_index(index_path: Union[str, os.PathLike]) -> np.ndarray:
        """
        Loads precomputed index from the file
        Args:
            index_path: path to index file

        Returns: numpy ndarray with shape (n_docs, emb_size)

        """
        return np.loadtxt(index_path)

    @property
    def index(self):
        """
        Matrix of vectorized documents
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
        Computed cosine similarity of documents and the query
        Args:
            documents: document embeddings shape=(n_docs, emb_size)
            query_vector: query vector shape=(emb_size, 1)

        Returns: vector filled with values of documents' cosine similarities shape=(n_docs, 1)

        """
        return (documents @ query_vector) / (
                norm(documents, axis=1, keepdims=True) * norm(query_vector, keepdims=True)
        )

    @staticmethod
    def dot_prod_similarity(documents: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        """
        Computes dot-product of document and query vectors
        Args:
            documents: document embeddings shape=(n_docs, emb_size)
            query_vector: query vector shape=(emb_size, 1)

        Returns: dot-product results shape=(n_docs, 1)

        """
        return documents @ query_vector

    @abstractmethod
    def compute_index(self) -> np.ndarray:
        """
        Computes index for corpus documents
        Returns: numpy ndarray with the shape = (n_docs, emb_size)

        """
        ...

    @abstractmethod
    def vectorize(self, text: str) -> np.ndarray:
        """
        Computes embedding of the document
        Args:
            text: text as a string

        Returns: vector with size = (emb_size, 1)

        """
        ...

    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        query_vector = self.vectorize(lemmatized_query)
        scores = self.similarity(self.index, query_vector)
        rank = (-scores).argsort(axis=0)[:, 0]
        return self.doc_idx[rank][:top_n].tolist()
