from typing import Iterable, Dict, List
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ..base.search_engine import SearchEngine
from ..base.dict_search import DictSearch
from ..base.matrix_search import MatrixSearch


class CountVectSearch(SearchEngine):
    """
    Inverted frequency index implemented through CountVectorizer of sklearn
    Attributes:
        _index: the result of CountVectorizer fit_transform function
        _vocabulary: mapping of terms to feature indices
    """
    def __init__(self, corpus: pd.DataFrame):
        super().__init__(corpus)
        self.doc_idx = self.doc_idx.to_numpy()

        vectorizer = CountVectorizer()
        self._index = vectorizer.fit_transform(self.text).transpose()
        self._vocabulary = vectorizer.vocabulary_

    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        vectorizer = CountVectorizer(vocabulary=self._vocabulary, binary=True)  # для вектора запроса использую бинарный
        query_vector = vectorizer.transform([lemmatized_query])
        metric = (query_vector @ self._index).toarray().reshape(-1,)
        rank = (-metric).argsort()
        return self.doc_idx[rank][:top_n].tolist()


class FreqDict(DictSearch):
    """
    Search based on frequency index implemented through python dictionaries
    Attributes:
        index: dictionary in form of {term: {doc_idx: frequency}}
    """
    def __init__(self, corpus: pd.DataFrame):
        super().__init__(corpus)
        self.index = self._compute_index(self.text.tolist(), self.doc_idx)

    @staticmethod
    def _compute_index(texts: Iterable[str], doc_idx: List[int]) -> Dict[str, Dict[int, int]]:
        index = defaultdict(lambda: defaultdict(int))
        for i, text in enumerate(texts):
            for w in text.split():
                index[w][doc_idx[i]] += 1
        return index


class FreqMatrix(MatrixSearch):
    """Frequency index implemented through matrix"""
    def __init__(self, corpus: pd.DataFrame):
        super().__init__(corpus)
        self._index, self._vocabulary = self.extract_features(self.text)
