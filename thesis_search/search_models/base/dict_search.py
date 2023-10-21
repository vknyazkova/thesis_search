from collections import defaultdict
from typing import Iterable, Dict

import pandas as pd

from .search_engine import SearchEngine


class DictSearch(SearchEngine):
    """
    Поиск при помощи частотного индекса
    """
    def __init__(self, corpus: pd.DataFrame):
        super().__init__(corpus)
        self.doc_idx = self.doc_idx.tolist()
        self._index = None  # {word: {doc_idx: score}}

    @property
    def index(self):
        """
        Индекс в виде словаря {word: {doc_idx: score}}
        """
        return self._index

    @index.setter
    def index(self, index: Dict[str, Dict[int, int]]):
        self._index = index

    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        scores = defaultdict(int)
        for word in lemmatized_query.split():
            if self.index.get(word):
                for doc in self.index[word]:
                    scores[doc] += self.index[word][doc]
        sorted_metric = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in sorted_metric[:top_n]]