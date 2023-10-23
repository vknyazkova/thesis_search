from abc import abstractmethod
from typing import Iterable

import pandas as pd


class BaseSearch:
    """
    Abstract class for search in corpus
    Attributes:
        doc_idx: pd.Series of document indices
        text: pd.Series of document texts
    """
    def __init__(self, corpus: pd.DataFrame):
        """
        Loads corpus
        Args:
            corpus: pandas dataframe with two columns - 1st documents' indices, 2nd - documents' texts
        """
        self.doc_idx, self.text = [corpus[col] for col in corpus.columns]

    @abstractmethod
    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        """
        Method that ranges documents by relevance to the query
        Args:
            lemmatized_query: string of lemmatized query
            top_n: number of relevant documents in the result

        Returns: list of indices of relevant documents
        """
        ...
