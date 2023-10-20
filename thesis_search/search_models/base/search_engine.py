from abc import abstractmethod
from typing import Iterable

import pandas as pd


class SearchEngine:
    """
    Абстрактный класс для поиска по корпусу
    """
    def __init__(self, corpus: pd.DataFrame):
        """
        Инициализирует модель
        Args:
            corpus: датафрейм с двумя столбцами - индекс документа и текст
        """
        self.doc_idx, self.text = [corpus[col] for col in corpus.columns]

    @abstractmethod
    def rank_documents(self, lemmatized_query: str, top_n: int) -> Iterable[int]:
        """
        Метод, который ранжирует документы по близости к запросу
        Args:
            lemmatized_query: строка лемматизированного запроса
            top_n: сколько релевантных документов выдавать

        Returns: список индексов релевантных документов
        """
        ...
