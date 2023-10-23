import os
from functools import partial
from typing import Union, Callable, Dict, Any
import re

import pandas as pd
import spacy
from spacy import Language

from ..utils.database import DBHandler
from .indexing import *


class SearchEngine:

    nlp = spacy.load("ru_core_news_sm", exclude=["ner"])
    downloadable = {'w2v': Word2VecSearch,
                    'ft': FastTextSearch}

    def __init__(self,
                 index_type: str,
                 implementation: str,
                 index_folder: Union[str, os.PathLike],
                 data_retriever: DBHandler,
                 defaults: Dict[str, Any],
                 preprocessor: Union[str, Callable[[str], str]] = 'lemmatize'):
        self.db = data_retriever
        self.corpus = index_type
        self.index_folder = index_folder
        self.defaults = defaults
        self.preprocessor = preprocessor

        self.model = self.init_model(index_type, implementation)

    @property
    def corpus(self):
        return self.corpus_

    @corpus.setter
    def corpus(self, idx_type):
        if idx_type == 'bert':
            self.corpus_ = pd.DataFrame(self.db.get_raw_texts(), columns=['id', 'text'])
        else:
            self.corpus_ = pd.DataFrame(self.db.get_lemmatized_texts(), columns=['id', 'lemmatized'])

    def init_model(self, idx_type, implementation):
        if idx_type == 'bm25':
            if implementation == 'BM25Matrices':
                return BM25Matrices(
                    corpus=self.corpus,
                    k=float(self.defaults['k']),
                    b=float(self.defaults['b'])
                )
            elif implementation == 'BM25Dict':
                return BM25Dict(
                    corpus=self.corpus,
                    k=float(self.defaults['k']),
                    b=float(self.defaults['b']))
            elif implementation == 'BM25Search':
                return BM25Search(
                    corpus=self.corpus
                )
            else:
                raise ValueError('Unknown implementation')
        elif idx_type == 'freq':
            if implementation == 'FreqMatrix':
                return FreqMatrix(
                    corpus=self.corpus
                )
            elif implementation == 'CountVectSearch':
                return CountVectSearch(
                    corpus=self.corpus
                )
            elif implementation == 'FreqDict':
                return FreqDict(
                    corpus=self.corpus
                )
        elif idx_type == 'w2v':
            if implementation == 'Word2VecSearch':
                return Word2VecSearch(
                    corpus=self.corpus,
                    model_name_=self.defaults['model_name'],
                    nlp_=self.nlp,
                    model_path=self.defaults['model_path'],
                    index_folder_=self.index_folder,
                    similarity_metric=self.defaults['similarity_metric']
                )
            else:
                raise ValueError('Unknown implememtation')
        elif idx_type == 'ft':
            if implementation == 'FastTextSearch':
                return FastTextSearch(
                    corpus=self.corpus,
                    model_name_=self.defaults['model_name'],
                    model_path=self.defaults['model_path'],
                    index_folder_=self.index_folder,
                    similarity_metric=self.defaults['similarity_metric']
                )
            else:
                raise ValueError('Unknown implementation')
        elif idx_type == 'bert':
            if implementation == 'BertIndex':
                return BertIndex(
                    corpus=self.corpus,
                    nlp_=self.nlp,
                    model_name=self.defaults['model_name'],
                    model_path=self.defaults['model_path'],
                    index_folder_=self.index_folder,
                    similarity_metric=self.defaults['similarity_metric']
                )
            else:
                raise ValueError('Unknown implementation')
        else:
            raise ValueError('Unknown index type')

    @property
    def preprocessor(self):
        return self.preprocessor_

    @preprocessor.setter
    def preprocessor(self, name: Union[str, Callable[[str], str]]):
        if isinstance(name, str):
            if name == 'lemmatize':
                self.preprocessor_ = partial(self.spacy_preprocessing, nlp=self.nlp)
            elif name == 'raw':
                self.preprocessor_ = lambda x: x
            else:
                raise ValueError('Wrong preprocessor name')
        else:
            self.preprocessor_ = name

    @staticmethod
    def spacy_preprocessing(text: str, nlp: Language) -> str:
        """
        Delete punctuation, stop-words and numbers and then lemmatize
        Args:
            text: string of text
            nlp: spacy nlp object

        Returns: lemmatized text
        """
        lemmatized = []
        text = re.sub(r'\s+', ' ', text)
        for token in nlp(text):
            if not token.is_punct and not token.is_stop and not token.is_digit:
                lemmatized.append(token.lemma_.lower())
        return ' '.join(lemmatized)

    @classmethod
    def download_model(cls,
                       idx_type: str,
                       model_path: Union[str, os.PathLike],
                       url: str):
        if idx_type in cls.downloadable:
            cls.downloadable[idx_type].download_model(url, model_path)
        else:
            raise ValueError('Wrong index type')

    def search(self, query, n):
        lemmatized_query = self.preprocessor(query)
        if not lemmatized_query:
            raise QueryError('Query has no content words. Please change your query to something more meaningful :(')
        found_documents = self.model.rank_documents(lemmatized_query, n)
        results = [self.db.get_thesis_info(i) for i in found_documents]
        return results


class QueryError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
