import os
from functools import partial
from pathlib import Path
from typing import Union, Dict, Any, Tuple, Callable

import pandas as pd
import spacy
from spacy import Language

from .. import DATA_FOLDER_PATH, LM_PATH, DEFAULT_LMS, INDEX_PATH
from ..search_models.indexing.bert_index import BertIndex
from ..search_models.indexing.bm25_matrix import BM25Matrices
from ..search_models.indexing.fasttext_index import FastTextSearch
from ..search_models.indexing.word2vec_index import Word2VecSearch
from ..utils.database import DBHandler
from ..utils.models import preprocessing


def download_models(model_folder: Union[str, os.PathLike],
                    models_info: Dict[str, Dict[str, str]],
                    search_engines: Dict[str, Any]):
    """
    Скачивает модели в папку model_folder
    Args:
        model_folder: папка, где будут храниться модели
        models_info: словарь вида {model_type: {'file': filename, 'source_link': link}}
            model_type - тип модели - w2v, ft
            filename - название файла с моделью
            link - ссылка на скачивание модели
        search_engines: словарь с классами для поиска

    Returns:

    """
    for model in models_info:
        filepath = Path(model_folder, models_info[model]['file'])
        if not filepath.exists():
            search_engines[model].download_model(models_info[model]['source_link'], filepath)


def init_defaults() -> Tuple[DBHandler,
                            Language,
                            Dict[str, pd.DataFrame],
                            Dict[str, Dict[str, Any]],
                            Dict[str, type],
                            Dict[str, Callable]]:
    """
    Initializes needed objects with defaults
    Returns:
        db - instance of DBHandler
        fast_nlp - spacy language pipeline
        corpus - dictionary {index_type: pandas dataframe}
        default_params - default parameters for different SearchEngines init
        search_engines - search engine class name for every index_type

    """
    db = DBHandler(Path(DATA_FOLDER_PATH, 'theses.db'))
    fast_nlp = spacy.load("ru_core_news_sm", exclude=["ner"])
    lemmatized_corpus = pd.DataFrame(db.get_lemmatized_texts(), columns=['id', 'lemmatized'])
    raw_corpus = pd.DataFrame(db.get_raw_texts(), columns=['id', 'text'])

    corpus = {
        'bm25': lemmatized_corpus,
        'w2v': lemmatized_corpus,
        'ft': lemmatized_corpus,
        'bert': raw_corpus
    }

    default_params = {
        'bm25': {'k': 2, 'b': 0.75},
        'w2v': {'nlp': fast_nlp,
                'model_path': Path(LM_PATH, DEFAULT_LMS['w2v']['file']),
                'model_name': Path(DEFAULT_LMS['w2v']['file']).stem,
                'index_folder': INDEX_PATH,
                'similarity_metric': 'cosine'},
        'ft': {'model_path': Path(LM_PATH, DEFAULT_LMS['ft']['file']),
               'model_name': Path(DEFAULT_LMS['ft']['file']).stem,
               'index_folder': INDEX_PATH,
               'similarity_metric': 'cosine'},
        'bert': {'nlp': fast_nlp,
                 'model_name': 'sbert_large_nlu_ru',
                 'model_path': 'ai-forever/sbert_large_nlu_ru',
                 'index_folder': INDEX_PATH,
                 'similarity_metric': 'cosine'}
    }

    search_engines = {
        'bm25': BM25Matrices,
        'w2v': Word2VecSearch,
        'ft': FastTextSearch,
        'bert': BertIndex
    }
    preprocessings = {
        'bm25': partial(preprocessing, nlp=fast_nlp),
        'w2v': partial(preprocessing, nlp=fast_nlp),
        'ft': partial(preprocessing, nlp=fast_nlp),
        'bert': lambda x: x,
    }

    return db, fast_nlp, corpus, default_params, search_engines, preprocessings
