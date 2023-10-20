import os
import re
from functools import partial
from pathlib import Path
from typing import Union, Dict, Any, Tuple, Callable, Iterable

import pandas as pd
import spacy
from spacy import Language

from .models import Thesis
from .. import DATA_FOLDER_PATH, INDEX_PATH, MODEL_DEFAULT_PARAMS
from ..search_models import BertIndex, BM25Matrices, FastTextSearch, Word2VecSearch
from ..utils.database import DBHandler


def download_models(models_info: Dict[str, Dict[str, Union[Path, str]]],
                    search_engines: Dict[str, Any]):
    """
    Скачивает модели в папку model_folder
    Args:
        models_info: словарь вида {model_type: {'file': filename, 'source_link': link}}
            model_type - тип модели - w2v, ft
            filename - путь к модели
            link - ссылка на скачивание модели
        search_engines: словарь с классами для поиска

    Returns:

    """
    for model in models_info:
        filepath = models_info[model]['file']
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
        'bm25': {'k': MODEL_DEFAULT_PARAMS['bm25']['k'],
                 'b': MODEL_DEFAULT_PARAMS['bm25']['b']},
        'w2v': {'nlp_': fast_nlp,
                'model_path': MODEL_DEFAULT_PARAMS['w2v']['file'],
                'model_name_': MODEL_DEFAULT_PARAMS['w2v']['file'].stem,
                'index_folder_': INDEX_PATH,
                'similarity_metric': MODEL_DEFAULT_PARAMS['w2v']['similarity_metric']},
        'ft': {'model_path': MODEL_DEFAULT_PARAMS['ft']['file'],
               'model_name_': MODEL_DEFAULT_PARAMS['ft']['file'].stem,
               'index_folder_': INDEX_PATH,
               'similarity_metric': MODEL_DEFAULT_PARAMS['w2v']['similarity_metric']},
        'bert': {'nlp_': fast_nlp,
                 'model_name': MODEL_DEFAULT_PARAMS['bert']['model_name'],
                 'model_path': MODEL_DEFAULT_PARAMS['bert']['model_path'],
                 'index_folder_': INDEX_PATH,
                 'similarity_metric': MODEL_DEFAULT_PARAMS['bert']['similarity_metric']}
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


def filter_texts(results: Iterable[Thesis], threshold: int = 100) -> Iterable[Thesis]:
    """
    Filters texts that have less than threshold length
    :param results:
    :param threshold:
    :return:
    """
    clean_theses = []
    for thesis in results:
        if len(thesis.abstract) < threshold:
            continue
        clean_theses.append(thesis)
    return clean_theses


def preprocessing(text: str, nlp: Language) -> str:
    """
    Удаляет пунктуацию, стоп-слова и числа, оставшееся лемматизирует
    Args:
        text: строка текста
        nlp: spacy nlp object

    Returns: лемматизированный текст
    """
    lemmatized = []
    text = re.sub(r'\s+', ' ', text)
    for token in nlp(text):
        if not token.is_punct and not token.is_stop and not token.is_digit:
            lemmatized.append(token.lemma_.lower())
    return ' '.join(lemmatized)


def short_lines_generator(text):
    max_len = 79
    tokens = text.split()
    length = 0
    line_start = 0
    for i, token in enumerate(tokens):
        if length + len(token) > max_len:
            line = ' '.join(tokens[line_start: i])
            yield line
            length = 0
            line_start = i
        length += len(token)

    if line_start != i:
        line = ' '.join(tokens[line_start:])
        yield line


def pprint_result(results: Iterable[tuple]):
    print('-' * 79)
    for result in results:
        print(f"название:\t{result[0]}\nгод:\t{result[1]}\nобразовательная программа:\t{result[2]}\n"
              f"студент:\t{result[3]}\nнаучный руководитель:\t{result[4]}")
        print("описание:", end="\t")
        try:
            for line in short_lines_generator(result[5]):
                print(line)
        except UnboundLocalError:
            print(result[5])
        if result[6]:
            print(f'ссылка на скачивание:\t{result[6]}')
        print('-' * 79)


