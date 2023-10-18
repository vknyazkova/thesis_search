from dataclasses import dataclass
from typing import Iterable, Tuple, Dict
import re

import pandas as pd
import progressbar
from rich.table import Table
from spacy.language import Language


@dataclass
class Thesis:
    thesis_id: int = None
    year: int = None
    title: str = None
    student: str = None
    supervisors: Iterable[str] = None
    learn_program: str = None
    abstract: str = None
    text_lemmatized: str = None
    file: str = None


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


# взято со стэковерфлоу)
class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


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


def pretty_table(result: Tuple[str, int, str, str, str, str, str]) -> Table:
    table = Table()
    fields = ['название', 'год', 'образовательная программа', 'студент', 'научный руководитель', 'описание', 'файл']
    for field, value in zip(fields, result):
        table.add_row(field, str(value))
    return table


def table_config(configs: Dict[str, dict]):
    tables = []
    for model in configs:
        table = Table(model)
        for k, v in configs[model].items():
            table.add_row(k, str(v))
        tables.append(table)
    return tables


def pandas_to_rich_table(df: pd.DataFrame):
    df = df.astype(str)
    table = Table('', *df.columns.tolist())
    for index, row in df.iterrows():
        table.add_row(index, *row.values)
    return table


class QueryError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message