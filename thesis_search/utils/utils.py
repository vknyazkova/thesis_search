import re
from typing import Iterable

from spacy import Language

from .models import Thesis


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


