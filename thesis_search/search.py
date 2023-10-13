from typing import Tuple, List

from spacy.language import Language

from thesis_search.search_models.base.search_engine import SearchEngine
from thesis_search.utils.database import DBHandler
from thesis_search.utils.models import preprocessing, pprint_result
from thesis_search import download_models, init_defaults
from thesis_search.config import DEFAULT_LMS, LM_PATH, INDEX_TYPES


def search_theses(
        query: str,
        nlp: Language,
        search_engine: SearchEngine,
        db: DBHandler,
        n: int
) -> List[Tuple[str, int, str, str, str, str, str]]:
    """
    Searches query in database documents
    Args:
        query: query string
        nlp: spacy language pipeline
        search_engine: search class instance
        db: database class instance
        n: number of relevant documents in search results

    Returns: result with info about every document6 containing (title, year, program, student, supervisor,
    annotation and link

    """
    lemmatized_query = preprocessing(query, nlp)
    found_documents = search_engine.rank_documents(lemmatized_query, n)
    results = [db.get_thesis_info(i) for i in found_documents]
    return results


if __name__ == '__main__':

    db, fast_nlp, corpus, default_params, search_engines = init_defaults()
    download_models(LM_PATH, DEFAULT_LMS, search_engines)

    ch = 1
    while ch == 1:
        idx_type = input(f'Выберите способ индексирования {INDEX_TYPES} : ')
        query = input('Введите ваш запрос: ')
        n = int(input('Количество документов в выдаче: '))

        if idx_type not in INDEX_TYPES:
            raise ValueError(f'Index type can be only one of those {INDEX_TYPES}')

        search_engine = search_engines[idx_type](corpus=corpus[idx_type], **default_params[idx_type])

        results = search_theses(query=query, nlp=fast_nlp, search_engine=search_engine,
                                db=db, n=n)
        pprint_result(results)

        ch = int(input('Продолжить поиск? (1 - да, 0 - нет): '))
