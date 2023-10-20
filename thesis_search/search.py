from typing import Tuple, List, Callable

from thesis_search.search_models.base.search_engine import SearchEngine
from thesis_search.utils.database import DBHandler
from thesis_search.utils.models import QueryError
from thesis_search.utils.utils import download_models, init_defaults, pprint_result
from thesis_search import INDEX_TYPES, MODEL_DEFAULT_PARAMS


def search_theses(
        query: str,
        preprocess: Callable,
        search_engine: SearchEngine,
        db: DBHandler,
        n: int
) -> List[Tuple[str, int, str, str, str, str, str]]:
    """
    Searches query in database documents
    Args:
        query: query string
        preprocess: callable to preprocess query
        search_engine: search class instance
        db: database class instance
        n: number of relevant documents in search results

    Returns: result with info about every document6 containing (title, year, program, student, supervisor,
    annotation and link

    """
    lemmatized_query = preprocess(query)
    if not lemmatized_query:
        raise QueryError('Query has no content words. Please change your query to something more meaningful :(')
    found_documents = search_engine.rank_documents(lemmatized_query, n)
    results = [db.get_thesis_info(i) for i in found_documents]
    return results


if __name__ == '__main__':

    db, fast_nlp, corpus, default_params, search_engines, preprocess = init_defaults()
    static_vector_models = {m: MODEL_DEFAULT_PARAMS[m] for m in MODEL_DEFAULT_PARAMS if MODEL_DEFAULT_PARAMS[m].get('source_link')}
    download_models(static_vector_models, search_engines)

    ch = 1
    while ch == 1:
        idx_type = input(f'Выберите способ индексирования {list(INDEX_TYPES.keys())} : ')
        query = input('Введите ваш запрос: ')
        n = int(input('Количество документов в выдаче: '))

        if idx_type not in INDEX_TYPES:
            raise ValueError(f'Index type can be only one of those {list(INDEX_TYPES.keys())}, but you entered {idx_type}')

        search_engine = search_engines[idx_type](corpus=corpus[idx_type], **default_params[idx_type])

        results = search_theses(query=query, preprocess=preprocess[idx_type], search_engine=search_engine,
                                db=db, n=n)
        pprint_result(results)

        ch = int(input('Продолжить поиск? (1 - да, 0 - нет): '))
