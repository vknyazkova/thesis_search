import typer
from pathlib import Path

import pandas as pd
from rich.console import Console

from ..config import DATA_FOLDER_PATH, LM_PATH, DEFAULT_LMS, INDEX_TYPES
from ..search import search_theses
from ..utils.models import pprint_result, pretty_table, pandas_to_rich_table, table_config
from thesis_search import download_models, init_defaults


app = typer.Typer()
console = Console()

db, fast_nlp, corpus, default_params, search_engines, preprocess = init_defaults()


@app.command(help='Ищет документы в корпусе по запросу')
def search(query: str,
           idx_type: str = typer.Option(
               default='bm25',
               help=f"Способ индексирования {INDEX_TYPES}"
           ),
           n: int = typer.Option(
               default=1,
               help='Количество документов в выдаче'
           ),
           style: str = typer.Option(
               default='text',
               help='Стиль вывода - text или table'
           )):

    if idx_type not in INDEX_TYPES:
        raise ValueError(f'Index type can be only one of those {INDEX_TYPES}')

    try:
        search_engine = search_engines[idx_type](corpus=corpus[idx_type], **default_params[idx_type])
    except FileNotFoundError:
        raise FileNotFoundError(f'Модель для этого способа индексации еще не скачена. Запустите команду '
                                f'"python -m thesis_search download {idx_type}", а потом попробуйте еще раз')
    results = search_theses(query=query, preprocess=preprocess[idx_type], search_engine=search_engine, db=db, n=n)

    if style == 'table':
        for result in results:
            table = pretty_table(result)
            console.print(table)
    elif style == 'text':
        pprint_result(results)


@app.command(help='Показывает статистику по реализованным методам поиска: время и память')
def stats():
    time_stats = pd.read_csv(Path(DATA_FOLDER_PATH, 'time_statistics.csv'), header=0, index_col=0)
    memory_stats = pd.read_csv(Path(DATA_FOLDER_PATH, 'memory_statistics.csv'), header=0, index_col=0)

    time_table = pandas_to_rich_table(time_stats)
    time_table.title = 'Время (в секундах), затрачиваемое на поиск (full - вместе с инициализацией, search_only - только поиск):'
    console.print(time_table)

    memory_table = pandas_to_rich_table(memory_stats)
    memory_table.title = 'Память (в байтах), затрачиваемая на поиск (full - вместе с инициализацией, search_only - только поиск):'
    console.print(memory_table)


@app.command(help="Показывает текущие конфигурации")
def show_config():
    tables = table_config(default_params)
    for t in tables:
        console.print(t)


@app.command(help="Скачивает векторную модель модель")
def download(model_type: str = typer.Argument(..., help='Какая модель - w2v или ft'),
             source_link: str = typer.Option(default=None,
                                             help='Ссылка на скачивание модели, если нет, то скачается дефолтная')):
    if model_type not in DEFAULT_LMS:
        raise ValueError(f'Can work only with {list(DEFAULT_LMS.keys())}')

    if not source_link:
        source_link = DEFAULT_LMS[model_type]['source_link']
        filename = DEFAULT_LMS[model_type]['file']
    else:
        filename = Path(source_link).with_suffix('.bin')
    models_info = {model_type: {'file': filename, 'source_link': source_link}}

    download_models(LM_PATH, models_info, search_engines)


if __name__ == "__main__":
    app()
