import typer
from pathlib import Path

import pandas as pd
from rich.console import Console

from .. import DATA_FOLDER_PATH, LM_PATH, INDEX_TYPES, MODEL_DEFAULT_PARAMS
from ..search import search_theses
from ..utils.utils import download_models, init_defaults, pprint_result
from .cli_utils import pretty_table, table_config, pandas_to_rich_table, change_config, remove_model_from_config

app = typer.Typer()
console = Console()

db, fast_nlp, corpus, default_params, search_engines, preprocess = init_defaults()
downloadable_models = [m for m in MODEL_DEFAULT_PARAMS if MODEL_DEFAULT_PARAMS[m].get('source_link')]


@app.command(help='Search documents relevant to the query')
def search(query: str,
           idx_type: str = typer.Option(
               default='bm25',
               help=f"Index type, one of: {list(INDEX_TYPES.keys())}"
           ),
           n: int = typer.Option(
               default=1,
               help='Number of documents in the result'
           ),
           style: str = typer.Option(
               default='text',
               help='Output style: plain text or table'
           )):

    if idx_type not in INDEX_TYPES:
        raise ValueError(f'Index type can be only one of those {list(INDEX_TYPES.keys())}')

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


@app.command(help='Show statistics of corpus search methods: time and memory')
def stats():
    time_stats = pd.read_csv(Path(DATA_FOLDER_PATH, 'time_statistics.csv'), header=0, index_col=0)
    memory_stats = pd.read_csv(Path(DATA_FOLDER_PATH, 'memory_statistics.csv'), header=0, index_col=0)

    time_table = pandas_to_rich_table(time_stats)
    time_table.title = 'Время (в секундах), затрачиваемое на поиск (full - вместе с инициализацией, search_only - только поиск):'
    console.print(time_table)

    memory_table = pandas_to_rich_table(memory_stats)
    memory_table.title = 'Память (в байтах), затрачиваемая на поиск (full - вместе с инициализацией, search_only - только поиск):'
    console.print(memory_table)


@app.command(help="Show models' configurations")
def show_config():
    tables = table_config(MODEL_DEFAULT_PARAMS)
    for t in tables:
        console.print(t)


@app.command(help="Change model config")
def change_model_config(
        model_type: str = typer.Argument(..., help=f'Model type: {list(MODEL_DEFAULT_PARAMS.keys())}'),
        config_name: str = typer.Argument(..., help=f'Config name to change '
                                                    f'(to see the available configs use command show-config)'),
        config_value: str = typer.Argument(..., help=f'New value for selected config')
):
    changed = change_config(model_type, config_name, config_value)
    tables = table_config({model_type: changed})
    for t in tables:
        console.print(t)


@app.command(help="Remove model")
def remove_model(
    model_type: str = typer.Argument(..., help=f'Model type: {list(MODEL_DEFAULT_PARAMS.keys())}')
):
    models_left = remove_model_from_config(model_type)
    print(f'Available models: {models_left}')


@app.command(help="Download vector model")
def download(model_type: str = typer.Argument(..., help=f'Модель одна из: {downloadable_models}'),
             source_link: str = typer.Option(default=None,
                                             help='Ссылка на скачивание модели, если нет, то скачается дефолтная')):
    if model_type not in downloadable_models:
        raise ValueError(f'Can work only with {downloadable_models}')

    if not source_link:
        source_link = MODEL_DEFAULT_PARAMS[model_type]['source_link']
        filename = MODEL_DEFAULT_PARAMS[model_type]['file']
    else:
        filename = Path(source_link).with_suffix('.bin')
    models_info = {model_type: {'file': filename, 'source_link': source_link}}

    download_models(LM_PATH, models_info, search_engines)


if __name__ == "__main__":
    app()
