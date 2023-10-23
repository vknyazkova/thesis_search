from pathlib import Path
from typing import List

import pandas as pd
import typer
from rich.console import Console

from .. import DATA_FOLDER, INDEX_TYPES, MODEL_DEFAULTS, INDEX_FOLDER
from ..search_models.search_engine import SearchEngine, download_model
from ..utils.utils import pprint_result
from ..utils.database import DBHandler
from .cli_utils import pretty_table, table_config, pandas_to_rich_table, change_config, remove_index_from_config, add_index_to_config

app = typer.Typer()
console = Console()
db = DBHandler(Path(DATA_FOLDER, 'theses.db'))
downloadable_models = [m for m in MODEL_DEFAULTS if MODEL_DEFAULTS[m].get('source_link')]


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
        search_engine = SearchEngine(
            index_type=idx_type,
            implementation=MODEL_DEFAULTS[idx_type]['implementation'],
            index_folder=INDEX_FOLDER,
            data_retriever=db,
            defaults=MODEL_DEFAULTS[idx_type],
            preprocessor=MODEL_DEFAULTS[idx_type]['preprocessor_']
        )
    except FileNotFoundError:
        raise FileNotFoundError(f'Модель для этого способа индексации еще не скачена. Запустите команду '
                                f'"python -m thesis_search download {idx_type}", а потом попробуйте еще раз')
    results = search_engine.search(query, n)

    if style == 'table':
        for result in results:
            table = pretty_table(result)
            console.print(table)
    elif style == 'text':
        pprint_result(results)


@app.command(help='Show statistics of corpus search methods: time and memory')
def stats():
    time_stats = pd.read_csv(Path(DATA_FOLDER, 'time_statistics.csv'), header=0, index_col=0)
    memory_stats = pd.read_csv(Path(DATA_FOLDER, 'memory_statistics.csv'), header=0, index_col=0)

    time_table = pandas_to_rich_table(time_stats)
    time_table.title = 'Время (в секундах), затрачиваемое на поиск (full - вместе с инициализацией, search_only - только поиск):'
    console.print(time_table)

    memory_table = pandas_to_rich_table(memory_stats)
    memory_table.title = 'Память (в байтах), затрачиваемая на поиск (full - вместе с инициализацией, search_only - только поиск):'
    console.print(memory_table)


@app.command(help="Show models' configurations")
def show_config():
    tables = table_config(MODEL_DEFAULTS, INDEX_TYPES)
    for t in tables:
        console.print(t)


@app.command(help="Change model config")
def change_model_config(
        model_type: str = typer.Argument(..., help=f'Model type: {list(MODEL_DEFAULTS.keys())}'),
        config_name: str = typer.Argument(..., help=f'Config name to change '
                                                    f'(to see the available configs use command show-config)'),
        config_value: str = typer.Argument(..., help=f'New value for selected config')
):
    changed = change_config(model_type, config_name, config_value)
    tables = table_config({model_type: changed})
    for t in tables:
        console.print(t)


@app.command(help="Remove index type")
def remove_indices(
    index_types: List[str] = typer.Argument(..., help=f'Some indices from those: {list(INDEX_TYPES.keys())}')
):
    models_left = remove_index_from_config(index_types)
    print(f'Available indices: {models_left}')


@app.command(help="Add index type")
def add_index(
        index_type: str = typer.Argument(..., help=f'One of those indices: {set(MODEL_DEFAULTS.keys()) - set(INDEX_TYPES.keys())}'),
        name: str = typer.Option(default=None, help="full name for index type")
):
    all_models = add_index_to_config(index_type, name)
    print(f'Available indices: {all_models}')


@app.command(help="Download vector model")
def download(model_type: str = typer.Argument(..., help=f'One of the following model type: {downloadable_models}'),
             source_link: str = typer.Option(default=None,
                                             help='Download link, if there are any the default model (from config) will be downloaded')):
    if model_type not in downloadable_models:
        raise ValueError(f'Can work only with {downloadable_models}')

    if not source_link:
        source_link = MODEL_DEFAULTS[model_type]['source_link']
        filename = MODEL_DEFAULTS[model_type]['model_path']
    else:
        filename = Path(source_link).with_suffix('.bin')
    download_model(model_type, filename, source_link)


if __name__ == "__main__":
    app()
