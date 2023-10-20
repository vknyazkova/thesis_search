from typing import Tuple, Dict

import pandas as pd
from rich.table import Table


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
            if not k.endswith('_'):
                table.add_row(k, str(v))
        tables.append(table)
    return tables


def pandas_to_rich_table(df: pd.DataFrame):
    df = df.astype(str)
    table = Table('', *df.columns.tolist())
    for index, row in df.iterrows():
        table.add_row(index, *row.values)
    return table
