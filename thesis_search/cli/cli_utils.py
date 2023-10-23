from typing import Tuple, Dict, Any
from pathlib import Path

import yaml
import pandas as pd
from rich.table import Table

from .. import HOME_PATH


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


def change_config(model_type: str,
                  config_name: str,
                  new_value: Any) -> Dict[str, Any]:
    with open(Path(HOME_PATH, 'config.yml'), 'r') as file:
        config = yaml.safe_load(file)

    if model_type not in config['models_defaults']:
        raise ValueError(f'Unknown model type, only can be one of those: {list(config["models_defaults"].keys())}')
    if config_name not in config['models_defaults'][model_type]:
        raise ValueError(f"Selected model don't have such parameter, only one of those:"
                         f" {list(config['models_defaults'][model_type].keys())}")

    config['models_defaults'][model_type][config_name] = new_value

    with open(Path(HOME_PATH, 'config.yml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    return config['models_defaults'][model_type]


def remove_model_from_config(model_type: str):
    with open(Path(HOME_PATH, 'config.yml'), 'r') as file:
        config = yaml.safe_load(file)

    config['models_defaults'].pop(model_type, None)

    with open(Path(HOME_PATH, 'config.yml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    return list(config['models_defaults'].keys())