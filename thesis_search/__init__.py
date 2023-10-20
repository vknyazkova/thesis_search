from pathlib import Path

import yaml


HOME_PATH = Path(__file__).resolve().parent.parent

with open(Path(HOME_PATH, 'config.yml'), 'r') as file:
    config = yaml.safe_load(file)

for t, folder in config['folders'].items():
    if not Path(folder).is_absolute():
        config['folders'][t] = Path(HOME_PATH, folder).resolve()
    else:
        config['folders'][t] = Path(folder)

DATA_FOLDER_PATH = config['folders']['data_folder']
INDEX_PATH = config['folders']['index_folder']
LM_PATH = config['folders']['lm_folder']

DEFAULT_LMS = config['default_vector_models']

INDEX_TYPES = config['index_types']

if __name__ == '__main__':
    ...
