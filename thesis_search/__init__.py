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

INDEX_TYPES = config['index_types']

MODEL_DEFAULT_PARAMS = config['models_defaults']
for m in ['w2v', 'ft']:
    if not Path(MODEL_DEFAULT_PARAMS[m]['file']).is_absolute():
        MODEL_DEFAULT_PARAMS[m]['file'] = Path(LM_PATH, MODEL_DEFAULT_PARAMS[m]['file']).resolve()
    else:
        MODEL_DEFAULT_PARAMS[m]['file'] = Path(MODEL_DEFAULT_PARAMS[m]['file'])


