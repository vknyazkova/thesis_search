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

DATA_FOLDER = config['folders']['data_folder']
INDEX_FOLDER = config['folders']['index_folder']
LM_FOLDER = config['folders']['lm_folder']

INDEX_TYPES = {k: v for k, v in config['models'].items()}

MODEL_DEFAULTS = config['defaults']
for m in ['w2v', 'ft']:
    if not Path(MODEL_DEFAULTS[m]['model_path']).is_absolute():
        MODEL_DEFAULTS[m]['model_path'] = Path(LM_FOLDER, MODEL_DEFAULTS[m]['model_path']).resolve()
    else:
        MODEL_DEFAULTS[m]['model_path'] = Path(MODEL_DEFAULTS[m]['model_path'])

if __name__ == '__main__':
    print(config)
