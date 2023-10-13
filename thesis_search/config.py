from pathlib import Path


HOME_PATH = Path(__file__).resolve().parent.parent
DATA_FOLDER_PATH = Path(HOME_PATH, "data/")
INDEX_PATH = Path(DATA_FOLDER_PATH, "indices")
LM_PATH = Path(DATA_FOLDER_PATH, "vector_models")


DEFAULT_LMS = {
    'w2v': {'file': 'ruwikiruscorpora_upos_cbow_300_10_2021.bin',
            'source_link': 'http://vectors.nlpl.eu/repository/20/220.zip'},
    'ft': {'file': 'cc.ru.300.bin',
           'source_link': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz'}
}

INDEX_TYPES = ['bm25', 'ft', 'w2v', 'bert']