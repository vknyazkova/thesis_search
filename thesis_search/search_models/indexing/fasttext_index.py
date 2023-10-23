__all__ = ['FastTextSearch']

import gzip
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Union

import fasttext
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..base.embedding_search import EmbeddingSearch
from ...utils.models import MyProgressBar
from ... import HOME_PATH


class FastTextSearch(EmbeddingSearch):

    def __init__(self,
                 corpus: pd.DataFrame,
                 model_name_: str,
                 model_path: Union[str, os.PathLike],
                 index_folder_: Union[str, os.PathLike],
                 similarity_metric='cosine'):
        """
        Class that implements search based on fasttext language model
        Args:
            corpus: pandas dataframe with two columns: text ids and texts themselves
            model_name_: string with model name (ex.:  'cc.ru.300.bin')
            model_path: path to vector model file
            index_folder_: folder where precomputed index should be stored
            similarity_metric: either 'cosine' or 'dot-prod'
        """
        super().__init__(corpus, model_name_, similarity_metric)

        self.model_path = model_path
        self.register_model()

        index_path = Path(index_folder_, f'{self.model_name}_index.txt').resolve()
        if not Path(index_path).exists():
            self.index = self.compute_index()
            self.save_index(index_path)
        else:
            self.index = self.load_index(index_path)

    @staticmethod
    def download_model(url: str, dest_path: Union[os.PathLike, str]) -> Union[str, os.PathLike]:
        print(f'Скачиваю модель {url}')
        dest_path = Path(dest_path)
        dest_folder = dest_path.parent
        dest_folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(HOME_PATH)) as tmpdirname:
            gz_path, _ = urllib.request.urlretrieve(url, Path(tmpdirname, 'model.gz'),
                                                    reporthook=MyProgressBar())
            with gzip.open(gz_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return dest_path

    def register_model(self):
        if self.model_name not in FastTextSearch.loaded:
            model = self.load_model(self.model_path)
            FastTextSearch.loaded[self.model_name]['model'] = model
            FastTextSearch.loaded[self.model_name]['ref_count'] = 1
        else:
            FastTextSearch.loaded[self.model_name]['ref_count'] += 1

    def load_model(self, model_path: Union[str, os.PathLike]):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError
        return fasttext.load_model(str(model_path))

    def compute_index(self) -> np.ndarray:
        print('Индексирую с помощью fasttext')
        index = []
        for doc in tqdm(self.text):
            doc = doc.replace('\n', ' ')  # потому что фасттекст ругается на \n
            doc_vector = self.model.get_sentence_vector(doc)
            index.append(doc_vector)
        return np.array(index)

    def vectorize(self, text: str) -> np.ndarray:
        return self.model.get_sentence_vector(text).reshape(-1, 1)
