import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Union, Iterable

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from spacy.language import Language

from ..base.embedding_search import EmbeddingSearch
from ... import HOME_PATH
from ...utils.models import MyProgressBar


class Word2VecSearch(EmbeddingSearch):

    def __init__(self, corpus: pd.DataFrame,
                 model_name_: str,
                 nlp_: Language,
                 model_path: Union[str, os.PathLike],
                 index_folder_: Union[str, os.PathLike],
                 similarity_metric='cosine'):
        """
        Class that implements search based on word2vec language model
        Args:
            corpus: pandas dataframe with two columns: text ids and texts themselves
            model_name_: string with model name (ex.:  'cc.ru.300.bin')
            nlp_: spacy pipeline
            model_path: path to vector model file
            index_folder_: folder where precomputed index should be stored
            similarity_metric: either 'cosine' or 'dot-prod'
        """
        super().__init__(corpus, model_name_, similarity_metric)
        self.nlp = nlp_
        self.model_path = model_path
        self.register_model()

        index_path = Path(index_folder_, f'{model_name_}_index.txt').resolve()
        if not Path(index_path).exists():
            self.index = self.compute_index()
            self.save_index(index_path)
        else:
            self.index = self.load_index(index_path)

    @staticmethod
    def download_zip_model(url: str,
                           dest_path: Union[os.PathLike, str]) -> Path:
        """
        Downloads zip archived model, unpacks and saves to dest_path
        Args:
            url: url where download model from
            dest_path: path where to save model

        Returns: path to the model

        """
        dest_path = Path(dest_path)
        dest_folder = dest_path.parent
        dest_folder.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=str(HOME_PATH)) as tmpdirname:
            zip_path, _ = urllib.request.urlretrieve(url, Path(tmpdirname, 'model.zip'),
                                                     reporthook=MyProgressBar())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            bin_filepath = [f for f in Path(tmpdirname).iterdir() if f.suffix == dest_path.suffix][0]
            dest_path = bin_filepath.rename(dest_path)
        return dest_path

    @staticmethod
    def download_model(url: str, model_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """

        Downloads model from the url
        Args:
            url: download link
            model_path: path to the model

        Returns: path to the model

        """
        print(f'Cкачиваю модель: {url} ...')

        model_format = ''.join(Path(url).suffixes)  # zip или bin.gz

        if model_format == '.zip':
            model_path = Word2VecSearch.download_zip_model(url, model_path)
        else:
            raise NotImplementedError('Я пока умею работать только с файлами типа .zip')
        return model_path

    def register_model(self):
        if self.model_name not in Word2VecSearch.loaded:
            model = self.load_model(self.model_path)
            Word2VecSearch.loaded[self.model_name]['model'] = model
            Word2VecSearch.loaded[self.model_name]['ref_count'] = 1
        else:
            Word2VecSearch.loaded[self.model_name]['ref_count'] += 1

    def load_model(self, model_path: Union[os.PathLike, str]):
        model_path = Path(model_path)

        binary = True if model_path.suffix == '.bin' else False

        if not model_path.exists():
            print('no file')
            raise FileNotFoundError(f'Can\'t find file in the specified path {str(model_path)}')

        return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary)

    def pos_label_text(self, text: str) -> Iterable[str]:
        """
        Adds pos labels to words in the text
        Args:
            text: text to label

        Returns: pos-labelled text

        """
        labeled_text = []
        for w in self.nlp(text):
            labeled_text.append(w.text + '_' + w.pos_)
        return labeled_text

    def compute_index(self):
        print("Считаю индекс через w2v...")
        index = []
        for text in tqdm(self.text):
            labeled_text = self.pos_label_text(text)
            try:
                index.append(self.model.get_mean_vector(labeled_text))
            except Exception:
                index.append(np.zeros(shape=self.model.vector_size))
        index = np.array(index)
        return index

    def vectorize(self, text: str) -> np.ndarray:
        labeled_text = self.pos_label_text(text)
        return self.model.get_mean_vector(labeled_text).reshape(-1, 1)
