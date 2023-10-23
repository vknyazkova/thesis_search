__all__ = ['BertIndex']

import os
from typing import Union, List
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from spacy.language import Language

from ..base.embedding_search import EmbeddingSearch


class BertIndex(EmbeddingSearch):
    def __init__(self,
                 corpus: pd.DataFrame,
                 nlp_: Language,
                 model_name: str,
                 model_path: str,
                 index_folder_: Union[os.PathLike, str],
                 similarity_metric='cosine'):
        """
        Class that implements search based on bert language model
        Args:
            corpus: pandas dataframe with two columns: text ids and texts themselves
            model_name: string with model name (ex.:  'sbert_large_nlu_ru_index')
            nlp_: spacy pipeline
            model_path: the model id of a pretrained model hosted inside a model repo on huggingface.co.
            index_folder_: folder where precomputed index should be stored
            similarity_metric: either 'cosine' or 'dot-prod'
        """
        super().__init__(corpus, model_name, similarity_metric)
        self.model_path = model_path
        self.nlp = nlp_

        self.register_model()

        index_path = Path(index_folder_, f'{self.model_name}_index.txt').resolve()
        if not Path(index_path).exists():
            self.index = self.compute_index()
            self.save_index(index_path)
        else:
            self.index = self.load_index(index_path)

    def register_model(self):
        if self.model_name not in BertIndex.loaded:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModel.from_pretrained(self.model_path)

            BertIndex.loaded[self.model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'ref_count': 1
            }
        else:
            BertIndex.loaded[self.model_name]['ref_count'] += 1

    @staticmethod
    def mean_pooling(model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def sentences_emb(self, sentences: List[str]) -> np.ndarray:
        """
        Computes sentence embeddings
        Args:
            sentences: list of string texts

        Returns: numpy ndarray with shape = (n_sents, emb_size)

        """
        encoded_input = BertIndex.loaded[self.model_name]['tokenizer'](
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt')  # input_ids, token_type_ids, attention_mask

        with torch.no_grad():
            model_output = BertIndex.loaded[self.model_name]['model'](**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.numpy()

    def compute_index(self):
        print('Считаю индекс через bert...')
        index = []
        for doc in tqdm(self.text.tolist()):
            if len(doc) > 1:
                sentences = [sent.text for sent in self.nlp(doc).sents]
                doc_emb = self.sentences_emb(sentences).mean(axis=0)
            else:
                doc_emb = np.zeros(shape=(1024,))
            index.append(doc_emb)
        return np.array(index)

    def vectorize(self, text: str) -> np.ndarray:
        return self.sentences_emb([text]).reshape(-1, 1)
