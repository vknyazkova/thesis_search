from dataclasses import dataclass
from typing import Iterable

import progressbar


@dataclass
class Thesis:
    thesis_id: int = None
    year: int = None
    title: str = None
    student: str = None
    supervisors: Iterable[str] = None
    learn_program: str = None
    abstract: str = None
    text_lemmatized: str = None
    file: str = None


# взято со стэковерфлоу)
class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


