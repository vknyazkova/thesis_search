import sqlite3
import os
from typing import Iterable, Union, Tuple
from dataclasses import asdict

from .models import Thesis


class DBHandler:

    def __init__(self, db_path: Union[os.PathLike, str]):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cur = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def add_supervisors(self, supervisors: Iterable[str]):
        supervisors = [(sup, ) for sup in supervisors]
        self.cur.executemany('''
            INSERT or IGNORE
            INTO supervisors (name)
            VALUES (?)''', supervisors)
        self.conn.commit()

    def add_program(self, program: str):
        self.conn.execute('''
            INSERT or IGNORE
            INTO programs (name)
            VALUES (?)''', (program, ))
        self.conn.commit()

    def add_supervising_info(self, thesis_id: int, supervisors: Iterable[str]):
        self.add_supervisors(supervisors)
        for sup in supervisors:
            info = {"id": thesis_id, "sup": sup}
            self.cur.execute('''
                INSERT INTO supervising_info (thesis_id, supervisor_id)
                VALUES (
                :id,
                (SELECT id FROM supervisors WHERE name = :sup)
                )''', info)
        self.conn.commit()

    def add_thesis(self, thesis_info: Thesis):
        thesis_info = asdict(thesis_info)
        self.cur.execute('''
            INSERT INTO theses (id, title, text, student, program_id, year)
            VALUES (
            :thesis_id,
            :title,
            :abstract,
            :student,
            (SELECT id FROM programs WHERE programs.name = :learn_program),
            :year
            )''', thesis_info)
        self.conn.commit()

    def add_file(self, thesis_id: int, filelink: str):
        self.cur.execute('''
            INSERT INTO files (thesis_id, link)
            VALUES (?, ?)''', (thesis_id, filelink))
        self.conn.commit()

    def add_thesis_info(self, text_info: Thesis):
        self.add_program(text_info.learn_program)
        self.add_supervising_info(text_info.thesis_id, text_info.supervisors)
        self.add_thesis(text_info)
        if text_info.file:
            self.add_file(text_info.thesis_id, text_info.file)
        self.conn.commit()

    def get_raw_texts(self) -> Iterable[Tuple[int, str]]:
        self.cur.execute('''
            SELECT id,  text
            FROM theses''')
        return self.cur.fetchall()

    def add_lemmatization(self, theses_id: Iterable[int], lemmatized: Iterable[str]):
        self.cur.executemany('''
            UPDATE theses
            SET lemmatized = (?)
            WHERE id = (?)''', zip(lemmatized, theses_id))
        self.conn.commit()

    def get_thesis_info(self, thesis_id: int) -> Tuple[str, int, str, str, str, str, str]:
        self.cur.execute('''
            SELECT theses.title, theses.year, programs.name, theses.student, supervisors.name, theses.text, files.link
            FROM theses
            LEFT JOIN programs
            ON programs.id = theses.program_id
            LEFT JOIN files
            ON files.thesis_id = theses.id
            LEFT JOIN supervising_info
            ON supervising_info.thesis_id = theses.id
            LEFT JOIN supervisors
            ON supervisors.id = supervising_info.supervisor_id
            WHERE theses.id = (?)''', (thesis_id, ))
        return self.cur.fetchone()

    def get_lemmatized_texts(self):
        self.cur.execute('''
            SELECT id, lemmatized
            FROM theses''')
        return self.cur.fetchall()
