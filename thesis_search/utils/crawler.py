import requests
import time
from typing import Iterable, Tuple

from bs4 import BeautifulSoup
from .models import Thesis


class HSEVKRScrapper:
    """
    Класс для скрэппинга страниц с темами вкр
    """

    request_url = 'https://www.hse.ru/n/vkr/api/'
    default_headers = {
        'authority': 'www.hse.ru',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7,ko;q=0.6',
        'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/117.0.0.0 Safari/537.36',
        'x-portal-language': 'ru',
    }

    def __init__(self, headers=None):
        self.__session = requests.session()
        self.headers = headers

    @property
    def headers(self):
        return self.__headers

    @headers.setter
    def headers(self, headers):
        if not headers:
            self.__headers = HSEVKRScrapper.default_headers
        else:
            self.__headers = headers

    def _get_data_unit(self, vkr_page: str) -> str:
        page = self.__session.get(vkr_page).text
        soup = BeautifulSoup(page, features="lxml")
        unit = soup.find("body", recursive=True).get('data-unit')
        return unit

    def parse_thesis_page(self, thesis_id: int) -> Tuple[str, str]:
        url = HSEVKRScrapper.request_url + str(thesis_id)
        response = self.__session.get(url, headers=self.headers).json()
        if response['file']:
            response['file'] = response['file']['url']
        return response['file'], response['abstract']

    def get_page_theses(self, unit: str, page: str, sleep: float = 1) -> Iterable[Thesis]:
        params = {
            'unit': unit,
            'page': page,
        }
        response = self.__session.get(HSEVKRScrapper.request_url, params=params, headers=self.headers)
        page_theses = response.json()['data']

        theses = []
        for thesis_dict in page_theses:
            thesis = self._create_thesis_info(thesis_dict)
            time.sleep(sleep)
            thesis.file, thesis.abstract = self.parse_thesis_page(thesis.thesis_id)
            theses.append(thesis)
        return theses

    def _create_thesis_info(self, thesis_dict: dict) -> Thesis:
        thesis = Thesis()
        thesis.thesis_id = thesis_dict['id']
        thesis.year = thesis_dict['year']
        thesis.title = thesis_dict['title']
        thesis.student = thesis_dict['student']
        thesis.supervisors = [sup['name'] for sup in thesis_dict['supervisors']]
        thesis.learn_program = thesis_dict['learnProgram']['title']
        return thesis

    def _get_total_pages(self, unit: str) -> int:
        response = self.__session.get(HSEVKRScrapper.request_url, params={'unit': unit}, headers=self.headers)
        return response.json()['totalPages']

    def crawl(self, vkr_page, pagelimit: int = 30, sleep: float = 0):
        data_unit = self._get_data_unit(vkr_page)
        pagelimit = min(self._get_total_pages(data_unit), pagelimit)
        theses = []
        for page in range(1, pagelimit + 1):
            theses.extend(self.get_page_theses(data_unit, str(page), sleep))
        return theses
