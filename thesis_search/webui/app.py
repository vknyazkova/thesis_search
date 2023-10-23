import time
from pathlib import Path

from flask import Flask, request, render_template

from thesis_search import INDEX_TYPES, DATA_FOLDER, MODEL_DEFAULTS, INDEX_FOLDER
from thesis_search.utils.database import DBHandler
from thesis_search.search_models.search_engine import QueryError, SearchEngine

app = Flask(__name__)
db = DBHandler(Path(DATA_FOLDER, 'theses.db'))

META = {'Year': 1, 'Program': 2, 'Student': 3, 'Supervisor': 4}
N_RESULTS = 10

search_engines = {}
for idx_type in INDEX_TYPES:
    search_engines[idx_type] = SearchEngine(
            index_type=idx_type,
            implementation=MODEL_DEFAULTS[idx_type]['implementation'],
            index_folder=INDEX_FOLDER,
            data_retriever=db,
            defaults=MODEL_DEFAULTS[idx_type],
            preprocessor=MODEL_DEFAULTS[idx_type]['preprocessor_']
        )


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/search')
def search():
    return render_template("search.html", indices=INDEX_TYPES)


@app.route('/result', methods=['POST', 'GET'])
def results():
    if request.method == 'POST':
        idx_type = request.form['index']
        query = request.form['query']
        try:
            start = time.time()
            results = search_engines[idx_type].search(query, N_RESULTS)
            exec_time = str(round(time.time() - start, 4)) + ' s'
        except QueryError as e:
            return render_template("result.html",
                                   query=query,
                                   idx_type=idx_type,
                                   error=e)

        return render_template("result.html",
                               query=query,
                               idx_type=idx_type,
                               n_docs=N_RESULTS,
                               time=exec_time,
                               meta=META,
                               results=results)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
