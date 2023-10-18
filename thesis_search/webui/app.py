import time

from flask import Flask, request, render_template

from thesis_search import init_defaults
from thesis_search.search import search_theses

app = Flask(__name__)
db, fast_nlp, corpus, default_params, search_engines = init_defaults()

initialized_search_engines = {idx: search_engines[idx](corpus=corpus[idx], **default_params[idx])
                              for idx in search_engines}

META = {'Year': 1, 'Program': 2, 'Student': 3, 'Supervisor': 4}
N_RESULTS = 10


@app.route('/')
def home_page():  # put application's code here
    return render_template('index.html')


@app.route('/search')
def search():
    return render_template("search.html")


@app.route('/result', methods=['POST', 'GET'])
def results():
    if request.method == 'POST':
        idx_type = request.form['index']
        query = request.form['query']

        start = time.time()
        results = search_theses(query=query, nlp=fast_nlp, search_engine=initialized_search_engines[idx_type],
                                db=db, n=N_RESULTS)
        exec_time = str(round(time.time() - start, 4)) + ' s'

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
