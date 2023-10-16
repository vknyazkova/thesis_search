from collections import namedtuple
import time

from flask import Flask, request, render_template

from thesis_search import init_defaults
from thesis_search.search import search_theses

app = Flask(__name__)
db, fast_nlp, corpus, default_params, search_engines = init_defaults()
Result = namedtuple('Result', ['meta', 'title', 'abstract'])


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
        search_engine = search_engines[idx_type](corpus=corpus[idx_type], **default_params[idx_type])

        start = time.time()
        results = search_theses(query=query, nlp=fast_nlp, search_engine=search_engine,
                                db=db, n=20)
        exec_time = time.time() - start

        formated_results = []
        for res in results:
            meta = [('year', res[1]),
                    ('program', res[2]),
                    ('student', res[3]),
                    ('supervisor', res[4]),
                    ('link', res[6])]
            result = Result(meta=meta, title=res[0], abstract=res[5])
            formated_results.append(result)

    return render_template("result.html",
                           query=query,
                           n_docs=1200,
                           time=exec_time,
                           results=formated_results)


if __name__ == '__main__':
    app.run()