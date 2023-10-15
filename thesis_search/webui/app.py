from flask import Flask, request, render_template

from thesis_search import init_defaults

app = Flask(__name__)
db, fast_nlp, corpus, default_params, search_engines = init_defaults()


@app.route('/')
def home_page():  # put application's code here
    return render_template('index.html')


@app.route('/search')
def search():
    return render_template("search.html")


@app.route('/result', methods=['POST', 'GET'])
def results():
    if request.method == 'POST':
        index = request.form['index']
        query = request.form['query']
        print(index, query)
        meta = [
            ('year', 2022),
            ('student', 'Гордеев Никита Владимирович')
        ]
        title = 'Автоматическое определение жанра песни по тексту и музыкальным метаданным: корпусное исследование'
        annotation = ('Автоматическое определение музыкальных жанров композиций - одна из актуальнейших задач music '
                      'information retrieval. Достижения в данной области широко используются в области музыкального '
                      'стриминга: информация о жанрах часто применяется в рекомендательных системах, автоматическом '
                      'составлении плейлистов. В данной работе предлагаются обученные на материалах двух корпусов '
                      'модели, которые распознают музыкальные жанры англоязычных песен по их текстовому наполнению и '
                      'характеризующих звучание метаданным.')

    return render_template("result.html",
                           query=query,
                           n_docs=1200,
                           time='5мс',
                           meta=meta,
                           title=title,
                           annotation=annotation)


if __name__ == '__main__':
    app.run()