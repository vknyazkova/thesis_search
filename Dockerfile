FROM python:3.10-slim

WORKDIR /usr/src/thesis_search

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# add all repo
COPY . .

VOLUME ["/usr/src/thesis_search"]
CMD ["python", "-m", "thesis_search.webui.app", "bm25", "w2v"]

EXPOSE 5000


