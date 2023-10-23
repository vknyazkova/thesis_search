FROM python:3.10-slim

WORKDIR /usr/src/thesis_search

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# add all repo
COPY . .

RUN python -m thesis_search remove-indices w2v ft bert

VOLUME ["/usr/src/thesis_search"]

CMD ["python", "-m", "thesis_search.webui.app"]

EXPOSE 5000


