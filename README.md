# About this corpus
This corpus consists of annotations of students' theses, which were written on 4th year in Higher School of Economics 
on 3 programs: [Fundamental and Computational Linguistics](https://www.hse.ru/en/ba/ling/), 
[Applied Mathematics and Information Science](https://www.hse.ru/en/ba/ami/) and 
[Software Engineering](https://www.hse.ru/en/ba/se/). Besides annotation itself corpus stores its information: 
title, year, student name, supervisor name and the program. Some theses also have a link to the full text, if it was 
publicly available on hse.ru. </br>
All texts were collected using self-written [web-scrapper](thesis_search/utils/crawler.py) and then stored in sql 
[database](docs/theses_database.png). </br>
Annotations are indexed in 5 different ways: frequency index, bm25, word2vec, fasttext and bert. Frequency and bm25 
indices have multiple implementations (using dictionary, matrices and ready-made libraries)


# How to install
## Using python
__Clone repository:__

either using command line 
```
git clone git@github.com:vknyazkova/thesis_search.git <local-path>
```
or using github desktop: ```Current repository``` -> ```Add``` ->```Clone repository...``` -> ```URL``` -> enter repo path https://github.com/vknyazkova/thesis_search -> select local path -> ```Clone``` </br>

__Open terminal in repository's root directory__

__Install libraries:__ 
```
pip install -r requirements.txt
```

__[Download](docs/cli.md#download) vector models__ 
```
python -m thesis_search download w2v
python -m thesis_search download ft
```
or [change config](docs/cli.md#change-model-config) to match path to the existing vector models 
```
python -m thesis_search change-model-config w2v model_path <absolute-path-to-w2v-model>
python -m thesis_search change-model-config ft model_path <absolute-path-to-ft-model>
```
(or do nothing and just follow hints during execution)

__Search__

either using web interface:
```
python -m thesis_search.webui.app
```
or using cli [search](docs/cli.md#search):
```
python -m thesis_search search QUERY --idx-type IDX-TYPE --n N --style STYLE
```

## Using docker
If you have docker installed, you can run web app simply running the following in the terminal:
```shell
docker run --name thesis_search -p 5000:5000 vknyazkova/thesis_search:base
```
By default, container only works with bm25 index. But you can also add freq and w2v index. (You can also try to add 
fasttext and bert index, but it is most likely that you won't have enough RAM for this.) For indices that do not need downloaded
.bin files you can just add them to models in config.yml (either manually or using cli) and restart container. 
For indices like w2v and ft you will also need downloaded .bin files. There you have 2 options: download model to 
container using cli or mount directory on you computer that stores those .bin files. </br>
- To download model 
  - go to ```Containers``` tab, then select ```thesis_search``` container, ```Exec``` tab and enter 
  ```python -m thesis_search download w2v``` (see. cli [docs](docs/cli.md#download))
    (or just write this command in another terminal window (not where you run docker) after ```docker exec thesis_search``` 
    like: ```docker exec thesis_search python -m thesis_search download w2v```)
  - then add index to config.yml (```python -m thesis_search add-index w2v --name word2vec```)
  - and restart container (either using restart button in docker desktop or running in terminal
    ```docker stop thesis_search```, ```docker start thesis_search```).</br>
- To mount directory 
  - stop container (```docker stop thesis_search```)
  - remove it (```docker rm thesis_search```), 
  - then run it again mounting directory where you store your models to the directory in container where code 
  expects to find those models (by default - /usr/src/thesis_search/data/vector_models)
  ```docker run -v <path-to-folder-with-models>:/usr/src/thesis_search/data/vector_models -p 5000:5000 --name thesis_search vknyazkovs/thesis_search:base```
  - check that filename in model_path for that index in config coincides with the filename in your folder 
  (```python -m thesis_search show-config```)
  - if not change this path to the name of your model ```python -m thesis_search change-model-config w2v model_path <your-model-name>```
  - After that add index to config (```python -m thesis_search add-index w2v --name word2vec```) and restart container

# Project structure 
- Folder [data](/data) stores all data files including database, precomputed indices and files with statistics. It 
also includes folders from config.yml, where program expects to find indices and vector models' .bin files
- Folder [docs](/docs) contains some documentation files (for cli and config.yml)
- In [scripts](/scripts) folder you can find jupyter notebooks that were used to collect corpus and count statistics
- Package [thesis_search](/thesis_search) stores all source code for this project
  - for command line interface see [cli](/thesis_search/cli)
  - for web interface see [webui](/thesis_search/webui)
  - package [search_models](/thesis_search/search_models) consists of [base](/thesis_search/search_models/base), where
  abstract classes are implemented, and of [indexing](/thesis_search/search_models/indexing), where you can find 
  implementations of concrete index types
- [utils](/thesis_search/utils) contains all other modules that are used in the project (including database handler and 
web scrapper)