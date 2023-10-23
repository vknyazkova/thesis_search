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
docker run --name thesis_search -p 5000:5000
```
By default, container only works with bm25 index. But you can also add freq and w2v index. (You can also try to add 
fasttext and bert index, but it is most likely that you won't have enough RAM for this.) For indices that do not need downloaded
.bin files you can just add them to models in config.yml (either manually or using cli) and restart container. 
For indices like w2v and ft you will also need downloaded .bin files. There you have 2 options: download model to 
container using cli or mount directory on you computer that stores those .bin files. </br>
To download model go to ```Containers``` tab, then select ```thesis_search``` container, ```Exec``` tab and enter 
```python -m thesis_search download w2v``` (see. cli [docs](docs/cli.md#download)), then add index to config.yml
(```python -m thesis_search add-index w2v --name word2vec```) and restart container (```docker stop thesis_search```, 
```docker start thesis_search```).</br>
To mount directory stop container, remove it (```docker rm thesis_search```), 
then run it again mounting directory where you store your models to the directory in container where
code expects to find those models (by default - /usr/src/thesis_search/data/vector_models)
```docker run -v <path-to-folder-with-models>:/usr/src/thesis_search/data/vector_models -p 5000:5000 thesis_search```. 
After that add index to config and restart container
