# CLI documentation

# search
Search documents that match query
```shell
python -m thesis_search search query --idx-type --n --style 
```
```query``` - query (if query has more than 1 word use quotes)</br>
```--idx-type``` - index type (for ex. ```bm25```) </br>
```--n``` - number of documents in the result </br>
```--style``` - style for showing results (plain text - ```text``` or table - ```table```)</br>
  
# show-config
Show current configurations of the models. _(these exactly configs can be changed using change-model-config command)_
```shell
python -m thesis_search show-config
```

# change-model-config
Change configuration of the model
```shell
python -m thesis_search change-model-config idx-type config-name config-value
```
```idx-type``` - model (for ex. ```w2v```) which configs you are going to change
```config-name``` - name of the configuration
```config-value``` - new value for selected configuration

# remove-indices
Remove indices from the configs (which means that removed indices won't be used in the project)
```shell
python -m thesis_search remove-indices idx-types
```
```idx-types``` - indices that you want to remove space-separated (for ex. ```w2v ft```)

# add-index
What indices to include in the project
```shell
python -m thesis_search add-index idx-type --name
```
```idx-type``` - index type to include in project 
```--name``` - full name for index, if None idx-type will be used as name (for ex. for w2v index name can be word2vec)

# download
Download pretrained embedding model
```shell
python -m thesis_search download idx-type --source-link
```
```idx-type``` - index type for which you need to download pretrained model (can be only ```w2v``` or ```ft```)
```--source-link``` - download link for pretrained model, if None will be used default link from configs. 
This option is quite limited: </br>
```ft``` - only link to .gz model, that can be loaded using fasttext library </br>
```w2v``` - only link for downloading .zip where .bin is stored

# stats
Show statistics about implemented indices (time and memory)
```shell
python -m thesis_search stats
```