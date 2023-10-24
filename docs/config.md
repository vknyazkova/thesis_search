# Configuration file 
This file explains how config.yml file is organized and what each parameter means.

## folders 
There are 3 main folders that are used in path resolving. </br>
```data_folder``` - path where project data is stored (database, other files like statistics)</br>
```index_folder``` - folder where precomputed indices stored (by default this folder is inside data_folder)</br>
```lm_folder``` - folder with .bin files of pretrained vector models </br>
It is not recommended to change those paths, but if you don't use docker and run this locally, you might want to
change lm_folder to some other folder, where you have those models already downloaded 

## models
Mapping of index-types used in this project to their names, that are displayed in the website. This mapping should 
contain only those index types, that you want to use for searching (for ex. if you want to search using only bm25 and w2v,
models should look like:
```yaml
models:
  bm25: bm25
  w2v: word2vec
```

## defaults
The default values for every index-type that is implemented in this project. Each index type has its own set of default 
parameters, but there are some repeated ones. </br>
```implementation``` - name of the class that implements this index type. This can be one of the classes that are part of
[indexing package](/thesis_search/search_models/indexing).</br>
```model_path``` - for static vector models path to their .bin file. This path can be relative (and will be resolved 
relative to ```lm_folder``` from configs) or absolute. </br> 
```source_link``` - for static vector models their download link (see [cli download](/docs/cli.md#download) docs for limitations)

