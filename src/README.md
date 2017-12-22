# Scripts
One use of this folder to create pipelines, processing or analysis that may not be general enough to deserve a function and be added to the package.
In addition, all processing that indeed uses premade functions, even if it not adds anything new to the already done
The idea

## data
Include here processing and generation of 

## models
This folder may include

# Files location
There are helper io functions in the module that use simple labels to get access to many different types of data. This specifications are directly accessible and modifiable from the file shortcuts.json, located on spikelearn/data/.

# Notes on Notebooks
It may be acceptable to generate or change data or figures while exploring it on Jupyter Notebooks. Care must be taken in this cases, such as to save all taken steps in a dedicated script here thereafter.
The best practice is to delete all Jupyter-created data and/or models, and re-run directly from the script, to make sure every data, figure and result is fully reproducible from the raw data alone.
