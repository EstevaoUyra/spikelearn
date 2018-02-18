# Scripts (src)
Every line of code written to process data or generate results has to be in this folder.

Also use this folder to create pipelines, processing or analysis that are not general enough to deserve a function in to the package.


## data
Include here processing steps ran upon datasets, that in turn originate novel datasets.
- Feature engineering
- Artifact filtering
- Smoothing

## models
This folder should include model-related programs such as:
- Hyperparameter searchs
- Learning curves
- Classifier comparison

## visualization
Any visualization script should have its filename beginning with its scope, followed by two underscores, and then the proper filename.

# Files location
## Load
There are helper io functions in the module that use simple labels to get access to many different types of data. This specifications are directly accessible and modifiable from the file shortcuts.json.
Models should only load from data/, not from models/, with the exception of loading optimized Hyperparameters or full Models. Processing of models's results is to be done entirely in the same script.
Visualization scripts can load from data/ and from models/s.

## Save
### data
Outputs from data have to be stored in data/{interim, processed}
### analysis
Outputs from analysis have to be stored in data/results
### visualization
Outputs from visualization should be in the corresponding folder inside reports/figures/
In addition, scripts creating more than one image as Output should create a folder with their proper filename, and output images there.

# Patterns
The following pattern will probably be repeated in many scripts to come:

- Parameters:
  - Parameters are specified both on the analysis and visualization scripts. They could be saved and loaded for clarity and simplification.
  - Common:
    - Analysis parameters: {window, train, test} size, n_splits
    - Meta-parameters: DSETS, id_variables

- Label/dataset nesting: Defines a folder to save results
  - TODO Save the folder-generating function to ease of use
  - Common: label_folder/dset_folder/

- Identifier variables nesting:
  - Variables we have interest in comparing results. 
  - Common: Regularization type and strength.


# Notes on Notebooks
It may be acceptable to generate data or figures while exploring it on Jupyter Notebooks. Care must be taken in this cases, such as to save all taken steps in a dedicated script here thereafter.
The best practice is to delete all Jupyter-created data and/or models, and re-run directly from the script, to make sure every data, figure and result is fully reproducible from the raw data alone.
