# multiclass-data-explorations
Data exploration pipeline for multiclass datasets.

## How to run

Build the Docker image:

```shell
docker build . -t multiclass-data-explorations
```

Reproduce the pipeline in the Docker container:

```shell
docker run --user "1000:1000" --name multiclass-data-explorations --rm -v .:/code multiclass-data-explorations poetry run dvc repro
```

Run Jupyter Lab in the Docker container:

```shell
docker run --user "1000:1000" --name multiclass-data-explorations --rm -v .:/code -p 8888:8888 multiclass-data-explorations poetry run jupyter lab --ip 0.0.0.0 --no-browser
```

and go to one of links that appear in terminal.

## Datasets used as examples

_Wine_
Loaded using scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html,
which is the copy of UCI ML Wine Data Set dataset https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
