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

## Datasets used as examples

_Wine_
Loaded using scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
