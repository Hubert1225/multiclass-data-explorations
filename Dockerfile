FROM python:3.10

ARG USERNAME=mde
ARG UID=1000
ARG GID=1000

WORKDIR /code

RUN groupadd --gid $GID $USERNAME \
    && useradd --uid $UID --gid $GID -m $USERNAME \
    && chown -R $USERNAME /code

COPY poetry.lock pyproject.toml ./
ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install poetry
RUN poetry install --no-ansi --no-root

ENV PYTHONPATH='/code/src/'

EXPOSE 8888

USER $USERNAME
