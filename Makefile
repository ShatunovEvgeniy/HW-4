SHELL := /bin/sh

install:
	poetry install

test:
	poetry run pytest tests

lint:
	poetry run pre-commit run --show-diff-on-failure --color=always --all-files

hooks:
	poetry run pre-commit install --install-hooks
