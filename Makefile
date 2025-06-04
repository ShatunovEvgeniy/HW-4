SHELL := /bin/sh

install:
	poetry install

test:
	poetry run pytest tests

lint:
	poetry run pre-commit run --show-diff-on-failure --color=always --all-files

hooks:
	poetry run pre-commit install --install-hooks

install_dataset:
	wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
	wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
	mkdir -p data
	unzip -qq images_background.zip -d data/
	unzip -qq images_evaluation.zip -d data/
	rm images_background.zip
	rm images_evaluation.zip

make train:
	poetry run python src/train.py

make inference:
    poetry run python src/inference.py
