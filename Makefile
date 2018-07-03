install:
	pipenv install --dev -e .

download_models:
	pipenv run python -m spacy download en
	pipenv run python -m nltk.downloader averaged_perceptron_tagger

clean:
	-rm -rf build
	-rm -rf target
	-find . -name "__pycache__" -type d -depth -exec rm -rf {} \;

clean_test:
	-rm -rf cover
	-rm .coverage

dist:
	make clean
	make download_models
	pipenv run python setup.py bdist_wheel --dist-dir target

test:
	make clean_test
	pipenv run nosetests --with-coverage --cover-html -s -v --cover-package=nerds

lint:
	pipenv run flake8 nerds --verbose
