install:
	pip install -e .

download_models:
	python -m spacy download en
	python -m nltk.downloader averaged_perceptron_tagger

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
	python setup.py bdist_wheel --dist-dir target 

test:
	make clean_test
	nosetests --with-coverage --cover-html -s --verbosity=2 --cover-package=nerds

lint:
	flake8 --ignore=W605,W504 --verbose nerds
