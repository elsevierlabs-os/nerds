from setuptools import setup, find_packages

setup(
    name="nerds",
    author="Elsevier Content & Innovation",
    install_requires=[
        'allennlp',
        'anago @ git+https://github.com/Hironsan/anago.git',
        'flair',
        'future',
        'h5py',
        'hyperopt',
        'joblib',
        'keras',
        'networkx==1.11',
        'numpy',
        'pyahocorasick',
        'pyyaml',
        'regex==2017.4.5',
        'scipy',
        'sklearn',
        'sklearn-crfsuite',
        'spacy',
        'tensorflow',
	'torch',
        'transformers',
        'pandas',
        'simpletransformers'
    ],
    tests_require=[
        'coverage',
        'nose'
    ],
    python_requires=">=3.6",
    version="1.0.0",
    packages=find_packages()
)
