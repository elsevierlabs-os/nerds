# nerds
![nerds logo](nerds.png)

# How to set up a DEV environment

Required Python version >= 3.6

## Setting up the environment with `pipenv`

`pipenv` is a utility that manages virtual environments and `pip` dependencies at the same time. To install it, navigate to the project's root directory and run:

```
pip3 install pipenv
```

This will make sure that `pipenv` uses your latest version of Python3, which is hopefully 3.6 or higher. Please refer to the [official website](https://docs.pipenv.org/) for more information on `pipenv`.

A Makefile has been created for convenience, so that you can install the project dependencies, download the required models, test and build the tool easily. Note that this is the preferred environment setup approach, the `Pipfile` and `Pipfile.lock` files ensure that you automatically have access to the installed packages in `requirements.txt` after you do a `make install` (see below).

## Setting up the environment using `conda`

Alternatively, if you are using the [Anaconda distribution of Python](https://www.anaconda.com/), you can also use `conda` to create an environment using the following command:

```
conda create -n nerds python=3.6 anaconda
```

You can then enter the newly created conda environment using the following command. After you run the various `make ...` commands, the packages listed in `requirements.txt` and the downloaded models will only be visible inside the `nerds` environment. This approach is usually preferred since it can help prevent version collisions between different environments, at the cost of more disk space.

```
conda activate nerds
```

and exit the environment using the following command.

```
conda deactivate
```

## Makefile specifications

To install all of the required packages for development and testing run:

```
make install
```

The tool will not run without an English language model and a tagger. To download [spacy's English language model](https://spacy.io/usage/models) and [NLTK's default tagger](https://www.nltk.org/api/nltk.tag.html#nltk.tag.perceptron.AveragedPerceptron) run:

```
make download_models
```

To execute the unit tests run:

```
make test
```

Code quality checks can be run with:

```
make lint
```

A wheel distribution of this tool can be created with:

```
make dist
```

# How to write your own NER model

NERDS is a framework that provides some NER capabilities - among which the option of creating ensembles of NER models - but primarily made to be extended. In the following sections we take a look at the basic data exchange classes, and how you can use them to create your own models.

## Understanding the main data exchange classes

There are 3 main classes in the `nerds.core.model.input.*` package that are used in our NER models: `Document`, `Annotation` and `AnnotatedDocument`.

A `Document` class is the abstract representation of a raw document. It should always implement the `plain_text_` attribute, that returns the plain text representation of the object, as it's the one where we are going to perform NER. Therefore, whenever we want to process any new type of document format - XML, PDF, JSON, brat, etc. - the only requirement is to write an adapter that reads the file(s) from an input directory and transforms them to `Document` objects. The default `Document` object works seamlessly with `.txt` files.

The `Annotation` class contains the data for a single annotation. This is the text (e.g. "fox"), the label (e.g. "ANIMAL") and the offsets that correspond to offsets in the `plain_text_` representation of a `Document` (e.g. 40-42).

> **Important to note**: The offsets is a 2-tuple of integers that represent the position of the first and the last character of the annotation. Be careful, because some libraries end the offset one character **after** the final character i.e. at `start_offset + len(word)`. This is not the case with us, we currently end the offsets at **exactly** the final character i.e. at `start_offset + len(word) - 1`.

Finally, the `AnnotatedDocument` class is a combination of `Document` and a list of `Annotation`, and it can represent two things:

*  Ground truth data (e.g. brat annotation files).
*  Predictions on documents after they run through our NER models.

The `AnnotatedDocument` class exposes the `annotated_text_` attribute which returns the plain text representation of the document with inline annotations.

## Extending the base model class

The basic class that every model needs to extend is the `NERModel` class in the `nerds.core.model.ner.base` package. The model class implements a `fit - transform` API, similarly to `sklearn`. To implement a new model, one must extend the following methods at minimum:

*  `fit`: Trains a model given a list of `AnnotatedDocument` objects.
*  `transform`: Gets a list of `Document` objects and transforms them to `AnnotatedDocument`.
*  `save`: Disk persistence of a model.
*  `load`: Disk persistence of a model.

Please note that **all** of the class methods, utility functions, etc. should operate on `Document` and `AnnotatedDocument` objects, to maintain compatibility with the rest of the framework. The only exception is "private" methods used internally in classes.

# Running experiments

So, let's assume you have a dataset that contains annotated text. If it's in a format that is already supported (e.g. [brat](http://brat.nlplab.org/standoff.html)), then you may just load it into `AnnotatedDocument` objects using the built-in classes. Otherwise, you will have to extend the `nerds.core.model.input.DataInput` class to support the format. Then, you may use the built-in NER models (or create your own) either alone, or in an ensemble and evaluate their predictive capabilities on your dataset.

In the `nerds.core.model.evaluate` package, there are helper methods and classes to perform k-fold cross-validation. Please, refer to the `nerds.examples` package where you may look at working code examples with real datasets.

# Contributing to the project

New models and input adapters are always welcome. Please make sure your code is well-documented and readable. Before creating a pull request make sure:

* `make test` shows that all the unit test pass.
* `make lint` shows no Python code violations.
