# PreCrimeBot

Python application for on-demand prediction of London crime,
my graduation project from Queen Mary, University of London.

You can find the relevant project report [here](report/report.pdf).

## Installation

This software requires use of Python 3.5+ with Pip. Once these conditions are
satisfied, you can install the application into its own virtual environment
using the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pipenv
pipenv install
```

This will install all of this project's required dependencies.

## Querying

### CLI

The simplest and most convenient way to interact with this application
using the CLI utility.

After installation and activation of the virtual environment, simply run:

```bash
python app/cli.py
```

This will start an interactive shell environment that is very straightforward to use. 

### REST API

Optionally you can also make use of the Flask powered REST API. To start it run:

```bash
python app/rest_api.py
```

This will start an HTTP server listening on `127.0.0.1:5000`.

The prediction endpoint has the following route:

```bash
GET /predict-crime/<date>/<address>
```

You can then call this endpoint using your favourite HTTP client.

For example, to find predict crime in Westminster for tomorrow, run:

```bash
CURL http://127.0.0.1:5000/predict-crime/tomorrow/westminster
```

Alternatively you can visit [http://127.0.0.1:5000/predict-crime/tomorrow/westminster](http://127.0.0.1:5000/predict-crime/tomorrow/westminster)
in your browser.

## Model retraining

Before retraining the models, you must first download the UK Police data.
This process is explained in the [data/README.md](data/README.md) file.

Once that's done, you can train the models using the training module:

```bash
python app/train.py
```

**NOTE:** This will attempt to train all the models that are defined in the
`app/classifier/classifier_factory.py` module. If you are only interested in
training some of the models, please comment out the irrelevant ones.
