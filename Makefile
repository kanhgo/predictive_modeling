install:
	pip install --upgrade pip &&\
		pip install -r requirements-app.txt	

install-note:
	pip install --upgrade pip &&\
		pip install -r requirements-note.txt

test:
	python -m pytest -vv --cov=app test_app.py

test_note:
	papermill Diabetes_predictor_v4.ipynb /tmp/output.ipynb

lint:
	pylint --disable=R,C,W1203,W0702,W0718 app.py test_app.py

format:
	black app.py test_app.py

all: install lint format test 

all-note: install test_note
