install:
	pip install --upgrade pip &&\
		pip install -r requirements-app.txt	

install-note:
	pip install --upgrade pip &&\
		pip install -r requirements-note.txt

test:
	python -m pytest -vv --cov=test_app test_app.py

test_note:
	python -m pytest --nbval Diabetes_predictor_v4.ipynb

lint:
	pylint --disable=R,C app.py test_app.py

format:
	black app.py