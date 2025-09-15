install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt	

test:
	python -m pytest -vv --cov=hello test_app.py

test_note:
	python -m pytest --nbval Diabetes_predictor_v4.ipynb

lint:
	pylint --disable=R,C app.py

format:
	black app.py