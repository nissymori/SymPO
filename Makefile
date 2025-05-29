format:
	black .
	isort .
	mypy .


check:
	black .
	isort .
	mypy .


clean:
	rm -rf .cache
	rm -rf .DS_Store
	rm -rf .env
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .mypy_cache