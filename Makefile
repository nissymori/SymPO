format:
	black .
	isort .


check:
	black .
	isort .


clean:
	rm -rf .cache
	rm -rf .DS_Store
	rm -rf .env
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .mypy_cache