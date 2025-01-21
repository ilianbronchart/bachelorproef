check:
    poetry run ruff format
    poetry run ruff check --fix
    poetry run mypy . 

run:
    poetry run fastapi dev src/main.py
    