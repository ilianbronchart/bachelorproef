streamlit:
    poetry run streamlit run app/main.py

check:
    poetry run ruff format
    poetry run ruff check --fix
    poetry run mypy . 
