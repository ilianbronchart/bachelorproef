# Controlled Experiment

## Running the application:

The database is prepopulted with the necessary data. To run the application to label data using the context of this experiment, do the following:

1. Create a `.env` file in `experiments/controlled_experiment` directory with the following content:
```env
CHECKPOINTS_PATH=../../checkpoints
SRC_PATH=../../src
```

2. From the root of the project:
```bash
poetry install
cd experiments/controlled_experiment
poetry run fastapi run ../../src/main.py
```