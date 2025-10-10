Training history helper

Purpose

This repository now includes `training_history.py`, a small helper that appends a JSON record for every training run to `training_history.jsonl`.

How it works

- Each time you finish a training run (in your notebook or script), call append_run(params, metrics, history, model_paths).
- Records are appended as newline-delimited JSON (JSON Lines). This makes it safe to append from multiple runs and easy to read with tools.

Usage in `model.ipynb`

The final save cell (Step 13) was updated to:
- Save the final model and weights to files.
- Prepare `params` and `metrics` dictionaries (extend these with your hyperparameters).
- Call `append_run(...)` to persist the run to `training_history.jsonl`.

Example (from notebook):

    from training_history import append_run
    append_run(params=params, metrics=metrics, history=history_combined, model_paths=model_paths)

Inspecting history

You can load all records in Python:

    from training_history import load_history
    records = load_history()
    print(len(records))
    print(records[-1])

Or view the file directly (it is newline-delimited JSON):

    training_history.jsonl

Notes and next steps

- If you run training many times, the history file may grow large; you can keep it under version control by ignoring `training_history.jsonl` and storing it externally.
- If you prefer CSV or a database, I can add exporters (CSV/SQLite) or a small web UI to browse runs.

If you want, I can also:
- Add a small helper that prints a compact table of recent runs.
- Add CLI utilities to filter/search runs by parameter values.

