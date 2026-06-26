# Contributing to S-JEPA

Thank you for your interest in this project. This guide explains how to set up
the project, how to write code that matches the repository style, and how to
send your changes. Please also read the
[Code of Conduct](CODE_OF_CONDUCT.md) and the coding rules in
[docs/codingstyles.md](docs/codingstyles.md).

## Ways to contribute

- Report a bug or ask a question by opening an issue.
- Improve the documentation (`README.md`, the bilingual guides in `docs/`).
- Fix a bug or add a feature with a pull request.
- Add or improve unit tests.

## Development setup

The project uses a `src/` layout, Python 3.10 or newer, and PyTorch. We
recommend a virtual environment.

```bash
git clone <your-fork-url>
cd sjepa
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[onnx]"           # editable install with the ONNX extras
pip install pytest pytest-cov      # test tools (the dev group)
```

This installs the console commands `trainsjepa`, `evalsjepa`, `buildh5ds`,
`exportw`, and `runs`.

The project trains on CPU or GPU. Use the ready configs under `cpu/configs/`
for quick local runs and `gpu/configs/` for full-scale runs. Keep the two sets
in sync: any change to a `cpu/` config must be mirrored to its `gpu/`
counterpart.

## Running the tests

All tests live in `tests/` and run with `pytest` (configured in
`pyproject.toml`).

```bash
pytest                  # run the whole suite
pytest tests/test_model.py -q
pytest --cov=sjepa      # with coverage
```

Every pull request must keep the whole suite green. Add tests for any function
or class method you add or change. Important modules also need tests for their
accuracy and run time with the right metrics (loss, performance, etc.).

## Code style

The full rules are in [docs/codingstyles.md](docs/codingstyles.md). The key
points:

- Follow the SOLID principles. Each class has one and only one responsibility.
- Use classes and modular code for each component.
- Keep functions short: no more than 16 statements per function or method.
- Every `while` loop must have a counter so it can never run forever.
- Check the return value of each function or method before the next step.
- Write comments, docstrings, and terminal logs in **simple English** (level A),
  with words a beginner can understand.
- In comments and docstrings, use only characters that exist on an English
  keyboard. No emoji and no special symbols.
- Log the hyperparameters and every step of the program for good traceability.
- Keep the beginner guides in `docs/` in both English and French up to date.

We format with `isort` and `yapf` (line length 120), configured in
`pyproject.toml`:

```bash
isort src tests
yapf -ir src tests
```

## Commit messages

Follow the short prefixed style already used in the history:

```
<type>: <short summary in the imperative>
```

Common types: `add`, `change`, `fix`, `docs`, `check`. Examples:

```
add: online GMM dead-cluster re-seeding
fix: training/validation history columns
docs: writing of readme
```

## Pull request flow

1. Fork the repository and clone your fork locally.
2. Create a branch for your change: `git checkout -b feature/my-feature`.
3. Make your change, with tests and updated docs.
4. Run `pytest` and the formatters; make sure everything passes.
5. Commit with a clear message and push: `git push origin feature/my-feature`.
6. Open a pull request that explains what changed and why.

## Reporting bugs

When you open a bug report, please include:

- What you did (the command and the config file).
- What you expected to happen.
- What happened instead (the error message or the wrong output).
- Your environment (operating system, Python version, CPU or GPU).

Thank you for helping make S-JEPA better.
