# Contributing to Rift

Thank you for your interest in contributing to Rift.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/Rift.git`
3. Install dependencies: `pip install -e ".[dev]"`
4. Set your Python path: `export PYTHONPATH=src`

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run linting: `ruff check src/ tests/`
5. Commit with a descriptive message
6. Open a Pull Request

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length: 100 characters
- Type hints are encouraged
- Docstrings for public functions and classes

## Testing

- All new features should include tests in `tests/`
- Run the full suite: `pytest tests/ -v --tb=short`
- Aim for meaningful coverage of core logic

## Areas for Contribution

- Temporal GNN extension (TGAT)
- Additional fraud patterns in the generator
- PDF export for audit reports
- Performance optimization for large graphs
- Documentation improvements
- Additional test coverage

## Reporting Issues

Please open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
