# Contributing to DigitalTwinSim

## Setup

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov

# Frontend
cd frontend && npm install
```

## Code Style

- **Python**: PEP 8, type hints for public APIs, `async/await` throughout
- **TypeScript**: strict mode, prefer named exports, functional components

## Running Tests

```bash
# Backend (all tests)
pytest tests/ -v

# Backend (skip slow/integration)
pytest tests/ -v -m "not slow"

# Frontend
cd frontend && npm run test:ci
```

## PR Process

1. Create a feature branch from `main`
2. Make changes, add tests for new functionality
3. Ensure all tests pass locally
4. Submit PR with description of changes
5. Tests must pass in CI before merge

## Test Requirements

- New API endpoints must have corresponding test in `tests/test_api_endpoints.py`
- New pure functions should have unit tests
- Integration tests go in `tests/test_integration.py`
- Frontend component tests use Vitest + Testing Library
