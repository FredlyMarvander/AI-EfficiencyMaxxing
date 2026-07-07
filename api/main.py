"""Legacy module kept so `python -m api.main` runs the batch agent."""

from main import main


if __name__ == "__main__":
    raise SystemExit(main())
