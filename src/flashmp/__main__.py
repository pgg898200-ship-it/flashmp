"""Enable `python -m flashmp` CLI invocation."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
