"""User-facing PPA report entry point.

The implementation remains compatible with the existing Level 0 report data
contract while the schema migration is handled separately.
"""

from .model_report import main


if __name__ == "__main__":
    main()
