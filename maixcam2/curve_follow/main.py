"""MaixPy entry point for the layered robot application."""

import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from robot.application import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
