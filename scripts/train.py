"""
End-to-end training entrypoint.

Run:  python -m scripts.train
"""
import logging

from src.nlp.train_greenwashing import train as train_greenwashing
from src.satellite.forest_classifier import train as train_forest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train")


def main():
    log.info("=" * 60)
    log.info("STEP 1/2: Training forest classifier (fast, CPU)")
    log.info("=" * 60)
    train_forest()

    log.info("=" * 60)
    log.info("STEP 2/2: Training greenwashing classifier (needs torch)")
    log.info("=" * 60)
    train_greenwashing()

    log.info("All models trained.")


if __name__ == "__main__":
    main()
