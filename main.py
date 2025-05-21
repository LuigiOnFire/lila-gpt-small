# main.py
from trainer import training, utils
import logging

def main():
    # Setup basic logging for the main script itself
    # The training module will reconfigure it more formally if needed
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        training.run_training()
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}", exc_info=True)
        # exc_info=True will log the full traceback

if __name__ == "__main__":
    main()
