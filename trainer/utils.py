# trainer/utils.py
import logging
import random
import numpy as np
import tensorflow as tf
from trainer import config

def setup_logging():
    """Conigures logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
        
def set_seeds(seed=config.RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # tf.config.experimental.enable_op_determinism() # For TF >= 2.8, optional idea from Gemini that might impact performance
    logging.info(f"Random seeds set to {seed}")
