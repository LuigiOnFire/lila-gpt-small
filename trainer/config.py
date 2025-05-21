# trainer/config.py

# SPARQL Configuration
SPARQL_ENDPOINT = "https://dbpedia.org/sparql" # Will be modified later for lila graph
# Query to fetch works
SPARQL_QUERY_WORKS_AND_TEXTS = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rfds: <http://www.w3.org/TR/REC-rdf-schema/#>

SELECT DISTINCT ?work ?text WHERE {
  ?work a dbo:EducationalInstitution ;
        dbo:abstract ?text .
}
LIMIT 200 # Adjust as needed
OFFSET 0  # For potential pagination
"""

# Data Loading and Preprocessing
VOCAB_SIZE = 10000   # For tokenizer
MAX_SEQ_LEN = 128    # Max input for input sequences into transformer
MIN_SENTENCE_LEN = 5 # Min number of words for word to be considered sentence

# Transformer Model Parameters
EMBEDDING_DIM = 256       # Dimension of token embeddings
NUM_TRANSFORMER_BLOCKS = 4 # Number of blocks in the transformer
NUM_HEADS = 8             # Number of attention heads
FF_DIM = 512              # Hidden layer size in ff layers in transformer
DROPOUT_RATE = 0.1

# Training Parameters
# All are subject to further tuning
BATCH_SIZE = 32 
EPOCHS = 5
LEARNING_RATE = 1e-4
BUFFER_SIZE_SHUFFLE = 1000 # 

# Logging and Reproducibility
LOG_FILE = "training.log"
RANDOM_SEED = 42

# Path for tokenizer persistence
TOKENIZER_PATH = "tokenizer.json"
