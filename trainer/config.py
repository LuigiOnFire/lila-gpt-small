# trainer/config.py

# SPARQL Configuration
SPARQL_ENDPOINT = "https://lila-erc.eu/sparql/lila_knowledge_base/sparql"
# Query to fetch works
SPARQL_QUERY_WORKS_AND_TEXTS = """
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX powla: <http://purl.org/powla/powla.owl#>

# get all documents in LiLa
SELECT * WHERE {
          ?libbro rdf:type  powla:Document.
            ?libbro dc:title ?title.
            } LIMIT 10
"""

SPARQL_QUERY_TEXTS = """
PREFIX co: <http://purl.org/ontology/co/core#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix ontolex: <http://www.w3.org/ns/lemon/ontolex#>
prefix lila: <http://lila-erc.eu/ontologies/lila/>
prefix powla: <http://purl.org/powla/powla.owl#>
prefix corpora: <https://lila-erc.eu/ontologies/lila_corpora/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
 
 SELECT ?label
 WHERE {
           BIND(<http://lila-erc.eu/data/corpora/Lasla/id/corpus/SenecaAd%20Lucilium%20Epistulae%20Morales> AS ?documentURI)
             
                
             ?sentenceLayer powla:hasDocument  ?documentURI.
               ?word powla:hasLayer ?sentenceLayer.
                 ?word  rdfs:label  ?label
                   
                 } limit 50
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
