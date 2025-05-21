# trainer/training.py
import tensorflow as tf
import logging
from trainer import config, model as model_builder, data_loader, preprocessing, utils

def run_training():
    utils.setup_logging()
    utils.set_seeds()

    logging.info("Starting training process...")

    # 1. Load data from RDF source
    logging.info("Fetching data from RDF store...")
    rdf_fetcher = data_loader.RDFDataFetcher(
        sparql_endpoint=config.SPARQL_ENDPOINT,
        query=config.SPARQL_QUERY_WORKS_AND_TEXTS
    )
    # Fetch an initial set of documents. For a truly dynamic system
    # this fetching coudl be integrated into a tf.data.Dataset.from_generator.
    # Here, we fetch once and then process.
    # You can adjust the limit in config.py
    fetched_documents = rdf_fetcher.fetch_works_and_texts(limit=config.SPARQL_QUERY_WORKS_AND_TEXTS.split("LIMIT")[-1].split()[0] if "LIMIT" in config.SPARQL_QUERY_WORKS_AND_TEXTS else 1000)

    if not fetched_documents:
        logging.error("No documents fetched. Aborting training.")
        return

    # 2. Preprocess Data
    logging.info("Initializing preprocessor...")
    text_preprocessor = preprocessing.TextPreprocessor(
        vocab_size=config.VOCAB_SIZE,
        max_seq_len=config.MAX_SEQ_LEN
    )

    # Create TensorFlow dataset. This will also fit the tokenizer on the fetched_documents.
    # Note: For very large datasets, fit tokenizer on a sample or load pre-trained.
    train_dataset = text_preprocessor.get_tf_dataset(
        fetched_documents,
        batch_size=config.BATCH_SIZE
    )

    # Get the actual vocabulary size after fitting the tokenizer
    # The Tokenizer's num_words is the max size. word_index gives the actual items.
    actual_vocab_size = len(text_preprocessor.tokenizer.word_index) + 1 # +1 for padding/OOV
    if actual_vocab_size == 1 and config.VOCAB_SIZE > 1 : # maybe only <unk> token
        logging.error("Tokenizer fitting resulted in a very small vocabulary (size {actual_vocabulary}. Check data and MIN_SENTENCE_LEN.")
        logging.error("This usually happens if no valid sequences are generated. Check MIN_SENTENCE_LEN and input text content.")
        # Check if dataset is empty 
        if tf.data.experimental.cardinality(train_dataset).numpy() == 0:
            logging.error("The training dataset is empty. Aborting training.")
            return 
        # If config.VOCAB_SIZE was small (e.g. 1), this might be okay.
        # But generally, if it's unexpectedly 1 or 2, it's an issue.
        if actual_vocab_size < 10 and config.VOCAB_SIZE > 100: # Heuristic for problem
            logging.warning(f"Actual vocabulary size ({actual_vocab_size}) is much smaller than configured ({config.VOCAB_SIZE}).")


    # Save the fitted tokenizer for inference or retraining
    text_preprocessor.save_tokenizer(config.TOKENIZER_PATH)
    logging.info(f"Effetive vocabulary size for model: {actual_vocab_size}")

    # 3. Build Model
    logging.info("Building Transformer model...")
    transformer_model = model_builder.build_transformer_lm(
        vocab_size=actual_vocab_size, # Use the actual vocab size from the tokenizer
        max_seq_len=config.MAX_SEQ_LEN,
        embed_dim=config.EMBEDDING_DIM,
        num_transformer_blocks=config.NUM_TRANSFORMER_BLOCKS,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        dropout_rate=config.DROPOUT_RATE
    )

    transformer_model.summary(print_fn=logging.info)

    # 4. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    transformer_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", # Use for integer targets
        metrics=["accuracy"]
    )
    logging.info("Model compiled.")

    # 5. Train Model
    logging.info(f"Starting model training for {config.EPOCHS} epochs...")

    # Optional: Callbacks
    callbacks = []
    # Example: EarlyStopping
    # early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # callbacks.append(early_stopping_cb)

    # Note: If your dataset is very large or comes from a generator that doesn't have a finite size easily,
    # you might need to specify `steps_per_epoch`.
    # For a dataset made from a fixed list of texts, Keras can usually infer this.
    try:
        history = transformer_model.fit(
            train_dataset,
            epochs=config.EPOCHS,
            callbacks=callbacks
            # validation_data=validation_dataset, # If you have a validation set
        )
        logging.info(f"Training completed.")
        logging.info(f"Training history: {history.history}")

        # Optionally, save the trained model
        transformer_model.save("transformer_lm_model.keras")
        logging.info("Trained model saved to transformer_lm_model.keras")

    except tf.errors.InvalidArgumentError as e:
        logging.error(f"TensorFlow InvalidArgumentError during training: {e}")
        logging.error("This can happen if vocabulary size in model doesn't match data, or due to issues with sequence lengths/padding.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)

if __name__ == '__main__':
    # This allows running training directly, but usually you'd use main.py
    run_training()
