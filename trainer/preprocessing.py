# trainer/preprocessing.py
import tensorflow as tf
import numpy as np
import logging
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from trainer import config

class TextPreprocessor:
    def __init__(self, vocab_size=config.VOCAB_SIZE, max_seq_len=config.MAX_SEQ_LEN):
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.fitted = False

    def fit_on_texts(self, texts_list):
        """Fits the tokenizer on a list of texts."""
        logging.info(f"Fitting tokenizer on {len(texts_list)} texts...")
        self.tokenizer.fit_on_texts(texts_list)
        self.fitted = True
        # Vocab size actual: self.tokenizer.num_words or len(self.tokenizer.word_index) + 1
        actual_vocab_size = len(self.tokenizer.word_index) + 1
        logging.info("Tokenizer fitted. Actual vocabulary size: {actual_vocab_size}")
        if actual_vocab_size < self.vocab_size:
            logging.warning(f"Actual vocab size ({actual_vocab_size}) is less than target ({self.vocab_size}).")
        # In case we want to save the tokenizer
        # self.save_tokenizer(config.TOKENIZER_PATH)

    def save_tokenizer(self, path):
        if not self.fitted:
            raise ValueError("Tokenizer has not been fitted yet.")
        with open(path, 'w') as f:
            json.dump(self.tokenizer.to_json(), f)
        logging.info(f"Tokenizer saved to {path}")

    def load_tokenizer(self, path):
        with open(path, 'r') as f:
            tokenizer_json = json.load(f)
            self.tokenizer = tf.keras.preprocessing.text.tokenizer.from_json(tokenizer_json)
        self.fitted = True
        self.vocab_size = self.tokenizer.num_words if self.tokenizer.num_words else len(self.tokenizer.word_index) + 1
        logging.info(f"Tokenizer loaded from {path}. Vocab size: {self.vocab_size}")


    def texts_to_sequences(self, texts_list):
        """Converts a list of texts to sequences of token IDs."""
        if not self.fitted:
            raise ValueError("Tokenizer has not been fitted. Call fit_on_texts() or load_tokenizer() first.")
        return self.tokenizer.texts_to_sequences(texts_list)

    def create_lm_dataset_from_texts(self, texts_list):
        """
        Crates a language modeling dataset (input_sequence, target_sequence)
        from a list of raw texts.
        Each text is tokenized, and then split into multiple overlapping sequences.
        Example: "this is a sentence" ->
          (input: [tok(this)], target: [tok(is)])
          (input: [tok(this), tok(is)], target: [tok(a)])
          ...
        Simplified for LM: input_seq = seq[:-1], output_seq = seq[1:]
        """
        if not self.fitted:
            self.fit_on_texts(texts_list) # Auto-fit if not already done

        tokenized_texts = self.texts_to_sequences(texts_list)

        input_sequences = []
        output_sequences = []

        for token_list in tokenized_texts:
            if len(token_list) < config.MIN_SENTENCE_LEN: # Skip very short sentences
                continue
            # Create overlapping sequences for language modelling
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_seq_partial = n_gram_sequence[:-1]
                output_seq_partial = n_gram_sequence[-1]

                # Ensure input is not empty
                if not input_seq_partial:
                    continue
                
                input_sequences.append(input_seq_partial)
                output_sequences.append(output_seq_partial)

            if not input_sequences:
                logging.warning("No sequences generated. Texts might be too short or tokenizer issue.")
                # Return empty tensors with expected shapes to prevent crashes downstream
                # Pad an empty sequence to max_seq_len-1
                dummy_input = pad_sequences([[]], maxlen=self.max_seq_len -1, padding='pre')
                # Output is a single token, but we will one-hot encode later, so shape issues are tricky
                # For now let's assume this means an empty batch will be produced.
                return tf.data.Dataset.from_tensor_slices(
                        (tf.zeros((0, self.max_seq_len -1 ), dtype=tf.int32),
                            tf.zeros((0,), dtype=tf.int32)) # Output is single token ID
                )

            # Pad input sequences
            padded_input_sequences = pad_sequences(input_sequences,
                                                    max_len = self.max_seq_len - 1, # -1 because target is the next token
                                                    padding='pre',
                                                    truncating='pre')
           
            # Convert output to numpy array
            output_tokens = np.array(output_sequences)

            dataset = tf.data.Dataset.from_tensor_slices((padded_input_sequences, output_tokens))
            return dataset

        def get_tf_dataset(self, rdf_texts_data, batch_size=config.BATCH_SIZE, shuffle=True):
            """
            Creates a complete tf.data pipeline from RDF fetched data.
            rdf_texts_data: list of dictionaries [{'work_uri': ..., 'text_content': ...}, ...]
            """
            all_texts = [item['text_content'] for item in rdf_texts_data]
            if not all_texts:
                logging.error("No texts provided to create dataset.")
                # Return an empty dataset to avoid crashing
                return tf.data.Dataset.from_tensor_slices(
                        (tf.zeros((0, self.max_seq_len -1 ), dtype=tf.int32),
                        tf.zeros((0,), dtype=tf.int32))
                ).batch(batch_size)


            # Fit tokenizer on all available textss if not already fitted
            # For very large datasets, consider fitting on a representative sample
            # or loading a pre-trained tokenizer.
            if not self.fitted:
                self.fit_on_texts(all_texts)
            # This processes all texts into one large dataset.
            # For very large dynamic data, you'd use tf.data.Dataset.from_generator
            # with the data_loader yielding texts one by one.
            # Here, we simplify by processing the initial batch of texts from RDF.
                                        
            dataset = self.create_lm_dataset_from_texts(all_texts)

            if shuffle:
                dataset = dataset.shuffle(buffer_size=config.BUFFER_SIZE_SHUFFLE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            logging.info("TensorFlow dataset created.")
            return dataset

# Example Usage (typically integrated into the training script)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sample_texts = [
        "This is the first sentence for training.",
        "Another sentence follows.",
        "And a third one to make the vocabulary larger."
    ]
    # Simmulated fetched data
    simulated_rdf_data = [{"work_uri": f"ex:doc{i}", "text_content": text} for i, text in enumerate(sample_texts)]

    preprocessor = TextPreprocessor(vocab_size=50, max_seq_len=10)

    # The get_tf_dataset method will fit the tokenizer internally if not already fitted
    tf_dataset = preprocessor.get_tf_dataset(simulated_rdf_data, batch_size=2)

    logging.info(f"Effective Vocab Size after fitting: {len(preprocessor.tokenizer.word_index) +1}")

    for input_batch, target_batch in tf_dataset.take(2): # Take a few batches to inspect
        logging.info(f"Input Batch Shape: {input_batch.shape}") # (batch_size, max_seq_len-1)
        logging.info(f"Target Batch Shape: {target_batch.shape}") # (batch_size, max_seq_len-1)
        logging.info(f"Input Batch Example: \n{input_batch[0].numpy()}")
        logging.info(f"Target Bath Example: \n{target_batch[0].numpy()}")
        # To convert back to words (for inspection):
        # reverse_word_map = dict(map(reversed, preprocessor.tokenizer.word_index.items()))
        # input_sentence = [reverse_word_map.get(i) for i in input_batch[0].numpy() if i != 0]
        # target_word = reverse_word_map.get(target_batch[0].numpy())
        # logging.info(f"Example Input Decoded: {' '.join(filter(None,input_sentence))}")
        # logging.info(f"Example Target Decoded: {target_word}")

    # Example of saving and loading tokenizer
    # preprocessor.save_tokenizer("temp_tokenizer.json")
    # new_preprocessor = TextPreprocessor(vocab_size=50, max_seq_len=10)
    # new_preprocessor.load_tokenizer("temp_tokenizer.json")
    # logging.info(f"Loaded tokenizer vocab size: {len(new_preproessor.tokenizer.word_index)+1}")
