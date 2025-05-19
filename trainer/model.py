# rdf_transformer_trainer/model.py
import tensorflow as tf
from tensorflow.keras import layers
from rdf_transformer_trainer import config
import numpy as np
import logging

class TokAndPosEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_seq_len, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        # mask_zero=True is important if using 0 for padding
        # It ignores embedding if it sees a value of 0, so it would be BAD for position embedding
        self.pos_emb = layers.Embedding(input_dim=max_seq_len, output_dim=embed_dim)

    # Deleted compte_mask function, wasn't referenced elsewhere
    def call(self, x):
        length = tf.shape(x)[1]
        token_embeds = self.token_emb(x) # (batch_size, seq_len, embed_dim)
        pos_embeds = x = self.pos_emb(x) # (batch_size, seq_len, embed_dim)
        return token_embeds + pos_embeds

def transformer_encoder_block(embed_dim, num_heads, ff_dim, dropout_rate=0.1, name="transformer_block"):
    inputs = layers.Input(shape=(None, embed_dim), name=f"{name}_input")
    # The mask is implicitly passed if the previous layer (PositionalEmbedding) generates one.
    # Or, it can be explicitly passed if `inputs` is a list/tuple [data, mask].
    # For self-attention, the attention_mask is derived from padding_mask automatically by MHA layer.

    # Multi-Head Attention
    # `supports_masking=True` on MHA means it will use the mask from previous layer
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate, name=f"{name}_mha"
    )(query=inputs, value=inputs, key=inputs) # Self-attention
    
    attention_output = layers.Dropout(dropout_rate, name=f"{name}_mha_dropout")(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layernorm1")(inputs + attention_output)

    # Feed Forward Network
    ffn_output = layers.Dense(ff_dim, activation="relu", name=f"{name}_ffn_dense1")(out1)
    ffn_output = layers.Dense(embed_dim, name=f"{name}_ffn_dense2")(ffn_output)
    
    ffn_output = layers.Dropout(dropout_rate, name=f"{name}_ffn_dropout")(ffn_output)
    encoder_output = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layernorm2")(out1 + ffn_output)
    
    return tf.keras.Model(inputs=inputs, outputs=encoder_output, name=name)


def build_transformer_lm(
    vocab_size,
    max_seq_len=config.MAX_SEQ_LEN,
    embed_dim=config.EMBEDDING_DIM,
    num_transformer_blocks=config.NUM_TRANSFORMER_BLOCKS,
    num_heads=config.NUM_HEADS,
    ff_dim=config.FF_DIM,
    dropout_rate=config.DROPOUT_RATE
):
    """Builds a Transformer-based language model."""
    # Input is sequence of token IDs (max_seq_len - 1 for LM task)
    inputs = layers.Input(shape=(max_seq_len - 1,), dtype="int32", name="input_token_ids")

    # Embedding with positional encoding
    embedding_layer = TokAndPosEmbedding(vocab_size, embed_dim, max_seq_len)
    x = embedding_layer(inputs) # Output has shape (batch, seq_len-1, embed_dim) and a mask

    # Transformer Encoder Blocks
    for i in range(num_transformer_blocks):
        x = transformer_encoder_block(
            embed_dim, num_heads, ff_dim, dropout_rate, name=f"transformer_block_{i}"
        )(x) # The mask is propagated through functional API

    # Output layer for language modeling (predicting the next token)
    # We want to predict a token ID for each position in the input sequence
    # For the task (input_seq[:-1], target_token), the model predicts P(token | input_seq[:-1])
    # If the task is (input_seq, target_seq where target_seq = input_seq[1:] + padding),
    # then the output layer should predict for each time step.
    # Current setup: input is seq_len-1, target is a single token.
    # So, we only need the output from the last relevant token embedding.
    # However, standard LM training predicts the next token for *each* token in the input.
    # Let's adjust this to a more standard LM: input = sequence, output = sequence of next token predictions
    # The loss function will then handle masking for padded positions.
    
    # If inputs are (batch, seq_len-1), and outputs are (batch, seq_len-1, vocab_size)
    # this means for each token in input, predict the next token.
    # The labels would be input_ids shifted by one.
    # Let's assume the data preproc gives (input_seq_len-1, single_target_token_id)
    # Then we take the output of the *last* non-padded token.
    # A GlobalAveragePooling1D or Flatten then Dense is simpler if we only want one prediction for the whole sequence.
    # For true next-token prediction at each step:
    # outputs = layers.Dense(vocab_size, name="output_logits")(x) # Shape: (batch, seq_len-1, vocab_size)
    # This would require target to be also (batch, seq_len-1)
    
    # For the current preprocessing (input_seq[:-1], output_token_id):
    # We need to get the representation for the *last input token* to predict the *next* token.
    # Since inputs are padded at the 'pre' (beginning), the last *actual* token is at the end of sequence.
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x) # (batch, embed_dim)
    # Or, if you want to use the output of the last token: x = x[:, -1, :]
    # However, GlobalAveragePooling can be more robust with padding.
    # Be careful with mask_zero=True in Embedding and how it interacts with pooling.
    # If Embedding mask_zero=True, pooling layers should respect the mask.
    # GlobalAveragePooling1D respects masks.

    outputs = layers.Dense(vocab_size, activation="softmax", name="output_softmax")(x) # (batch, vocab_size)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_language_model")
    
    logging.info("Transformer Language Model built successfully.")
    return model

# Example Usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    effective_vocab_size = config.VOCAB_SIZE # Assume this is known (e.g. from preprocessor)
    
    model = build_transformer_lm(
        vocab_size=effective_vocab_size,
        max_seq_len=config.MAX_SEQ_LEN,
        embed_dim=config.EMBEDDING_DIM,
        num_transformer_blocks=config.NUM_TRANSFORMER_BLOCKS,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM
    )
    model.summary(line_length=120)

    # Test with dummy input
    # (batch_size, max_seq_len-1)
    dummy_input = np.random.randint(0, effective_vocab_size, size=(2, config.MAX_SEQ_LEN - 1))
    predictions = model.predict(dummy_input)
    logging.info(f"Output predictions shape: {predictions.shape}") # Should be (batch_size, vocab_size)
