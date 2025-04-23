# simple_bpe
A simple implementation of Byte Pair Encoding

## Tokenizer Overview

The `Tokenizer` class implements a Byte-Pair Encoding (BPE) tokenization algorithm. It converts text into tokens for natural language processing tasks. The tokenizer works by:

1. Starting with individual bytes (UTF-8 encoded characters)
2. Iteratively merging the most frequent pairs of tokens
3. Building a vocabulary of these merged tokens

## Key Functions

### `train(text, vocab_size)`

This function trains the tokenizer on input text

### `encode(text)`

Converts text into a sequence of token IDs:

### `decode(ids)`

Converts token IDs back to text

### `save(file_prefix)`

Saves the tokenizer model

### `load(model_file)`

Loads a previously saved tokenizer model
