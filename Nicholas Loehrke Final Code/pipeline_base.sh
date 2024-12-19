#!/bin/bash

#####################################################################################
#                                    TRAINING                                       #
#####################################################################################

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="runs/$TIMESTAMP"
INPUT_REVIEWS="data/raw_reviews.csv"
HIDDEN_STATE_EMBEDDING_SIZE=$(python -c "from config.config import config; print(config.ffnn.hidden_state_embedding_size)")

mkdir -p "$OUTPUT_DIR"

cp config/config.py $OUTPUT_DIR/config.py

# Create training/test splits
python -m scripts.create_train_test_datasets $INPUT_REVIEWS --train_outfile $OUTPUT_DIR/raw_training_reviews.csv --test_outfile $OUTPUT_DIR/raw_test_reviews.csv --test_size 0.15

# Preprocess
python -m scripts.preprocess_reviews $OUTPUT_DIR/raw_training_reviews.csv -o $OUTPUT_DIR/preprocessed_training_reviews.csv 

# Generate training embeddings
python -m scripts.generate_embeddings fasttext/crawl-300d-2M-subword.bin $OUTPUT_DIR/preprocessed_training_reviews.csv -o $OUTPUT_DIR/training_embeddings.pkl

# Train RNN
python -m scripts.train_rnn $OUTPUT_DIR/training_embeddings.pkl -o $OUTPUT_DIR/rnn.model

# Extract hidden states
python -m scripts.extract_hidden_states $OUTPUT_DIR/rnn.model $OUTPUT_DIR/training_embeddings.pkl -d $HIDDEN_STATE_EMBEDDING_SIZE -o $OUTPUT_DIR/training_hidden_states.pkl

# Train FFNN
python -m scripts.train_ffnn $OUTPUT_DIR/training_hidden_states.pkl -o $OUTPUT_DIR/ffnn.model

#####################################################################################
#                                   EVALUATION                                      #
#####################################################################################

# Generate test embeddings
python -m scripts.generate_embeddings fasttext/crawl-300d-2M-subword.bin $OUTPUT_DIR/raw_test_reviews.csv -o $OUTPUT_DIR/test_embeddings.pkl

# Evaluate RNN
python -m scripts.evaluate_rnn $OUTPUT_DIR/rnn.model $OUTPUT_DIR/test_embeddings.pkl | tee $OUTPUT_DIR/rnn_eval.txt

# Extract hidden states
python -m scripts.extract_hidden_states $OUTPUT_DIR/rnn.model $OUTPUT_DIR/test_embeddings.pkl -d $HIDDEN_STATE_EMBEDDING_SIZE -o $OUTPUT_DIR/test_hidden_states.pkl

# Evaluate FFNN
python -m scripts.evaluate_ffnn $OUTPUT_DIR/ffnn.model $OUTPUT_DIR/test_hidden_states.pkl | tee $OUTPUT_DIR/ffnn_eval.txt