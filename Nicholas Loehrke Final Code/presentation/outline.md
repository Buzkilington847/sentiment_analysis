> # Notes
> Are Fasttext embeddings normalized??

# UT Dallas Presentation Outline

### Goal

Sentiment analysis on Amazon review data

### Model pipeline

- Topology overview ***(PICTURE)***
    - Preprocessing
    - Use Fasttext as feature extractor ***(Am I using 'feature extractor' correctly here?)***
    - Use RNN as feature extractor
    - FFNN as final classifier
- Preprocessing
    - Augmentation ***(PICTURE)***
        - Shuffling
        - Chunking
- Embeddings - Fasttext  ***(PICTURE)***
    - Shortcomings of Word2vec
    - Trained using CBOW on common crawl ***(PICTURE)***
    - $n$-gram's for OOV words
        - subword of 'refrigerator': ```(['refrigerator', '<ref', 'refr', 'efri', 'frig', 'rige', 'iger', 'gera', 'erat', 'rato', 'ator', 'tor>'], array([6315, 3998276, 
3822544, 3278539, 2069117, 3246884, 3006258,
       3159920, 2716211, 3195125, 3616757, 3672916]))```
- RNN
    - Hyperparameter's
    - Topology
        - LSTM (maybe mention GRU)
        - Fully connect layers based on last hidden state
        - Output layer (2 classes)
    - Training ***(PICTURE)***
        - Handling augmentation
        - Folds
- FFNN
    - Hyperparameter's
    - Topology
        - Input layer ordering (hidden states), padding (0-vectors)
        - Relu
        - Output layer (2 classes)
    - Training ***(PICTURE)***
        - Handling augmentation
        - Folds
- Technology
    - Python
        - pandas, numpy, scikit-learn
        - PyTorch
            - Almost entirely because GPU was easier to setup
    - Bash
- Code architecture
    - Modular Python scripts glued together with Bash
    - 
        ```
        #!/bin/bash

        #####################################################################################
        #                                    TRAINING                                       #
        #####################################################################################

        TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
        OUTPUT_DIR="runs/$TIMESTAMP"
        INPUT_REVIEWS="data/raw_reviews.csv"
        # INPUT_REVIEWS="data/raw_reviews_small.csv"
        HIDDEN_STATE_EMBEDDING_SIZE=$(python -c "from config.config import config; print(config.ffnn.hidden_state_embedding_size)")

        mkdir -p "$OUTPUT_DIR"

        cp config/config.py $OUTPUT_DIR/config.py

        # Create training/test splits
        python -m scripts.create_train_test_datasets $INPUT_REVIEWS --train_outfile $OUTPUT_DIR/raw_training_reviews.csv --test_outfile $OUTPUT_DIR/raw_test_reviews.csv --test_size 0.15

        # Preprocess
        python -m scripts.preprocess_reviews $OUTPUT_DIR/raw_training_reviews.csv -o $OUTPUT_DIR/preprocessed_training_reviews.csv --chunk 10 --max_chunks 100 --shuffle_chunk 30

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
        python -m scripts.evaluate_rnn $OUTPUT_DIR/rnn.model $OUTPUT_DIR/test_embeddings.pkl

        # Extract hidden states
        python -m scripts.extract_hidden_states $OUTPUT_DIR/rnn.model $OUTPUT_DIR/test_embeddings.pkl -d $HIDDEN_STATE_EMBEDDING_SIZE -o $OUTPUT_DIR/test_hidden_states.pkl

        # Evaluate FFNN
        python -m scripts.evaluate_ffnn $OUTPUT_DIR/ffnn.model $OUTPUT_DIR/test_hidden_states.pkl
        ```