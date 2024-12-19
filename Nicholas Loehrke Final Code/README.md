# 4940 Undergraduate Research

## Installation
- Python (tested with 3.10.6)
- requirements.txt
    - Note: most dependencies probably don't need the exact version given in 'requirements.txt'
- FastText
    - Download a pretrained model from https://fasttext.cc/docs/en/english-vectors.html and update /config/config.py to point to the download path

## Running
- Use 'pipeline_bash.sh' as a reference to creating your own pipeline scripts
- Call 'pipeline_base.sh' or a 'pipeline_base.sh' derived script in the root directory to train and evaluate the model. This script will generate a 'runs' directory and a working directory within 'runs' with the training and evaluation data
