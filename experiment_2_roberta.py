# Function to calculate the accuracy of our predictions vs labels
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm
import sentencepiece
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from time import gmtime, strftime
import datetime
import random
from sklearn.metrics import classification_report
import logging
import os

from utils import *
from train import train_model

if __name__ == '__main__':
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    timing = str(strftime("%Y_%m_%d__%H_%M_%S", gmtime()))
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = logging.getLogger(f"logs/base_test_{timing}.log")

    logging.basicConfig(filename = f"base_test_{timing}.log",
                        filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)

    # Set variables
    max_length = 50
    learning_rate = 1e-5
    epsilon = 1e-8 # args.adam_epsilon  - default is 1e-8.
    epochs = 3

    logging.info(f'Max length: {max_length}')
    logging.info(f'Learning rate: {learning_rate}')
    logging.info(f'Epsilon: {epsilon}')

    # Baseline model

    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

    train_dataloader, validation_dataloader = create_data_loader(tokenizer, 
                                                                max_length)

    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large",
                                                        num_labels = 3, # The number of output labels--2 for binary classification.
                                                        output_attentions = False, # Whether the model returns attentions weights.
                                                        output_hidden_states = False) # Whether the model returns all hidden-states.

    logging.info(tokenizer)
    logging.info(model)

    model.cuda()
    optimizer = AdamW(model.parameters(),
                    lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = epsilon # args.adam_epsilon  - default is 1e-8.
                    )

    # Total number of training steps is [number of batches] x [number of epochs]. 
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    model = train_model(model,
                    optimizer,
                    scheduler,
                    train_dataloader,
                    validation_dataloader,
                    epochs
                    )
                    
    model_path = f"{timing}_model.pt"
    torch.save(model, model_path)
    logging.shutdown()