# Function to calculate the accuracy of our predictions vs labels
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm
import sentencepiece
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from time import gmtime, strftime
import datetime
import random
from sklearn.metrics import classification_report
import logging
import os

from utils import *
from train import train_model_experiment, train_bert_model_experiment

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

    max_length = 50

    model = train_bert_model_experiment(max_length)

    for model_name in ["xlm-roberta-base", "distilbert-base-multilingual-cased", "xlm-roberta-large"]:
        model = train_model_experiment(model_name, max_length)