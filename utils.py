import numpy as np
import datetime
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def create_data_loader(tokenizer,
                       max_length,
                       dataset):
  
  df = pd.read_csv(dataset)

  df = df.dropna()

   # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []

  # For every sentence...
  for _, row in df.iterrows():
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer(
          row['premise'], # Premise to encode
          row['hypothesis'],  # Hypothesis to encode.
          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
          max_length = max_length,           # Pad & truncate all sentences.
          padding = 'max_length',
          return_attention_mask = True,   # Construct attn. masks.
          truncation=True,
          return_tensors = 'pt',     # Return pytorch tensors.
          )
      
      # Add the encoded sentence to the list.    
      input_ids.append(encoded_dict['input_ids'])
      
      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(df['label'].tolist())

  # Combine the training inputs into a TensorDataset.
  dataset = TensorDataset(input_ids, attention_masks, labels)

  # Create a 90-10 train-validation split.

  # Calculate the number of samples to include in each set.
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size

  # Divide the dataset by randomly selecting samples.
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  print('{:>5,} training samples'.format(train_size))
  print('{:>5,} validation samples'.format(val_size))

  # The DataLoader needs to know our batch size for training, so we specify it 
  # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
  # size of 16 or 32.
  batch_size = 16

  # Create the DataLoaders for our training and validation sets.
  # We'll take training samples in random order. 
  train_dataloader = DataLoader(
              train_dataset,  # The training samples.
              sampler = RandomSampler(train_dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )

  # For validation the order doesn't matter, so we'll just read them sequentially.
  validation_dataloader = DataLoader(
              val_dataset, # The validation samples.
              sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )
  
  return train_dataloader, validation_dataloader

def get_test_results(model_path, tokenizer, max_length):
    dataset_path = "data/test.csv"

    model = torch.load(model_path)
    model.eval()

    epoch_pred = np.array([])
    epoch_labels = np.array([])

    # Create data loader
    df = pd.read_csv(dataset_path)

    df = df.dropna()

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for _, row in df.iterrows():
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer(
            row['premise'], # Premise to encode
            row['hypothesis'],  # Hypothesis to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = max_length,           # Pad & truncate all sentences.
            padding = 'max_length',
            return_attention_mask = True,   # Construct attn. masks.
            truncation=True,
            return_tensors = 'pt',     # Return pytorch tensors.
            )
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
      
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks)

    test_dataloader = DataLoader(
              dataset,
              batch_size = 32
          )

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids, 
                        #token_type_ids=None, 
                        attention_mask=b_input_mask,
                        return_dict=True)    

        logits = result.logits       

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        epoch_pred = np.concatenate([epoch_pred, pred_flat])
    
    df = pd.read_csv(dataset_path)
    df['prediction'] = [int(i) for i in epoch_pred.tolist()]
    df[['id', 'prediction']].to_csv("submission.csv")
    return df[['id', 'prediction']]