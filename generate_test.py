from utils import *
from transformers import AutoTokenizer

if __name__ == "__main__":
    model_path = "<<PATH TO TRAINED MODEL>>"
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    max_length = 30
    df = get_test_results(model_path, tokenizer, max_length)