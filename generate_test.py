from utils import *
from transformers import AutoTokenizer

if __name__ == "__main__":
    model_path = "model/xlm-roberta-large-2021_08_04__14_34_28.pt"
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    max_length = 30
    df = get_test_results(model_path, tokenizer, max_length)