import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple

import lightning as L
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(../lit-gpt/tokenizer.py)
model = AutoModelForCausalLM.from_pretrained(../lit-gpt/model.py)
def chat():
    user_input = "Hello, how are you?"
    while True:
        # encoding the user input and adding end-of-string token
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        # getting model output
        with torch.no_grad():
            output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
        # decoding the output and removing the input part
        output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {output_text}")
        # next user input
        user_input = input("You: ")
if __name__ == '__main__':
    chat()
