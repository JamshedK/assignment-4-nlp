from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
import outlines
from datasets import load_dataset
import json
import time
from sklearn.metrics import accuracy_score, classification_report
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
trans_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# use the outlines to wrap the model
# https://dottxt-ai.github.io/outlines/latest/features/models/transformers/
model = outlines.from_transformers(trans_model, tokenizer)

# Define the output categories using Literal
SentimentCategory = Literal[
    "very positive", "positive", "neutral", "negative", "very negative"
]

# define the general chat messages template
messages = [
    [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Classify this sentence into 5 categories: very positive, positive, neutral, negative, very negative.\n Text: {{ text }}",
                },
            ],
        },
    ],
]

# Different LLM will use different details to support the above chat format.
# such as adding special tokens or special separators between messages.

prompt_template = tokenizer.apply_chat_template(messages, tokenize=False)[0]
outline_template = outlines.Template.from_string(prompt_template)

# Load SST-5 dataset
sst_dataset = load_dataset("SetFit/sst5")
test_dataset = sst_dataset["test"]


def run_inference():
    predictions = []
    ground_truths = []
    for example in test_dataset:
        text = example["text"]
        true_label_name = example["label_text"]

        predicted_label = model(
            outline_template(text=text), SentimentCategory, max_new_tokens=10
        )

        predictions.append(predicted_label)
        ground_truths.append(true_label_name)

        print(f"Text: {text}")
        print(f"True Label: {true_label_name}")
        print(f"Predicted Label: {predicted_label}")
        print("-" * 20)

    return predictions, ground_truths


def main():
    start_time = time.time()
    predictions, ground_truths = run_inference()
    end_time = time.time()

    print(f"Total inference time: {end_time - start_time:.2f} seconds")

    print("\nClassification Report:")
    print(
        classification_report(
            ground_truths,
            predictions,
            labels=[
                "very positive",
                "positive",
                "neutral",
                "negative",
                "very negative",
            ],
        )
    )

    print("\nAccuracy Score:")
    print(accuracy_score(ground_truths, predictions))


if __name__ == "__main__":
    main()
