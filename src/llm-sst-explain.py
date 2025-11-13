import os
import time
import json
import csv
from typing import Literal
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. Setup: Environment, Model, Tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
trans_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Wrap the model with outlines to generate structured output
model = outlines.from_transformers(trans_model, tokenizer)


# 2. Define the desired structured output using Pydantic
class SentimentAnalysis(BaseModel):
    reason: str = Field()
    label: Literal["very positive", "positive", "neutral", "negative", "very negative"]


# 3. Create the prompt template
# This prompt guides the model to first reason and then provide the label in a structured format.
messages = [
    [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that analyzes sentiment.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze the sentiment of the following sentence. First, provide a detailed explanation of your reasoning, considering the specific words, phrases, and overall tone in one sentence. Then classify it into exactly one of these 5 categories: very positive, positive, neutral, negative, very negative.\n\nSentence: {{ text }}\n\ ",
                },
            ],
        },
    ],
]

prompt_template = tokenizer.apply_chat_template(messages, tokenize=False)[0]
outline_template = outlines.Template.from_string(prompt_template)

# 4. Load the dataset
sst_dataset = load_dataset("SetFit/sst5")
test_dataset = sst_dataset["test"]


def run_inference_with_explanation():
    """
    Runs inference on the test dataset, generating a sentiment analysis
    with a reason for each example.
    """
    predictions = []
    ground_truths = []
    results = []

    # Using a smaller subset for demonstration to run faster.
    # To run on the full dataset, remove the slicing [:50].
    for example in test_dataset.select(range(50)):
        text = example["text"]
        true_label_name = example["label_text"]

        # Generate structured output using outlines with Pydantic model
        json_response = model(
            outline_template(text=text),
            SentimentAnalysis,
            max_new_tokens=200,  # Increased tokens for complete reasoning
        )
        print(f"Raw JSON Response: {json_response}")
        # Parse the JSON response into a Pydantic object
        structured_response = SentimentAnalysis.model_validate_json(json_response)
        print(structured_response)

        # Access attributes directly from the Pydantic object
        predictions.append(structured_response.label)
        ground_truths.append(true_label_name)

        # Store results for CSV output
        results.append(
            {
                "text": text,
                "true_label": true_label_name,
                "predicted_label": structured_response.label,
                "reason": structured_response.reason,
                "correct": true_label_name == structured_response.label,
            }
        )

        print(f"Text: {text}")
        print(f"True Label: {true_label_name}")
        print(f"Predicted Label: {structured_response.label}")
        print(f"Reason: {structured_response.reason}")
        print("-" * 30)

    return predictions, ground_truths, results


def save_results_to_csv(results, filename="llm_sst_results.csv"):
    """
    Save the inference results to a CSV file.
    """
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["text", "true_label", "predicted_label", "reason", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\nResults saved to {filename}")


def main():
    """
    Main function to execute the inference and print evaluation metrics.
    """
    start_time = time.time()
    predictions, ground_truths, results = run_inference_with_explanation()
    end_time = time.time()

    print(f"\nTotal inference time: {end_time - start_time:.2f} seconds")

    # Save results to CSV
    save_results_to_csv(results)

    # Define label order for consistent matrix display
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]

    print("\nClassification Report:")
    print(
        classification_report(
            ground_truths,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print("\nAccuracy Score:")
    print(accuracy_score(ground_truths, predictions))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(ground_truths, predictions, labels=labels)
    print("Labels order: very negative, negative, neutral, positive, very positive")
    print("Rows = True labels, Columns = Predicted labels")
    print(cm)

    # Pretty print confusion matrix with labels
    print("\nConfusion Matrix (with labels):")
    print(f"{'':>15}", end="")
    for label in labels:
        print(f"{label:>15}", end="")
    print()

    for i, true_label in enumerate(labels):
        print(f"{true_label:>15}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>15}", end="")
        print()


if __name__ == "__main__":
    main()
