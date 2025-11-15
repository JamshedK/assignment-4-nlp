from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import math
import os
import torch.optim as optim
from sklearn.metrics import classification_report


# load ssh-5 datasest
def load_data():
    ds = load_dataset("SetFit/sst5")
    return ds["train"], ds["validation"], ds["test"]


def dataset_to_lists(dataset):
    # convert dataset to lists
    texts = dataset["text"]
    labels = dataset["label"]
    return texts, labels


def load_glove_embeddings(glove_path):
    """Load GloVe embeddings from file"""
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index


def build_vocab(train_texts, min_freq=1):
    counter = Counter()
    for text in train_texts:
        words = text.lower().split()
        counter.update(words)
    # filter words by min_freq
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    vocab = set(sorted(vocab))
    vocab = list(vocab)
    special_tokens = ["<pad>", "<unk>"]
    vocab = special_tokens + vocab
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return word_to_ix, vocab


def create_embedding_matrix(word_to_ix, embeddings_index, embedding_dim):
    # get the size of the vocabulary
    vocab_size = len(word_to_ix)
    # create a matrix of shape (vocab_size, embedding_dim) initialized with zeros
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # fill the matrix with the embeddings from embeddings_index
    for word, i in word_to_ix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # If word not found in GloVe, initialize with random vector
            embedding_matrix[i] = np.random.normal(scale=0.1, size=(embedding_dim,))
    return embedding_matrix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        num_classes,
        pretrained_embeddings,
        dropout=0.1,
        max_len=50,
    ):
        super(TransformerSentimentClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # Embedding layer with pretrained GloVe
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len)
        batch_size, seq_len = inputs.size()

        # Create padding mask (True for padding tokens)
        pad_token_id = 0  # <pad> token
        src_key_padding_mask = inputs == pad_token_id  # (batch_size, seq_len)

        # Get embeddings
        embedded = self.embedding(inputs)  # (batch_size, seq_len, embedding_dim)

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(
            -1, batch_size, -1
        )  # (1, batch_size, embedding_dim)
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
        embedded = torch.cat(
            [cls_tokens, embedded], dim=0
        )  # (seq_len+1, batch_size, embedding_dim)

        # Scale embeddings by sqrt(embedding_dim) as in original Transformer
        embedded = embedded * math.sqrt(self.embedding_dim)

        # Add positional encoding
        embedded = self.pos_encoder(embedded)

        # Update padding mask to account for CLS token
        cls_mask = torch.zeros(batch_size, 1, device=inputs.device, dtype=torch.bool)
        src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            embedded, src_key_padding_mask=src_key_padding_mask
        )

        # Use CLS token output for classification
        cls_output = transformer_output[0]  # (batch_size, embedding_dim)

        # Classification
        logits = self.classifier(cls_output)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs


def train_model_mini_batch(
    model,
    loss_function,
    optimizer,
    word_to_ix,
    train_text,
    label_text,
    num_epochs=50,
    batch_size=32,
    max_len=50,
):

    sequences = []
    labels = []
    for text, label in zip(train_text, label_text):
        context_idxs = [
            word_to_ix.get(w, word_to_ix["<unk>"]) for w in text.lower().split()
        ]
        # ensure we handle variable length input by padding
        if len(context_idxs) > max_len:
            context_idxs = context_idxs[:max_len]
        else:
            context_idxs = context_idxs + [word_to_ix["<pad>"]] * (
                max_len - len(context_idxs)
            )

        sequences.append(context_idxs)
        labels.append(label)

    # convert to tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create DataLoader for mini-batch training
    dataset = TensorDataset(sequences_tensor, labels_tensor)
    # shuffle and batch the data at each epoch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for context_batch, target_batch in dataloader:
            model.zero_grad()
            log_probs = model(context_batch)
            loss = loss_function(log_probs, target_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(log_probs, 1)
            correct_predictions += (predicted == target_batch).sum().item()
            total_predictions += target_batch.size(0)

        epoch_accuracy = correct_predictions / total_predictions
        losses.append(total_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
            print(f"Training Accuracy: {epoch_accuracy * 100:.2f}%")

    return losses


def evaluate_model(model, word_to_ix, test_text, test_labels, max_len=50):
    # set the model to evaluation mode
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []

    # no gradient computation during evaluation
    with torch.no_grad():
        for text, label in zip(test_text, test_labels):
            # get the indexes for the words in the text
            context_idxs = [
                word_to_ix.get(w, word_to_ix["<unk>"]) for w in text.lower().split()
            ]
            # if the length of context_idxs is greater than max_len, truncate it
            if len(context_idxs) > max_len:
                context_idxs = context_idxs[:max_len]
            # else pad it with <pad> token
            else:
                context_idxs = context_idxs + [word_to_ix["<pad>"]] * (
                    max_len - len(context_idxs)
                )
            # convert to tensor
            context_tensor = torch.tensor([context_idxs], dtype=torch.long)
            # get the model predictions
            log_probs = model(context_tensor)
            # get the predicted label
            predicted_label = torch.argmax(log_probs, dim=1).item()
            predicted_labels.append(predicted_label)
            if predicted_label == label:
                correct += 1
            total += 1
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return predicted_labels


def main(embedding_path, EMBEDDING_DIM):
    # Load data
    train_data, dev_data, test_data = load_data()
    train_texts, train_labels = dataset_to_lists(train_data)
    dev_texts, dev_labels = dataset_to_lists(dev_data)
    test_texts, test_labels = dataset_to_lists(test_data)
    print(
        f"Train size: {len(train_texts)}, Dev size: {len(dev_texts)}, Test size: {len(test_texts)}"
    )

    # Build vocabulary
    word_to_ix, vocab = build_vocab(train_texts, min_freq=1)
    print(f"Vocab size: {len(vocab)}")

    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(
        word_to_ix, embeddings_index, EMBEDDING_DIM
    )

    # Transformer hyperparameters
    nhead = 2 if EMBEDDING_DIM == 50 else 4
    num_encoder_layers = 3
    dim_feedforward = 128
    dropout = 0.1 if EMBEDDING_DIM == 50 else 0.3
    max_len = 50
    # Add 1 to max_len for CLS token
    pos_encoding_max_len = max_len + 1

    # Create Transformer model
    model = TransformerSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        num_classes=5,
        pretrained_embeddings=embedding_matrix,
        dropout=dropout,
        max_len=pos_encoding_max_len,  # Use max_len + 1 for positional encoding
    )

    print("Transformer Model Architecture:")
    print(model)
    print(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Define loss function and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Train the model
    print("Training Transformer model...")
    losses = train_model_mini_batch(
        model,
        loss_function,
        optimizer,
        word_to_ix,
        train_texts,
        train_labels,
        num_epochs=50,
        batch_size=32,
        max_len=max_len,
    )

    # Evaluate the model
    print("Evaluating model...")
    predicted_labels = evaluate_model(
        model, word_to_ix, test_texts, test_labels, max_len
    )

    # Performance metrics
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, predicted_labels, target_names=labels, digits=4
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print("Labels order: very negative, negative, neutral, positive, very positive")
    print("Rows = True labels, Columns = Predicted labels")
    print(cm)


if __name__ == "__main__":
    # NOTE: Uncomment one of the following lines to set the path for embeddings
    embedding_path = "../data/glove.6B.300d-subset.txt"
    # embedding_path = "../data/glove.6B.50d-subset.txt"

    embeddings_index = load_glove_embeddings(embedding_path)
    # NOTE:  Uncomment one of the following lines to set the embedding dimension
    EMBEDDING_DIM = 300
    # EMBEDDING_DIM = 50

    print(f"Training Transformer with {embedding_path}")
    main(embedding_path, EMBEDDING_DIM)
