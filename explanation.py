import torch  # Core PyTorch library for tensor computations and neural networks
import torch.nn as nn  # Provides neural network layers and modules
import torch.nn.functional as F  # Functional interface for neural network layers
import torch.optim as optim  # Optimization algorithms like SGD, Adam, etc.
from torch.utils.data import Dataset, DataLoader  # Utilities for dataset handling and batching
import string  # For string manipulation and character sets
from tqdm import tqdm  # Progress bar library for loops
import os  # Operating system interfaces

# ====== Hyperparameters ======
MAX_SEQ_LENGTH = 512   # Maximum sequence length for positional embeddings
EMBED_DIM = 128        # Dimension of the token embeddings
NUM_HEADS = 8          # Number of attention heads in the Transformer
HIDDEN_DIM = 512       # Dimension of the feedforward network in the Transformer
NUM_LAYERS = 2         # Number of Transformer encoder layers
BATCH_SIZE = 64        # Batch size for training
EPOCHS = 10            # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for the optimizer

# ====== Data Preparation ======

# Create a set of all printable ASCII characters
all_chars = string.printable  # Includes digits, letters, punctuation, etc.
vocab_size = len(all_chars)   # Total number of unique characters in the vocabulary

# Create mappings from characters to indices and vice versa
char2idx = {ch: idx for idx, ch in enumerate(all_chars)}  # Maps each character to a unique index
idx2char = {idx: ch for idx, ch in enumerate(all_chars)}  # Maps each index back to its character

# Load data from a text file
with open('sample_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()  # Read the entire text file into a string

# Remove any characters not in our vocabulary to ensure consistent encoding
text = ''.join([ch for ch in text if ch in all_chars])

# Define a custom dataset class inheriting from PyTorch's Dataset
class ChatDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length  # Length of each input sequence
        self.data = text              # The entire text data
        self.inputs = []              # List to store input sequences
        self.targets = []             # List to store target sequences
        self.prepare_data()           # Prepare the data upon initialization

    def prepare_data(self):
        # Loop over the text to create input-target pairs
        for i in range(len(self.data) - self.seq_length):
            seq = self.data[i:i + self.seq_length]                      # Input sequence of length seq_length
            target = self.data[i + 1:i + self.seq_length + 1]           # Target sequence shifted by one character
            self.inputs.append([char2idx[ch] for ch in seq])            # Convert input characters to indices
            self.targets.append([char2idx[ch] for ch in target])        # Convert target characters to indices

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.inputs)

    def __getitem__(self, idx):
        # Retrieve the input-target pair at the specified index
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),   # Input tensor
            torch.tensor(self.targets[idx], dtype=torch.long)   # Target tensor
        )

# ====== Model Definition ======

# Define the SimpleGPT model class inheriting from nn.Module
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super(SimpleGPT, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)             # Token embedding layer
        self.pos_embed = nn.Embedding(MAX_SEQ_LENGTH, embed_dim)     # Positional embedding layer

        # Define a single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,            # Dimension of the embedding vector
            nhead=num_heads,              # Number of attention heads
            dim_feedforward=hidden_dim,   # Dimension of the feedforward network
            activation='gelu',            # Activation function
            batch_first=True              # Input tensors will have shape (batch_size, seq_length, embedding_dim)
        )
        # Stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)  # Final linear layer to map to vocabulary size

    def forward(self, x):
        seq_length = x.size(1)  # Get the length of the input sequence
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)  # Create position indices
        positions = positions.clamp(max=MAX_SEQ_LENGTH - 1)  # Ensure positions do not exceed MAX_SEQ_LENGTH
        x = self.embed(x) + self.pos_embed(positions)  # Sum token and positional embeddings

        # Generate a causal mask to prevent the model from "seeing" future tokens
        src_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device)
        x = self.encoder(x, mask=src_mask)  # Pass the embeddings through the Transformer encoder
        logits = self.fc_out(x)             # Apply the final linear layer to get logits
        return logits                       # Return the logits for each token position

# ====== Training Preparation ======

# Determine the device to run the model on (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the dataset and dataloader
dataset = ChatDataset(text, seq_length=32)  # Create a dataset with sequence length of 32
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # DataLoader for batching and shuffling

# Instantiate the model and move it to the selected device
model = SimpleGPT(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer with specified learning rate

# ====== Training Function ======

def train_model():
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        epoch_loss = 0  # Initialize the loss for this epoch
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')  # Progress bar for the current epoch

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the selected device

            optimizer.zero_grad()  # Reset gradients from the previous iteration
            outputs = model(inputs)  # Forward pass through the model

            # Reshape outputs and targets to match the loss function's expected input shape
            outputs = outputs.view(-1, vocab_size)  # Flatten the outputs to shape (batch_size * seq_length, vocab_size)
            targets = targets.view(-1)              # Flatten the targets to shape (batch_size * seq_length)

            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()                     # Backward pass to compute gradients
            optimizer.step()                    # Update model parameters

            epoch_loss += loss.item()  # Accumulate the loss
            progress_bar.set_postfix(loss=loss.item())  # Update the progress bar with the current loss

        avg_loss = epoch_loss / len(dataloader)  # Calculate the average loss for the epoch
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')  # Print the average loss

    # Save the trained model's state dictionary to a file
    torch.save(model.state_dict(), 'simple_gpt.pth')
    print("Training completed and model saved as 'simple_gpt.pth'.")

# ====== Chatbot Interface ======

def generate_response(model, prompt, max_length=100):
    model.eval()  # Set the model to evaluation mode
    # Convert the prompt into a list of character indices
    generated = [char2idx.get(ch, char2idx[' ']) for ch in prompt]
    # Create a tensor from the indices and add a batch dimension
    input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():  # Disable gradient calculation for inference
        for _ in range(max_length):
            seq_length = input_ids.size(1)  # Current length of the input sequence
            positions = torch.arange(0, seq_length, device=device).unsqueeze(0)  # Positional indices
            positions = positions.clamp(max=MAX_SEQ_LENGTH - 1)  # Ensure positions do not exceed MAX_SEQ_LENGTH

            # Get embeddings and sum them
            x = model.embed(input_ids) + model.pos_embed(positions)
            # Generate a causal mask for the sequence
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(device)
            # Pass through the Transformer encoder
            x = model.encoder(x, mask=src_mask)
            logits = model.fc_out(x)  # Get logits from the final linear layer

            # Get logits for the last token in the sequence
            next_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            probabilities = F.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1)  # Shape: [batch_size, 1]
            input_ids = torch.cat([input_ids, next_token], dim=1)  # Append the new token to the input sequence

            # Get the token ID as an integer
            next_token_id = next_token.item()
            if idx2char[next_token_id] == '\n':  # Stop if a newline character is generated
                break

    # Convert the sequence of indices back into characters
    output_text = ''.join([idx2char[idx] for idx in input_ids[0].tolist()])
    response = output_text[len(prompt):]  # Extract the generated response by removing the prompt
    return response.strip()  # Return the response without leading/trailing whitespace

def chat():
    print("\nWelcome to the Simple GPT Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()  # Get input from the user
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input:
            continue  # Skip empty inputs

        # Append a newline to indicate the end of user input
        prompt = user_input + '\n'
        response = generate_response(model, prompt)  # Generate a response using the model
        print(f"Assistant: {response}")  # Display the assistant's response

# Entry point of the script
if __name__ == "__main__":
    model_file = 'simple_gpt.pth'  # Filename for the saved model
    if os.path.exists(model_file):
        print(f"Loading the model from '{model_file}'...")
        # Load the model's state dictionary
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()  # Set the model to evaluation mode
    else:
        train_model()  # Train the model if a saved model is not found
    chat()  # Start the chatbot interface
