import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
from tqdm import tqdm
import os

# ====== Hyperparameters ======
MAX_SEQ_LENGTH = 512
EMBED_DIM = 128
NUM_HEADS = 8
HIDDEN_DIM = 512
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# ====== Data Preparation ======

# Create a set of all printable ASCII characters
all_chars = string.printable
vocab_size = len(all_chars)

# Create mappings from characters to indices and vice versa
char2idx = {ch: idx for idx, ch in enumerate(all_chars)}
idx2char = {idx: ch for idx, ch in enumerate(all_chars)}

# Load data
with open('sample_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove any characters not in our vocabulary
text = ''.join([ch for ch in text if ch in all_chars])

class ChatDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        self.data = text
        self.inputs = []
        self.targets = []
        self.prepare_data()

    def prepare_data(self):
        for i in range(len(self.data) - self.seq_length):
            seq = self.data[i:i + self.seq_length]
            target = self.data[i + 1:i + self.seq_length + 1]
            self.inputs.append([char2idx[ch] for ch in seq])
            self.targets.append([char2idx[ch] for ch in target])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )

# ====== Model Definition ======

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super(SimpleGPT, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(MAX_SEQ_LENGTH, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        positions = positions.clamp(max=MAX_SEQ_LENGTH - 1)
        x = self.embed(x) + self.pos_embed(positions)
        src_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device)
        x = self.encoder(x, mask=src_mask)
        logits = self.fc_out(x)
        return logits

# ====== Training Preparation ======

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ChatDataset(text, seq_length=32)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleGPT(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ====== Training Function ======

def train_model():
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape outputs and targets for the loss function
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    # Save the trained model
    torch.save(model.state_dict(), 'simple_gpt.pth')
    print("Training completed and model saved as 'simple_gpt.pth'.")

# ====== Chatbot Interface ======

def generate_response(model, prompt, max_length=100):
    model.eval()
    generated = [char2idx.get(ch, char2idx[' ']) for ch in prompt]
    input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            seq_length = input_ids.size(1)
            positions = torch.arange(0, seq_length, device=device).unsqueeze(0)
            positions = positions.clamp(max=MAX_SEQ_LENGTH - 1)
            x = model.embed(input_ids) + model.pos_embed(positions)
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(device)
            x = model.encoder(x, mask=src_mask)
            logits = model.fc_out(x)
            next_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)  # Shape: [batch_size, 1]
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Get the token ID as an integer
            next_token_id = next_token.item()
            if idx2char[next_token_id] == '\n':
                break

    output_text = ''.join([idx2char[idx] for idx in input_ids[0].tolist()])
    response = output_text[len(prompt):]
    return response.strip()

def chat():
    print("\nWelcome to the Simple GPT Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Append a newline to indicate the end of user input
        prompt = user_input + '\n'
        response = generate_response(model, prompt)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    model_file = 'simple_gpt.pth'
    if os.path.exists(model_file):
        print(f"Loading the model from '{model_file}'...")
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
    else:
        train_model()
    chat()
