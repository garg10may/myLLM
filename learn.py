import math

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(123)


# Step 1: read the training text.
with open("theVerdict.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Step 2: turn text into GPT-2 token ids.
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# Step 3: choose a small learning setup.
max_length = 100
batch_size = 4

if len(token_ids) <= max_length:
    raise ValueError("The text is too short for this demo.")


# Step 3: build training examples.
# Each input has 8 tokens and each target is shifted by 1 token.
input_sequences = []
target_sequences = []

for start_index in range(0, len(token_ids) - max_length, max_length):
    chunk = token_ids[start_index : start_index + max_length + 1]
    input_sequences.append(torch.tensor(chunk[:-1], dtype=torch.long))
    target_sequences.append(torch.tensor(chunk[1:], dtype=torch.long))

all_inputs = torch.stack(input_sequences)
all_targets = torch.stack(target_sequences)


# Step 4: split into train and validation sets.
split_index = int(len(all_inputs) * 0.9)
train_inputs = all_inputs[:split_index]
train_targets = all_targets[:split_index]
val_inputs = all_inputs[split_index:]
val_targets = all_targets[split_index:]

if len(train_inputs) == 0 or len(val_inputs) == 0:
    raise ValueError("The dataset is too small for the current train/validation split.")


# Step 5: look at the first batch manually.
sample_inputs = train_inputs[:batch_size]
sample_targets = train_targets[:batch_size]

print("First batch of input token ids:")
print(sample_inputs)
print()

print("First batch of target token ids:")
print(sample_targets)
print()


# Step 6: create token embeddings and position embeddings.
vocab_size = 50257
d_model = 64

token_embedding_layer = nn.Embedding(vocab_size, d_model)
position_embedding_layer = nn.Embedding(max_length, d_model)

token_embeddings = token_embedding_layer(sample_inputs)
position_ids = torch.arange(max_length)
position_embeddings = position_embedding_layer(position_ids)
input_embeddings = token_embeddings + position_embeddings

print(f"sample input shape: {sample_inputs.shape}")
print(f"token embeddings shape: {token_embeddings.shape}")
print(f"position embeddings shape: {position_embeddings.shape}")
print(f"input embeddings shape: {input_embeddings.shape}")
print()


# Step 7: reproduce chapter3 style causal attention on the sample embeddings.
W_query = nn.Linear(d_model, d_model, bias=False)
W_key = nn.Linear(d_model, d_model, bias=False)
W_value = nn.Linear(d_model, d_model, bias=False)

queries = W_query(input_embeddings)
keys = W_key(input_embeddings)
values = W_value(input_embeddings)

attention_scores = queries @ keys.transpose(1, 2)
causal_mask = torch.triu(torch.ones(max_length, max_length, dtype=torch.bool), diagonal=1)
attention_scores = attention_scores.masked_fill(causal_mask, -torch.inf)
attention_weights = torch.softmax(attention_scores / math.sqrt(d_model), dim=-1)
context_vectors = attention_weights @ values

print(f"attention scores shape: {attention_scores.shape}")
print(f"attention weights shape: {attention_weights.shape}")
print(f"context vectors shape: {context_vectors.shape}")
print()


# Step 8: build a small GPT-like model without custom classes.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.ModuleDict(
    {
        "token_embedding": nn.Embedding(vocab_size, d_model),
        "position_embedding": nn.Embedding(max_length, d_model),
        "blocks": nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(d_model, d_model),
                        "k_proj": nn.Linear(d_model, d_model),
                        "v_proj": nn.Linear(d_model, d_model),
                        "out_proj": nn.Linear(d_model, d_model),
                        "norm1": nn.LayerNorm(d_model),
                        "norm2": nn.LayerNorm(d_model),
                        "ff_in": nn.Linear(d_model, 256),
                        "ff_out": nn.Linear(256, d_model),
                        "dropout": nn.Dropout(0.1),
                    }
                )
                for _ in range(2)
            ]
        ),
        "final_norm": nn.LayerNorm(d_model),
        "lm_head": nn.Linear(d_model, vocab_size),
    }
)
model = model.to(device)

num_heads = 4
head_dim = d_model // num_heads

if d_model % num_heads != 0:
    raise ValueError("d_model must be divisible by num_heads.")


# Step 9: define the forward pass as regular functions.
def run_multi_head_attention(x, block):
    batch_size, seq_len, _ = x.shape

    queries = block["q_proj"](x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    keys = block["k_proj"](x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    values = block["v_proj"](x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    scores = queries @ keys.transpose(-2, -1)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, -torch.inf)
    weights = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
    weights = block["dropout"](weights)

    context = weights @ values
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return block["out_proj"](context)


def model_forward(input_ids):
    batch_size, seq_len = input_ids.shape

    token_embeddings = model["token_embedding"](input_ids)
    position_ids = torch.arange(seq_len, device=input_ids.device)
    position_embeddings = model["position_embedding"](position_ids)
    x = token_embeddings + position_embeddings

    for block in model["blocks"]:
        attention_output = run_multi_head_attention(x, block)
        x = block["norm1"](x + block["dropout"](attention_output))

        feed_forward = block["ff_out"](F.relu(block["ff_in"](x)))
        x = block["norm2"](x + block["dropout"](feed_forward))

    x = model["final_norm"](x)
    return model["lm_head"](x)


# Step 10: run one forward pass before training.
example_logits = model_forward(sample_inputs.to(device))
print(f"logits shape before training: {example_logits.shape}")
print()


# Step 11: train the model.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


def evaluate_loss(inputs_tensor, targets_tensor):
    model.eval()
    losses = []

    with torch.no_grad():
        for start_index in range(0, len(inputs_tensor), batch_size):
            input_batch = inputs_tensor[start_index : start_index + batch_size].to(device)
            target_batch = targets_tensor[start_index : start_index + batch_size].to(device)

            logits = model_forward(input_batch)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_batch.reshape(-1))
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


for epoch in range(3):
    permutation = torch.randperm(len(train_inputs))
    train_loss_total = 0.0
    train_batches = 0

    for start_index in range(0, len(train_inputs), batch_size):
        batch_indices = permutation[start_index : start_index + batch_size]
        input_batch = train_inputs[batch_indices].to(device)
        target_batch = train_targets[batch_indices].to(device)

        logits = model_forward(input_batch)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_batch.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_total += loss.item()
        train_batches += 1

    train_loss = train_loss_total / train_batches
    val_loss = evaluate_loss(val_inputs, val_targets)
    print(f"epoch {epoch + 1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")

print()


# Step 12: generate a few tokens from the trained model.
prompt_ids = tokenizer.encode("Every effort moves you", allowed_special={"<|endoftext|>"})
generated = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

model.eval()
with torch.no_grad():
    for _ in range(12):
        model_input = generated[:, -max_length:]
        logits = model_forward(model_input)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token_id), dim=1)

generated_text = tokenizer.decode(generated.squeeze(0).tolist())

print("Generated text:")
print(generated_text)
