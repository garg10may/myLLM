import torch
import torch.nn as nn
import tiktoken
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from chapter2 import create_dataloader_v1


class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, d_ff):
        super().__init__()
        if d_in != d_out:
            raise ValueError("d_in and d_out must match for this TransformerBlock.")
        self.attention = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_out, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_out),
        )
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention block with residual connection
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward block with residual connection
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_in, d_out, context_length, dropout, num_heads, num_layers, d_ff):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_in)
        self.pos_embedding = nn.Embedding(context_length, d_in)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_in, d_out, context_length, dropout, num_heads, d_ff) 
              for _ in range(num_layers)]
        )
        
        self.final_norm = nn.LayerNorm(d_out)
        self.output_head = nn.Linear(d_out, vocab_size)
        self.context_length = context_length
    
    def forward(self, x):
        b, seq_len = x.shape
        
        # Token and positional embeddings
        token_emb = self.token_embedding(x)
        pos_indices = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(pos_indices)
        
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Transformer blocks
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_head(x)
        
        return logits


def evaluate_loss(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            b, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(b * seq_len, vocab_size)
            targets_flat = target_ids.view(b * seq_len)

            loss = loss_fn(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader is empty. Reduce batch_size or adjust split.")

    return total_loss / num_batches


def generate_sample_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    device,
    temperature=0.8,
    top_k=40,
):
    was_training = model.training
    model.eval()

    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -model.context_length:]
            logits = model(x_cond)
            next_token_logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                top_logits, top_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
                probs = torch.softmax(top_logits / temperature, dim=-1)
                sampled_index = torch.multinomial(probs, num_samples=1)
                next_token_id = top_indices.gather(-1, sampled_index)
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token_id), dim=1)

    generated_text = tokenizer.decode(x.squeeze(0).tolist())

    if was_training:
        model.train()

    return generated_text


def train_gpt(
    model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    sample_prompt,
    sample_tokens,
    sample_temperature,
    sample_top_k,
    num_epochs,
    learning_rate,
    device,
):
    """
    Train the GPT model
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    
    if len(train_dataloader) == 0:
        raise ValueError(
            "train_dataloader is empty. Reduce batch_size, reduce max_length, "
            "or set drop_last=False when creating the dataloader."
        )
    if len(val_dataloader) == 0:
        raise ValueError(
            "val_dataloader is empty. Reduce batch_size or adjust validation split."
        )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_dataloader):
            num_batches += 1
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Reshape for loss computation
            b, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(b * seq_len, vocab_size)
            targets_flat = target_ids.view(b * seq_len)
            
            # Compute loss
            loss = loss_fn(logits_flat, targets_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}")
        
        train_loss = total_loss / num_batches
        val_loss = evaluate_loss(model, val_dataloader, loss_fn, device)
        sample_output = generate_sample_text(
            model=model,
            tokenizer=tokenizer,
            prompt=sample_prompt,
            max_new_tokens=sample_tokens,
            device=device,
            temperature=sample_temperature,
            top_k=sample_top_k,
        )
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        print(f"Sample Output: {sample_output}\n")


if __name__ == "__main__":
    # Hyperparameters
    torch.manual_seed(123)
    
    vocab_size = 50257  # GPT-2 tokenizer
    d_in = 768
    d_out = 768
    context_length = 100
    dropout = 0.1
    num_heads = 12
    num_layers = 12
    d_ff = 3072  # 4 * d_out
    
    learning_rate = 1e-4
    num_epochs = 200
    batch_size = 32
    sample_prompt = "Every effort moves you"
    sample_tokens = 20
    sample_temperature = 0.8
    sample_top_k = 40
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load training data
    with open('theVerdict.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = tiktoken.get_encoding('gpt2')
    
    full_dataloader = create_dataloader_v1(
        text,
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        shuffle=False,
        drop_last=False,
    )

    full_dataset = full_dataloader.dataset
    dataset_size = len(full_dataset)
    if dataset_size < 2:
        raise ValueError("Dataset is too small to split into train and validation sets.")

    val_size = max(1, int(0.1 * dataset_size))
    train_size = dataset_size - val_size
    if train_size == 0:
        train_size = 1
        val_size = dataset_size - 1

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(123),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    # Train
    train_gpt(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        sample_prompt=sample_prompt,
        sample_tokens=sample_tokens,
        sample_temperature=sample_temperature,
        sample_top_k=sample_top_k,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
    )
    
    # Save model
    torch.save(model.state_dict(), 'gpt_model.pth')
    print("Model saved to gpt_model.pth")
