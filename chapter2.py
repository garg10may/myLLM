import os
import importlib
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        #max_length -> context size
        #stride -> step size between chunks
        self.input_ids = []
        self.target_ids = []
        
        #tokenize the text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length

        #Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1] #predict one new chunk at a time
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

if __name__ == "__main__":
    with open('theVerdict.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    dataloader = create_dataloader_v1(
        text, batch_size=8, max_length=4, stride=4, shuffle=False
    )

    inputs, targets = next(iter(dataloader))

    vocab_size = 50257 #gpt2 tokenizer
    otput_dim = 256 #our choice
    context_length = 4 #our choice, what we called as max_length previously

    token_embedding_layer = torch.nn.Embedding(vocab_size, otput_dim)
    position_embedding_layer = torch.nn.Embedding(context_length, otput_dim)

    token_embeddings = token_embedding_layer(inputs)  #convert token ids to embeddings
    pos_embeddings = position_embedding_layer(torch.arange(context_length)) #convert position ids to embeddings
    input_embeddings = token_embeddings + pos_embeddings
    print(inputs.shape)
    print(token_embedding_layer.weight.shape)
    print(token_embeddings.shape)
    print(pos_embeddings.shape)
    print(input_embeddings.shape)

