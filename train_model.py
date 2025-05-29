
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
EMBED_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 2
BATCH_SIZE = 2
EPOCHS = 5

class CodeDataset(Dataset):
    def __init__(self, filepath, max_samples=None):
        self.pairs = []
        self.token2id = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self._load(filepath, max_samples)
        self._build_vocab()

    def _load(self, filepath, max_samples):
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line.strip())
                if "source" in obj and "target" in obj:
                    self.pairs.append((obj["source"], obj["target"]))

    def _build_vocab(self):
        idx = len(self.token2id)
        for src, tgt in self.pairs:
            for word in src.split() + tgt.split():
                if word not in self.token2id:
                    self.token2id[word] = idx
                    idx += 1

    def encode(self, text):
        ids = [self.token2id.get(w, self.token2id["<unk>"]) for w in text.split()]
        ids = ids[:MAX_LEN - 2]
        return torch.tensor([self.token2id["<sos>"]] + ids + [self.token2id["<eos>"]], dtype=torch.long)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return self.encode(src), self.encode(tgt)

    def __len__(self):
        return len(self.pairs)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_batch, tgt_batch

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output.transpose(0, 1))

def train(model, dataloader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :].reshape(-1)

            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    dataset = CodeDataset("data/codetotest_train.jsonl", max_samples=3000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = TransformerModel(vocab_size=len(dataset.token2id), d_model=EMBED_SIZE, nhead=NUM_HEADS, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    train(model, dataloader, optimizer, criterion)