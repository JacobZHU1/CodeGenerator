import torch
import math
import torch.nn as nn
from dataset import MemoryEfficientCodeDataset
from model import LiteTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4

def load_model(checkpoint_path, vocab_size):
    model = LiteTransformer(
        vocab_size=vocab_size,
        d_model=EMBED_SIZE,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def generate_text(model, tokenizer, input_text, max_len=256):
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer.encode(input_text)
        src = torch.tensor(src_tokens, dtype=torch.long, device=DEVICE).unsqueeze(1)  # (seq_len, batch=1)

        memory = model.transformer.encoder(
            model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model)),
            src_key_padding_mask=(src == 0).transpose(0, 1)
        )

        tgt_tokens = [tokenizer.token2id["<sos>"]]
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long, device=DEVICE).unsqueeze(1)
            tgt_emb = model.pos_encoder(model.embedding(tgt_tensor) * math.sqrt(model.d_model))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(0)).to(DEVICE)

            output = model.transformer.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask,
                memory_key_padding_mask=(src == 0).transpose(0, 1)
            )
            logits = model.fc_out(output)
            next_token = logits[-1].argmax(dim=-1).item()
            if next_token == tokenizer.token2id["
            if next_token == tokenizer.token2id["<eos>"]:
                break
            tgt_tokens.append(next_token)

        id2token = {v: k for k, v in tokenizer.token2id.items()}
        return " ".join([id2token.get(tok, "<unk>") for tok in tgt_tokens[1:]])  # exclude <sos>

if __name__ == "__main__":
    # 加载数据集以获取 tokenizer
    dataset_path = "./data/codetotest_train.jsonl"
    dataset = MemoryEfficientCodeDataset(dataset_path, max_samples=1000)

    # 加载模型<eos>"]:
                break
            tgt_tokens.append(next_token)

        id2token = {v: k for k, v in tokenizer.token2id.items()}
        return " ".join([id2token.get(tok, "<unk>") for tok in tgt_tokens[1:]])  # exclude <sos>

if __name__ == "__main__":
    # 加载数据集以获取 tokenizer
    dataset_path = "./data/codetotest_train.jsonl"
    dataset = MemoryEfficientCodeDataset(dataset_path, max_samples=1000)

    # 加载模型
    model_path = "./model_epoch_4.pt"
    model = load_model(model_path, vocab_size=len(dataset.token2id))

    # 输入 C++ 函数
    cpp_code = """
    int add(int a, int b) {
        return a + b;
    }
    """

    # 生成 GTest
    generated_gtest = generate_text(model, dataset, cpp_code)
    print("==== Generated Google Test Code ====")
    print(generated_gtest)
