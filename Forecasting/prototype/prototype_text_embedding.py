import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model, LlamaTokenizer, LlamaModel

# 读取文本
df = pd.read_csv("prototype/exchange/100/reasoner_timeseries_prototype_with_text.csv")
texts = df["text_description"].tolist()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# llama
tokenizer = LlamaTokenizer.from_pretrained("/mnt/share/cjq/LLMs/llama-2-7b")
model = LlamaModel.from_pretrained("/mnt/share/cjq/LLMs/llama-2-7b")
model.to(device)
model.eval()
wte = model.embed_tokens
hidden_dim = wte.embedding_dim

embeddings = []
lengths = []
with torch.no_grad(): 
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        ids = torch.tensor(ids, device=device)

        emb = wte(ids)          # (L, hidden_dim)
        embeddings.append(emb)
        lengths.append(len(ids))

max_len = max(lengths)
print("max_len:", max_len)

padded_embeddings = []
with torch.no_grad(): 
    for emb in embeddings:
        cur_len = emb.shape[0]
        if cur_len < max_len:
            pad_len = max_len - cur_len
            pad = torch.zeros(pad_len, hidden_dim, device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=0)
        padded_embeddings.append(emb)

text_embedding_tensor = torch.stack(padded_embeddings)
text_embedding_tensor = text_embedding_tensor.cpu()
print("Final tensor shape:", text_embedding_tensor.shape)
torch.save(text_embedding_tensor, "prototype/exchange/100/reasoner_prototype_text_embeddings.pt")