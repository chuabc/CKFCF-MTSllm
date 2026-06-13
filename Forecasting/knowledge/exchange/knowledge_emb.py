import torch
import pandas as pd
from transformers import LlamaTokenizer, LlamaModel
import json
# 读取文本
itr_num =2
df_know = pd.read_csv(f"CKFF/Data_all/Filtered_knowledge/exchange/AUD_USD_itr{itr_num}.csv")
df_know['pred_date'] = pd.to_datetime(df_know['pred_date'])
# tokenizer
tokenizer = LlamaTokenizer.from_pretrained("llama-2-7b")
model = LlamaModel.from_pretrained("llama-2-7b")
model.eval()
wte = model.embed_tokens
hidden_dim = wte.embedding_dim
PREFIX_TEXT = (
    "Exchange rate time series forecasting.\n"
    "AUD/USD denotes the exchange rate quoted as the amount of USD per 1 AUD.\n"
)

SUFFIX_TEXT = "The historical data are:\n"

prefix_ids = tokenizer(PREFIX_TEXT, add_special_tokens=False)["input_ids"]
suffix_ids = tokenizer(SUFFIX_TEXT, add_special_tokens=False)["input_ids"]

knowledge_ids_list = []
date_list = []
knowledge_spans_list = []
for t, text in zip(df_know['pred_date'], df_know['knowledge']):
    date_list.append(int(pd.to_datetime(t).strftime('%Y%m%d')))
    text = text.replace("'", '"')
    knowledge_spans = []
    ids = []
    try:
        knowledge_obj = json.loads(text)
    except:
        knowledge_ids_list.append(ids)
        knowledge_spans_list.append(knowledge_spans)
        continue
    
    if not isinstance(knowledge_obj, list) or len(knowledge_obj) == 0:
        knowledge_ids_list.append(ids)
        knowledge_spans_list.append(knowledge_spans)
        continue

    PREFIX_KNOWLEDGE_TEXT = ("Reference knowledge: ")
    prefix_knowledge_ids = tokenizer(PREFIX_KNOWLEDGE_TEXT, add_special_tokens=False)["input_ids"]
    ids.extend(prefix_knowledge_ids)
    cur_len = len(prefix_ids + prefix_knowledge_ids)
    for i, k in enumerate(knowledge_obj):
        content = k.get("content", "")
        rationale = k.get("rationale", "")
        k_text = (
            f"- knowledge: {content} "
            f"(rationale: {rationale})\n"
        )
        k_ids = tokenizer(k_text, add_special_tokens=False)["input_ids"]
        start = cur_len
        end = cur_len + len(k_ids)
        knowledge_spans.append({
            "start": start,
            "end": end,
            "pred_date": t.strftime("%Y-%m-%d"),
            "knowledge_type": k.get("knowledge_type", ""),
            "content": k.get("content", "")
        })
        ids.extend(k_ids)
        cur_len = end

    knowledge_ids_list.append(ids)
    knowledge_spans_list.append(knowledge_spans)

max_knowledge_len = max(len(ids) for ids in knowledge_ids_list)

knowledge_embeddings = []

with torch.no_grad():
    for knowledge_ids in knowledge_ids_list:
        ids_pk = prefix_ids + knowledge_ids
        emb_pk = wte(
            torch.tensor(ids_pk, dtype=torch.long)
        ).detach()

        if len(knowledge_ids) < max_knowledge_len:
            pad_len = max_knowledge_len - len(knowledge_ids)
            pad_emb = torch.zeros(pad_len, hidden_dim, device=emb_pk.device, dtype=emb_pk.dtype)
            emb_pk = torch.cat([emb_pk, pad_emb], dim=0)

        emb_suffix = wte(
            torch.tensor(suffix_ids, dtype=torch.long)
        ).detach()

        emb = torch.cat([emb_pk, emb_suffix], dim=0)

        knowledge_embeddings.append(emb)

knowledge_embeddings = torch.stack(knowledge_embeddings)


date2idx = {date: i for i, date in enumerate(date_list)}


torch.save(
    date2idx,   # dict: int(YYYYMMDD) -> idx
    f"Forecasting/knowledge/exchange/itr{itr_num}/date2idx.pt"
)
torch.save(
    knowledge_embeddings,   # Tensor [N, Lk, hidden]
    f"Forecasting/knowledge/exchange/itr{itr_num}/0.pt"
)
torch.save(
    knowledge_spans_list,
    f"Forecasting/knowledge/exchange/itr{itr_num}/span_0.pt"
)