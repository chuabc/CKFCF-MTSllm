import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer, LlamaConfig, LlamaModel, LlamaTokenizer
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from models.Llama_arch import AccustumLlamaModel
from math import sqrt
class CKFCFMTSllm(nn.Module):
    
    def __init__(self, configs, device):
        super(CKFCFMTSllm, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.pred_len = configs.pred_len
        self.stride = configs.stride
        self.d_model = configs.d_model

        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        self.llm_model = AccustumLlamaModel.from_pretrained('llama-2-7b', output_attentions=True, output_hidden_states=True, local_files_only=True)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8,
            lora_alpha=16,
            lora_dropout=0.3,
            target_modules=["q_proj", "k_proj"]
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        self.d_model = configs.d_model
        self.in_layer = nn.Linear(self.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        for layer in (self.llm_model, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0
        prototypes_np = np.load("prototype/exchange/100/prototype_centers.npy")
        self.prototypes = torch.from_numpy(prototypes_np).float().to(device=device)
        self.prototype_text_embeddings = torch.load("prototype/exchange/100/reasoner_prototype_text_embeddings.pt", map_location="cpu").to(device=device)
        self.mha = nn.MultiheadAttention(embed_dim=configs.d_model, num_heads=8, batch_first=False).to(device=device)
        self.layer_norm = nn.LayerNorm(configs.d_model).to(device=device)

    def forward(self, x, knowledge_emb):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev 

        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        sim = torch.matmul(x, self.prototypes.T)
        best_idx = torch.argmax(sim, dim=-1)
        selected_text_embeddings = self.prototype_text_embeddings[best_idx]
        x_emb = self.in_layer(x).permute(1, 0, 2)
        text_patch = rearrange(selected_text_embeddings, 'b n l d -> (n l) b d')
        attn_output, _ = self.mha(query=x_emb, key=text_patch, value=text_patch)
        outputs_emb = self.layer_norm(x_emb + attn_output)
        outputs_emb = outputs_emb.permute(1, 0, 2)
        outputs_emb = torch.cat([knowledge_emb, outputs_emb], dim=1) 


        outputs = self.llm_model(inputs_embeds=outputs_emb).last_hidden_state
        outputs = self.out_layer(outputs[:, -self.patch_num:, :].reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means 

        return outputs