import torch

class KnowledgeBankDict:
    def __init__(self, base_path):
        self.base_path = base_path
        self.cache = {}   # feat_id -> tensor

    def get(self, feat_id):
        if feat_id not in self.cache:
            path = f"{self.base_path}/{feat_id}.pt"
            emb = torch.load(path, map_location="cpu")
            emb.requires_grad_(False)
            self.cache[feat_id] = emb
        return self.cache[feat_id]
    

class KnowledgeBankSingle:
    def __init__(self, base_path):
        self.base_path = base_path
        self.cur_feat_id = None
        self.cur_emb = None

    def get(self, feat_id):
        if self.cur_feat_id != feat_id:
            # 释放旧的（Python 会自动回收，Tensor 无引用后释放内存）
            self.cur_emb = None

            path = f"{self.base_path}/{feat_id}.pt"
            emb = torch.load(path, map_location="cpu")
            emb.requires_grad_(False)

            self.cur_feat_id = feat_id
            self.cur_emb = emb

        return self.cur_emb


class KnowledgeSpanBankSingle:
    def __init__(self, base_path):
        self.base_path = base_path
        self.cur_feat_id = None
        self.cur_emb = None

    def get(self, feat_id):
        if self.cur_feat_id != feat_id:
            # 释放旧的（Python 会自动回收，Tensor 无引用后释放内存）
            self.cur_emb = None

            path = f"{self.base_path}/span_{feat_id}.pt"
            emb = torch.load(path, map_location="cpu")
            # emb.requires_grad_(False)

            self.cur_feat_id = feat_id
            self.cur_emb = emb

        return self.cur_emb