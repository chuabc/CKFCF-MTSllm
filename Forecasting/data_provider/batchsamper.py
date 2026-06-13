import math
import random
from torch.utils.data import Sampler

class FeatureBatchSampler(Sampler):
    """
    每个 batch 只来自同一个 feat_id
    Dataset __getitem__ 接收 (feat_id, time_idx)
    """
    def __init__(self, tot_len: int, num_features: int, batch_size: int, shuffle: bool = True, drop_last: bool = True):
        """
        Args:
            tot_len: 每个变量可取的时间窗口数
            num_features: 变量数（enc_in）
            batch_size: batch size
            shuffle: 是否打乱 batch 顺序
            drop_last: 是否丢弃不足 batch_size 的 batch
        """
        self.tot_len = tot_len
        self.num_features = num_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 每个变量能产生多少个 batch
        self.num_batches_per_feat = (
            tot_len // batch_size
            if drop_last
            else math.ceil(tot_len / batch_size)
        )

    def __iter__(self):
        # 变量顺序
        feat_ids = list(range(self.num_features))
        if self.shuffle:
            random.shuffle(feat_ids)

        for feat_id in feat_ids:
            # 时间窗口索引
            time_indices = list(range(self.tot_len))
            if self.shuffle:
                random.shuffle(time_indices)

            for i in range(self.num_batches_per_feat):
                start = i * self.batch_size
                end = start + self.batch_size

                batch_time_idx = time_indices[start:end]

                if len(batch_time_idx) < self.batch_size and self.drop_last:
                    continue

                # Dataset 需要的是 (feat_id, time_idx)
                yield [(feat_id, t) for t in batch_time_idx]

    def __len__(self):
        return self.num_features * self.num_batches_per_feat

