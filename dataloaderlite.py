import os
import torch
import numpy as np 

def load_tokens(filename):
    file_path = os.path.join("edu_fineweb10B", filename)
    npt = np.load(file_path)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# 对于大数据集无法一次性加载进内存，分成许多个小文件，每个文件包含一部分数据，采用如下方式加载数据。
class DataLoaderLite:
    """
    params:
    - batch_size: number of samples per batch
    - context_length: number of tokens per sample
    - rank: rank of the current process
    - world_size: total number of processes
    
    for use:
    .......
    for epoch in range(self.epochs):
        dataloader.set_epoch(epoch)
        for i in range(num_batches):
            input, target = dataloader.next_batch()
    .......
    """
    def __init__(self, batch_size, context_length, rank, world_size, split, shuffle=True, seed=1234, dataset_path="edu_fineweb10B"):
        self.batch_size = batch_size
        self.context_length = context_length
        self.rank = rank
        self.world_size = world_size
        self.dataset_path = dataset_path
        self.split = split
        assert self.split in ["train", "val"]
        shards = os.listdir(dataset_path)
        shards = [s for s in shards if split in s]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.reset()


    def reset(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.shards), generator=g).tolist()
            self.shards = [self.shards[i] for i in indices]
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.context_length * self.rank 
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.reset()

    def next_batch(self):
        batch_size, context_length = self.batch_size, self.context_length
        if self.current_position + (batch_size * context_length * self.world_size + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = batch_size * context_length * self.rank
        buf = self.tokens[self.current_position : self.current_position+batch_size*context_length+1]
        input = (buf[:-1]).view(batch_size, context_length) # inputs
        target = (buf[1:]).view(batch_size, context_length) # targets
        # advance the position in the tensor
        self.current_position += batch_size * context_length * self.world_size # 每隔world_size个进程，取一个batch
        # if loading the next batch would be out of bounds, advance to next shard
        return input, target
    

