# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Appendix A: Introduction to PyTorch (Part 3)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# NEW imports:
import os
import inspect
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from omegaconf import OmegaConf
import math
from dataloaderlite import DataLoaderLite
from model import GPTModel
import time
import tiktoken
from model import generate_and_print_sample

# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    assert torch.cuda.is_available(), "for now we need CUDA for DDP"
    # # rank of machine running rank:0 process
    # # here, we assume all GPUs are on the same machine
    # os.environ["MASTER_ADDR"] = "localhost"
    # # any free port on the machine
    # os.environ["MASTER_PORT"] = "2048"
    # if platform.system() == "Windows":
    #     # Disable libuv because PyTorch for Windows isn't built with support
    #     os.environ["USE_LIBUV"] = "0"

    # # initialize process group
    # if platform.system() == "Windows":
    #     # Windows users may have to use "gloo" instead of "nccl" as backend
    #     # gloo: Facebook Collective Communication Library
    #     init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # else:
        # nccl: NVIDIA Collective Communication Library
    print(f"initializing process group with rank {rank} and world_size {world_size}")
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"initialized process group with rank {rank} and world_size {world_size}")

    torch.cuda.set_device(rank)

def get_lr(it, configs):
    # 1) linear warmup for warmup_iters steps
    if it < configs.warmup_steps:
        return configs.max_lr * (it+1) / configs.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > configs.max_steps:
        return configs.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - configs.warmup_steps) / (configs.max_steps - configs.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return configs.min_lr + coeff * (configs.max_lr - configs.min_lr)

def set_seed(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer




if __name__ == "__main__":
    os.chdir("/data_all/intern05/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg")

    world_size = 2


    print("=> loading configuration from config.yaml") 
    configs = OmegaConf.load("/data_all/intern05/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/config.yaml")
    # 设置随机种子, 保证结果可重现
    set_seed(configs)
    
    # 获取 RANK 和 WORLD_SIZE 环境变量，默认为 0 和 1, 由命令行传入
    # ddp_rank 是全局唯一的，用于区分所有进程。
    # ddp_local_rank 是单台机器局部唯一的，用于区分同一节点上的进程。
    ddp_rank = int(os.environ.get('RANK', 0))
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 初始化分布式进程组
    ddp_setup(ddp_rank, world_size)

    # 主进程
    print(f"ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, world_size: {world_size}")
    master_process = ddp_rank == 0 
    device = f'cuda:{ddp_local_rank}'
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # 设置当前进程使用的 CUDA 设备
    torch.cuda.set_device(device)

    if master_process:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())

    if configs.debug:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 16,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }
    # 创建模型
    if master_process:
        print("=> creating GPT2 model")
        print(f"=> GPT2 config: {GPT_CONFIG_124M}")
        with open("log/log.txt", "a") as f:
            f.write(f"GPT2 config: {GPT_CONFIG_124M}\n")

    assert configs.total_size % (configs.batch_size * GPT_CONFIG_124M["context_length"] * world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = configs.total_size // (configs.batch_size * GPT_CONFIG_124M["context_length"] * world_size)
    if master_process:
        print(f"total desired data size: {configs.total_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        with open("log/log.txt", "a") as f:
            f.write(f"total desired data size: {configs.total_size}\n")
            f.write(f"=> calculated gradient accumulation steps: {grad_accum_steps}\n")


    train_loader = DataLoaderLite(batch_size=configs.batch_size, context_length=GPT_CONFIG_124M["context_length"], rank=ddp_rank, world_size=world_size, split="train")
    val_loader = DataLoaderLite(batch_size=configs.batch_size, context_length=GPT_CONFIG_124M["context_length"], rank=ddp_rank, world_size=world_size, split="val")
   

    # 记录日志
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")

    if master_process:
        with open(log_file, "w") as f: # open for writing to clear the file
            pass
        with open("log/log.txt", "w") as f:
            f.write(f"GPT2 config: {GPT_CONFIG_124M}\n")

    if configs.precision == "tf32":
        torch.set_float32_matmul_precision('high')
        if master_process:
            print("=> using tf32 precision")
    elif configs.precision == "bf16":
        if master_process:
            print("=> using bf16 precision")
    else:
        if master_process:
            print("=> using fp32 precision")


    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    raw_model = model
    # 如果在compile后再保存原模型，模型的参数keys会增加一个_orig_mod前缀，因为compile，DDP，原模型参数共享，所以最好一开始就保存
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs.max_lr, weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: get_lr(it, configs)/ configs.max_lr)

    if master_process:
        print("=> starting training")
        with open(log_file, "a") as f:
            f.write("starting training\n")
    loss_list = []
    # 训练模型
    for step in range(configs.max_steps):
        last_step = step == configs.max_steps - 1
        t0 = time.time()
        # train on training set
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            # only require sync on the last micro-step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            input, target = train_loader.next_batch()
            input, target = input.to(device), target.to(device)
            if configs.precision == "bf16":
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # using bf16
                    logits = model(input)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            else:
                logits = model(input)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            loss = loss / grad_accum_steps
            # loss accum records the digital loss value， don't need to backpropagate, so detach
            loss_accum += loss.detach()

            loss.backward() # 将loss反向传播累积，直到model.require_backward_grad_sync为True时，才进行梯度更新
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG) # 通信：average loss across all processes
        lr = get_lr(step, configs)
        optimizer.step()
        lr_scheduler.step()
        torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = configs.batch_size * GPT_CONFIG_124M["context_length"] * world_size * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        loss_list.append(loss_accum.item())
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}\n")

        # eval on validation set
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    input, target = train_loader.next_batch()
                    input, target = input.to(device), target.to(device)
                    if configs.precision == "bf16":
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # using 
                            logits = model(input)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
                    else:
                        logits = model(input)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG)


            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                
                # save model checkpoint
                if step >= 0 and (step % 5000 == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': GPT_CONFIG_124M,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'optimizer': optimizer.state_dict()
                    }
                    # you might also want to add optimizer.state_dict() and
                    # rng seeds etc., if you wanted to more exactly resume training
                    torch.save(checkpoint, checkpoint_path)
                    print(f"saved checkpoint to {checkpoint_path}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} saved {checkpoint_path}\n")
                    tokenizer = tiktoken.get_encoding("gpt2")
                    text = generate_and_print_sample(
                            model, tokenizer, device, context_size=GPT_CONFIG_124M["context_length"], start_context="Life is"
                        )
                    with open(log_file, "a") as f:
                        f.write(f"{step} sample {text}\n")
                    
    destroy_process_group()  # NEW: cleanly exit distributed mode
    # 绘制loss曲线保存为pdf
    plt.plot(loss_list)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.savefig("loss.pdf")
    