from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import inspect
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size:int = 50257
    n_layer: int = 12
    n_head: int = 12  #the num of self_attention heads
    n_embd: int = 768 
class CausalSelfAttention(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # projection q,k,v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        '''
        1. 创建一个形状为 (config.block_size, config.block_size) 的全为1的张量。
        2. 获取这个张量的下三角部分,其他位置元素设为0。
        3. 将这个下三角矩阵的形状调整为 (1, 1, config.block_size, config.block_size)。
        4.将这个下三角矩阵注册为模块的一个缓冲区，并命名为 "bias"。
        这种下三角矩阵在实现注意力机制时很常用,尤其是自回归模型中,用于确保模型只关注当前时间步及之前的时间步,而不会看到未来的时间步。
        '''
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size)) 
    def forward(self,x):
        B,T,C = x.size() #Batch size, sequence length, embedding dimensionality
        qkv = self.c_attn(x) #(batch_size, seq_len, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd,dim=2) #在第2个维度,以n_embd大小来切割
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B,n_head,T,hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        '''
        矩阵乘法对于高维度而言是按照最后两个维度来做点积
        k的最后一个维度就是q,k,v的维度
        '''
        # att = q @ k.transpose(-1,-2) * 1.0/math.sqrt(k.size(-1)) #(B,n_head,seq_len,seq_len)
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf')) # att上三角位置的分数用负无穷填充
        # att = F.softmax(att,dim=-1) #(B,n_head,seq_len,seq_len)
        # y = att @ v #(B,n_head,T,T) @ (B,n_head,T,hs) -> (B,n_head,T,hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C) #(B,T,C)
        return y
    
class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding block_size means the lenth of the sequence
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        # weight sharing cheme
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

      
    def _init_weights(self,module):
        std = 0.02
        if isinstance(module,nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self,idx,target=None):
        B, T = idx.size()
        assert T <= self.config.block_size,f"Cannot forward the sequence of length{T} which is higher than block_size{self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) #(T, n_embd)
        tok_emb = self.transformer.wte(idx) #(B, T, n_embd)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        return logits,loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # which is bias 
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # if master_process:
        #     print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer