# Day 6
# Changes : Flash Attention
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
# Changes : fix the num of the vocab_size : better make sure we have all the num of power of 2
# Changes : GO to GPT3 paper, we adopted the same hyperparams
# https://blog.csdn.net/zhaohongfei_358/article/details/122820992
# Changes : LR Scheduler
# Changes: Align with GPT3 paper: Batch Size 0.5M (tokens)



# First task, let's load GPT2 124M into the class that we need to develop from scratch
# GPT2 is a slightly modified Transformers version, cf the paper, and it is a Decoder Only architecture.
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import tiktoken
import tiktoken
import torch._dynamo
import time
import inspect
torch._dynamo.config.suppress_errors = True


os.environ['HF_HOME'] = r"/media/sunxu/HIKSEMI/Huggingface"


# --------------------------------------------------------

# 5 CasualAttention , it is not just an attention, it is a multi-head attention
class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        """ transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
            transformer.h.0.attn.c_attn.bias torch.Size([2304])
            transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
            transformer.h.0.attn.c_proj.bias torch.Size([768])
        """
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # What are the main point of CasualSelfAttention? 多头注意力机制注意点在哪里
        ## D——model维向量先经过一个Linear Layer,在分解为h个head 计算attention, 最终将attention 连接在一起后再经过一层linear layer输出，在整个过程中需要四个输入和输出维度都是—d_model的Linear layer
        ## 整个模型(batch_size, seq_len,d_model) 输出也是一样的
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # 创建一个bias
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size))
        # 不是param，我们要登记为buffer ,现在名字是Bias

    def forward(self, x):
        # 解维度
        B, T, C = x.size()
        # batch_size, seq_length,embed_dimension (n_embed)
        # C = num of head * head size
        # eg in GPT2 124 M n_head = 12 , hs =64 o it is 768 channels in the transformers
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        # k,q, v originally from line 40, size are of B T C
        # then we say it is a n_head attention, it means,C will be subdivided into [num.head, C/num_head]
        ## 优点： 不同的头可以关注不同的方面，可以并行计算注意力（问题，头不同数量了还能一样吗？），在不显著增加参数的情况下增加模型表达能力
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                  2)  # （B，T，num_head,head size) 专制变成 -> (B,num_head,T,head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                  2)  # (B，T，num_head,channel(head size)) 专制变成 -> (B,num_head,T,head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                  2)  # B，T，num_head,channel(head size)) 专制变成 -> (B,num_head,T,head_size)
        # print(q.shape)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # attention 计算公式中的根号
        # # (B,num_head,T,head_size) @ (B,num_head,head_size,T) -> (B,num_head,T,T)
        # att = att.masked_fill(self.bias[:, :, T, T] == 0, float('-inf'))  # decoder 特点 保证了未来的token永远不会和前面联系
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # B，num_head,T,T @ B,num_head,T,head_size -> B, num_head,T,head_size

        # Day 6 Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # B,T,num_head,head_size -> 通过view 变回 B T C
        # output projection
        y = self.c_proj(y)
        return y

    # 4 MLP


class MLP(nn.Module):
    """
    transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
    transformer.h.0.mlp.c_fc.bias torch.Size([3072])
    transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
    transformer.h.0.mlp.c_proj.bias torch.Size([768])

    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)  # 为什么4？
        self.gelu = nn.GELU(approximate='tanh')  # 为什么GELU？ GELU不和Relu一样，relu会强行抹平，但是gelu不会，不是完全直线，cf torch.nn网站GELU
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


# 4 END ，在4结束后，我们发现还有一个CasualSelfAttention没有搭建-> 5
# 3 begin 我们需要考虑block，因为我们刚才在搭建GPT2骨架的时候将其直接使用了
# 我们知道的是，block里也是为了生成一层一层的网络，只不过是hidden,所以这个class 也一定继承nn.Module

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 通过查看模型介绍可以发现 每一个block实际上会生成一个ln_1,attn,ln_2,mlp 就是一个h_0....
        # 看transformer那个图
        self.ln_1 = nn.LayerNorm(config.n_embed)  # 768
        self.attn = CasualSelfAttention(config)  #
        self.ln_2 = nn.LayerNorm(config.n_embed)  # 768
        self.mlp = MLP(config)  # config??

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # where information get exchanged ,shared

        x = x + self.mlp(self.ln_2(x))  # where the data thinks
        return x
    # 3 end ,我们MLP 还没定义，第四部我们来定义MLP


# 1
@dataclass
class GPTconfig:
    # we need to correspond to the model architecture which are below:
    # transformer.wte.weight torch.Size([50257, 768])
    # transformer.wpe.weight torch.Size([1024, 768])
    # transformer.h.0.ln_1.weight torch.Size([768])
    # transformer.h.0.ln_1.bias torch.Size([768])
    # transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
    # transformer.h.0.attn.c_attn.bias torch.Size([2304])
    # transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
    # transformer.h.0.attn.c_proj.bias torch.Size([768])
    # transformer.h.0.ln_2.weight torch.Size([768]) -.........
    #     ...
    # transformer.h.11.mlp.c_proj.bias torch.Size([768])
    # transformer.ln_f.weight torch.Size([768])
    # transformer.ln_f.bias torch.Size([768])
    # lm_head.weight torch.Size([50257, 768])
    block_size: int = 1024  # max seq length
    vocab_size: int = 50257  # num of token
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


# 2 begin
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 因为前面我们可以看到main container 是 transformer.wte....所以我们应该有一个attribute为transformer，而且我们这个下面有wte, wpe。。。 所以我们应该考虑把他做成一个字典 用ModuleDict

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embed),  # 对于第一个是word token embedding, [token size, embed size]
            wpe=nn.Embedding(config.block_size, config.n_embed),
            # H 这里能直接用nn.Embedding吗？ 不能，首先什么是nn》Embedding？是一个wrapper of array
            # 而H后面不是weight或者string 而是0， 1, 。。。。数字，也就是说这里面其实有很多层，多少层呢？ config.n_layer
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 代表hidden
            # Block也就是说让他生成一个网络，我们在外部添加就好了
            ln_f=nn.LayerNorm(config.n_embed),  # 作normalization

        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Day 4 New
        self.transformer.wte.weight = self.lm_head.weight  # weight sharing
        self.apply(self._init_weights)

        # 1.14 在GPT2 模型中，实际上他们在一开始的权重initialize 的过程中采用了stdev = 0.02 for all weights except for wpe , which is 0.01, bias as 0
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02  # default 1.20
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # we want the one of the sqrt of num of layers
                # the number of residual layers is twice o
                std *= (2 * self.config.n_layer) ** -0.5
                # 这个2怎么来的？解释在1小时21 ， 每一个transformer中的layer 都有两个residual block that adds to the residual pathway
                # Attn + Mlp = 2
            torch.nn.init.normal_(module.weight, std=std, mean=0.0)
            # 这个std 怎么得到的？ 实际上是1/sqrt(n) n是输入的维度, 768, 1024, 2304, 3072....0.002 是最接近的 empirical value
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02, mean=0.0)
    def forward(self, idx, targets=None):  # 为什么这里是idx？ # 53分钟视频 引入新的hype param
        # input 是什么？ 是token sequences,但是我们给GPT的不是token, 而是Token idx, 每一个批次中有(B,T) T代表token idx 数量，我们应该确保输入的要比T小
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length{T}, it is longer than block size {self.config.block_size}"
        # create a position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # this should be shape [T]
        pos_embed = self.transformer.wpe(pos)  # T, n_embed ？
        tok_embed = self.transformer.wte(idx)  # B,T,n_embed ？
        x = tok_embed + pos_embed
        # forward in transformers
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        loss = None
        logits = self.lm_head(x)  # B,T,vocab_size
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # it will first convert logits into 2dim, (B*T, vocab_size) and then targets into 1dim (B*T)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """ load pretrained GPT2 weigth from hf"""
        assert model_type in {'gpt2', 'gpt2-medium'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)  # %代表什么？
        # n_layers, n_embed, n_head from the model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),

        }[model_type]  # 这是什么意思？
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create a from-scratch minGPT
        config = GPTconfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
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
    def configure_optimizers(self,weight_decay,learning_rate,device):
        # 2小时29分钟
        # starts with all registered parameters
        # 我们人为的将所有的参数分成 1. 会weight decay 2. 不会weight decay
        # 1 dim tensors 不会weight decay, 例如layer norm, bias
        # 而参与matrix multiplication 的参数会weight decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups, any parameters that is 2D will be weight decay? ??????
        # otherwise no weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {"params": nodecay_params, 'weight_decay':0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params) # numel() 返回的是一个tensor中元素的个数
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, num_decay_params: {num_decay_params}")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, num_nodecay_params: {num_nodecay_params}")
        # create a adamW optimizer
        # 检查是否有fused adam，早期pytorch并没有fused adam
        # Kernel fusion  for the optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # ?
        used_fused = fused_available and 'cuda' in device
        print(f"using fused adam: {used_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8,fused=used_fused)
        return optimizer

class DataLoaderLite:
    def __init__(self,B,T):
        self.B = B
        self.T = T
        with open('./input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded text with {len(self.tokens)} tokens")
        print(f"1 epoch = {self.tokens.size(0) // (B*T)} batches")

        # current state
        self.current_position = 0
    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position:self.current_position + B*T+1] # we need one additional token for the y values
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        # 不要在这里就挪到GPU 浪费资源
        self.current_position += B*T
        # important to know that we are advancing by B*T tokens
        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_position = 0
        return x,y

# -----------------------
max_lr = 3e-4 # 在GPT3的文章中，我们可以在Table2.1中找到这个超参数
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it): # 不是很懂 # GPT
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr *(it+1) / warmup_steps
    # 2) if it > lr_decay_iter , return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between use cosine decay down to min lr
    decay_ratio = (it-warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

num_return_sequences = 5
max_length = 30
device = 'cuda'

train_loader = DataLoaderLite(4,32)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
total_batch_size,B,T = 524288,16,1024 # 2**19 ~0.5M tokens
assert total_batch_size % (B*T) == 0, "make sure the total batch size is divisible"
grad_accum_steps = total_batch_size // (B*T) # 一次是16*1024= 16384 个token 一共我们要处理2**19 = 524288 个token 所以一共是32 steps
# 这也就是说在下面的forward backward 我们要做32次才能step(),查看下面的代码
print(f"gradient accumulation steps: {grad_accum_steps} with total desired batch size {total_batch_size}")
max_steps = 50
# Model Initialization
model = GPT(GPTconfig(vocab_size=50304))
model.to(device)
torch.compile(model)
# torch.set_float32_matmul_precision('highest') -> torch.float32 when do the inner calculation
# torch.set_float32_matmul_precision('high') -> torch.TF32 when do the inner calculation
# torch.set_float32_matmul_precision('medium') -> bflat16 when do the inner calculation

# optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate = 6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
    # one epoch = 2640 batches, 也就是steps = 2640, 但是我们只50
    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    optimizer.zero_grad()
    # 1.43 我们用autocast只需要包围我们的计算，不要包围我们的backward 和step
    loss_accum = 0.0
    ###2小时39 我们需要做32次steps才更新一次 ,但下面的代码有些问题，查看notebook 31
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # 不要用fploat16,会需要gradient scaler
            # loss 是cross-entropy 也是reduction by mean ,但是我们在做gradient accumulation 的时候，我们需要除以grad_accum_steps
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        # 2小时45
        loss_accum += loss.detach()  # 做这一步只是为了print

    # import code; code.interact(local=locals())
        loss.backward()
    norms = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # prevent the model being shocked by the gradient magnitude
    lr = get_lr(step)
    for param_group in optimizer.param_groups:  # we need to iterate over the param_groups to set the lr
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # this will wait for everything to finish
    t1 = time.time()
    dt = (t1 - t0) * 1000 # milliseconds
    # How many tokens we are processing per second?
    tokens_per_sec = (train_loader.B * train_loader.T*grad_accum_steps) / (t1 - t0)
    print(f"step {step}, loss :{loss_accum.item()} | Norms {norms:.4f}|lr: {lr:.4e}| dt:{dt:.2f}ms | tok/sec {tokens_per_sec:.2f},") # convert tensor to float
import sys; sys.exit(0)

