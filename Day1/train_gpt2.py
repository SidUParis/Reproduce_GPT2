# First task, let's load GPT2 124M into the class that we need to develop from scratch 
# GPT2 is a slightly modified Transformers version, cf the paper, and it is a Decoder Only architecture. 
from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math
import os
import tiktoken
os.environ['HF_HOME'] = r"/media/sunxu/HIKSEMI/Huggingface"
#--------------------------------------------------------

# 5 CasualAttention , it is not just an attention, it is a multi-head attention 
class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        """ transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
            transformer.h.0.attn.c_attn.bias torch.Size([2304])
            transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
            transformer.h.0.attn.c_proj.bias torch.Size([768])
        """
        super().__init__()
        assert config.n_embed % config.n_head == 0 
        

        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)
        self.n_head = config.n_head 
        self.n_embed = config.n_embed
        # 创建一个bias  
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
        # 不是param，我们要登记为buffer ,现在名字是Bias
        
    def forward(self,x):
        
        # Decoupling the q,k,v
        B,T,C = x.size()  

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embed,dim=2)
        # k,q, v originally from line 40, size are of B T C
        # then we say it is a n_head attention, it means,C will be subdivided into [num.head, C/num_head]
        ## 优点： 不同的头可以关注不同的方面，可以并行计算注意力（问题，头不同数量了还能一样吗？），在不显著增加参数的情况下增加模型表达能力
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # （B，T，num_head,head size) 专制变成 -> (B,num_head,T,head_size)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B，T，num_head,channel(head size)) 专制变成 -> (B,num_head,T,head_size)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)# B，T，num_head,channel(head size)) 专制变成 -> (B,num_head,T,head_size)
        print(q.shape)
        att = (q@k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1))) # attention 计算公式中的根号 
        # (B,num_head,T,head_size) @ (B,num_head,head_size,T) -> (B,num_head,T,T)
        att = att.masked_fill(self.bias[:,:,T,T]==0,float('-inf')) # decoder 特点 保证了未来的token永远不会和前面联系
        att = F.softmax(att,dim=-1)
        y = att @ v #     B，num_head,T,T @ B,num_head,T,head_size -> B, num_head,T,head_size
        y = y.transpose(1,2).contiguous().view(B,T,C) # B,T,num_head,head_size -> 通过view 变回 B T C
        # output projection
        y = self.c_proj(y)
        return y 

#4 MLP 
class MLP(nn.Module):
    """ 
    transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
    transformer.h.0.mlp.c_fc.bias torch.Size([3072])
    transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
    transformer.h.0.mlp.c_proj.bias torch.Size([768])
    
    """
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed,4*config.n_embed) # 为什么4？ 
        self.gelu = nn.GELU(approximate = 'tanh') # 为什么GELU？ GELU不和Relu一样，relu会强行抹平，但是gelu不会，不是完全直线，cf torch.nn网站GELU
        self.c_proj = nn.Linear(4*config.n_embed,config.n_embed)
    def forward(self,x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        # 通过查看模型介绍可以发现 每一个block实际上会生成一个ln_1,attn,ln_2,mlp 就是一个h_0....
        # 看transformer那个图
        self.ln_1 = nn.LayerNorm(config.n_embed) # 768
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed) # 768 
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x)) # where information get exchanged ,shared
        x = x + self.mlp(self.ln_2(x)) # where the data thinks 
        return x


#1
@dataclass
class GPTconfig:
    #we need to correspond to the model architecture which are below:
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
    block_size: int = 1024 # max seq length  
    vocab_size: int = 50257 # num of token 
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768  
#2 begin
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embed), # 对于第一个是word token embedding, [token size, embed size]
            wpe = nn.Embedding(config.block_size,config.n_embed),
            #H 这里能直接用nn.Embedding吗？ 不能，首先什么是nn》Embedding？是一个wrapper of array 
            # 而H后面不是weight或者string 而是0， 1, 。。。。数字，也就是说这里面其实有很多层，多少层呢？ config.n_layer
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 代表hidden
            # Block也就是说让他生成一个网络，我们在外部添加就好了
            ln_f = nn.LayerNorm(config.n_embed), # 作normalization

        ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)
        

    def forward(self,idx):
        B,T = idx.size()
        assert T <= self.config.block_size ,f"cannot forward, because it surpluses the longest sequences" # block size is the limit of the sequence length
        # get pos, tok embedding
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device) # shape [T]
        pos_embed = self.transformer.wpe(pos) # T, n_embed
        tok_embed = self.transformer.wte(idx) # B,T,n_embed
        x = tok_embed + pos_embed # B,T,n_embed
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    @classmethod 
    def from_pretrained(cls,model_type):
        """ load pretrained GPT2 weigth from hf"""
        assert model_type in {'gpt2','gpt2-medium'}
        from transformers import GPT2LMHeadModel 
        print("loading weights from pretrained gpt: %s" %model_type) # %代表什么？
        # n_layers, n_embed, n_head from the model type
        config_args = { 
                       'gpt2': dict(n_layer=12,n_head=12,n_embed=768),
                       'gpt2-medium': dict(n_layer=24,n_head=16,n_embed=1024),
                       }[model_type] # ？
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 
        
        # create a from-scratch minGPT
        config = GPTconfig(**config_args)
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
    
#-----------------------
model = GPT.from_pretrained('gpt2')
print("good")

# first try with weight initlaized by huggingface

num_return_sequences = 5

max_length = 30
model.eval()
device = 'cuda'
model.to(device)
# prefix token


enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = tokens.to(device)


# print(x.size())
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits,dim=-1)
        topk_probs,topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs,1)
        xcol = torch.gather(topk_indices,-1,ix)# xcol 是新生成的token
        x = torch.cat((x,xcol),dim=-1) # 把xcol 和 x 拼接在一起 表示生成的old_token + new_token
        # print(x)
for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">,",decoded)