import torch.nn as nn
import torch
import torch.functional as F

#positional embedding
class Positional Embedding(nn.module):
    
#点乘得到attention*value的模块
class ScaledDotProductAttention(nn.module):
    #初始化，定义drop_out和sqrt(d_k)，不过
	def __init__(self,temperature,attn_dropout=0.1):
        super().__init__()#继承自nn.module
        self.temperature=temperature  #根据公式就是sqrt(d_k)
        self.dropout=nn.Dropout(attn_dropout) #设定dropout率，返回一个类
    #前向
    def forward(self,q,k,v,mask=None):
        #根据q,k算attention,torch.matmul计算矩阵点乘
        attn=torch.matmul(q/self.temperature,k.transpose(2,3)) #q*kt/sqrt(dk)
        if mask is not None:
            attn=attn.masked_fill(mask==0,-1e9) #mask里为0的赋值10^-9
        attn=self.dropout(F.softmax(attn,dim=-1)) #softmax再dropout
        output=torch.matmul(attn,v)
        return output attn

#MultiHeadAttention
class MultiHeadAttention(nn.module):
    def __init__(self,n_head,d_model,d_k,d_v,dropout=0.1): #d_k=d_q
        super.__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.d_v=d_v
    
        self.w_qs=nn.Linear(d_model,n_head*d_k,bias=False)  #embedding后的输入，转化成n_head个querries，原来是batch_size*len_sequence*d_model，变成batch_size*n_head*d_k
        self.w_ks=nn.Linear(d_model,n_head*d_k,bias=False)
        self.w_vs=nn.Linear(d_model,n_head*d_v,bias=False)
        self.attention=ScaledDotProductAttention(temperature=d_k**0.5)
        self.fc=nn.Linear(n_head*d_v,d_model,bias=False) #把n_head*d_v维度,变成d_model维，然后可以输入下一个模块
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(d_model)   #Add&Norm里的LayerNorm层
        
    def forward(self,q,k,v,mask=None):  #前向需要确定q,k,v的值以及要不要mask，encoder里不用，decoder里要用
        d_k,d_v,n_head=self.d_k,self.d_v,self.n_head
        bs,len_q,len_k,len_v=q.size(0),q.size(1),k.size(1)
        residual=q #残差结构，后面add&norm用
        q=self.w_qs(q).view(bs,len_q,n_head,d_k)    #可以并行计算的优越性，可以直接计算一个batch，整个sequence里的所有q,k,v，然后再转换成对应维度
        k=self.w_ks(k).view(bs,len_k,n_head,d_k)
        v=self.w_vs(v).view(bs,len_v,n_head,d_v)
        #转换成计算attention的维度
        q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2) #转成bs,n_head,len_q,d_k，便于可以计算一个sequence每个querry对每个key的attention
        #算mask
        if mask is not None:
            mask=mask.unsqueeze(1)  ####操作看不懂？？？？ head axis broadcasting
        
        #算masked或not masked attention
        q,attn=self.attention(q,k,v,mask=mask)
        #转成bs*len_q*n_head*d_k，然后用view把不同head里的q,k concat起来
        q=q.transpose(1,2).contiguous().view(bs,len_q,-1) #.contiguous()方法的作用https://zhuanlan.zhihu.com/p/64551412,可以用.reshape替代contiguous().view
        #转换维度
        q=self.dropout(self.fc(q))
        #Add and Norm
        q=q.add(residual)
        q=self.layer_norm(q)
    return q ,attn
        
#FeedForward
class PositionwiseFeedForward(nn.module):
    def __init__(self,d_in,d_hid,dropout=0.1):
        super().__init__()
        #两层的ffn
        self.w_1=nn.Linear(d_in,d_hid) #pointwise，先从in到hidden layer
        self.w_2=nn.Linear(d_hid,d_in) #输出的维度和输入的维度一样，都是d_model，不然咋add
        slef.layer_norm=nn.LayerNorm(d_in,eps=1e-6) #1e-6代表归一化时分母加1e-6，防止分母为0
        self.dropout=nn.Dropout(dropout)
    #前向
    def forward(self,x):
        residual=x
        x=self.w_2(F.relu(self.w_1(x)))
        x=self.dropout(x)
        x=x.add(residual)
        x=self.layer_norm(x)
    return x
        
        
#利用上面的俩实现一个Encoder Layer
class EncoderLayer(nn.module):
    def __init__(self,d_model,d_inner,n_head,d_k,d_v,dropout=0.1):     #d_inner就是中间层，就是d_hid
        super(EncoderLayer,self).__init__()
        #一个mulitihead attention模块加一个ffn
        self.slf_attn=MultiHeadAttention(n_head,d_model,d_k,d_v,dropout=dropout)
        self.pos_ffn=PositionwiseFeedForward(d_model,d_inner,dropout=dropout)
    def forward(self,x,slf_attn_mask=None): #输入和self attention的mask
        enc_output,enc_slf_attn=self.slf_attn(x,x,x,mask=attn_mask)
        enc_output=self.pos_ffn(enc_output)
    return enc_output enc_slf_attn  #返回输出和自注意力
        
        
        