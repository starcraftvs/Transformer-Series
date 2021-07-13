import torch.nn as nn
import torch
import torch.functional as F
import numpy as np

#尽量在forward里面直接调用层，不需要输入其他参数
#positional embedding 还没看？？？？
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)#(1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
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
        return output, attn

#MultiHeadAttention
class MultiHeadAttention(nn.module):
    def __init__(self,n_head,d_model,d_k,d_v,dropout=0.1): #d_k=d_q
        super().__init__()
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
        q=q.transpose(1,2).contiguous().view(bs,len_q,-1) #.contiguous()方法的作用https://zhuanlan.zhihu.com/p/64551412, 可以用.reshape替代contiguous().view
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
        enc_output,enc_slf_attn=self.slf_attn(x,x,x,mask=slf_attn_mask)
        enc_output=self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn  #返回输出和自注意力
        
        
#利用上面俩实现一个Decoder Layer和Encoder就多一个MultiheadAttention Layer，然后输入不一样
class DecoderLayer(nn.module):
    def __init__(self,d_model,d_inner,n_head,d_k,d_v,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.slf_attn=MultiHeadAttention(n_head,d_model,d_k,d_v,dropout=dropout)
        self.enc_attn=MultiHeadAttention(n_head,d_model,d_k,d_v,dropout=dropout)
        self.pos_ffn=PositionwiseFeedForward(d_model,d_inner,dropout=dropout)
    def forward(self,dec_input,enc_input,slf_attn_mask=None,dec_enc_attn_mask=None):
        dec_output,dec_slf_attn=self.slf_attn(dec_input,dec_input,dec_input,mask=slf_attn_mask)
        dec_output,dec_enc_attn=self.enc_attn(dec_output,enc_input,enc_input,mask=dec_enc_attn_mask)
        dec_output=self.pos_ffn(dec_output)
        return dec_output,dec_slf_attn,dec_enc_attn
        
        
#实现整个encoder模块，包括wrod embedding,positional embedding,几个encoder layer
class Encoder(nn.module):
    def __init__(self,n_src_vocab,d_word_vec,n_layers,n_head,d_k,d_v,d_model,d_inner,pad_idx,dropout=0.1,n_position=200,scale_emb=False): #最后一个是决定要不要在输入的embedding就做一个除以sqrt(d_k)的scale操作
        super().__init__()
        self.src_word_emb=nn.Embedding(n_src_vocab,d_word_vec,padding_idx=pad_idx)  #词典词的总数量，转换成的每个词的维度，sequence长度不一样时，少的部分补什么数字
        self.position_enc=PositionalEncoding(d_word_vec,n_position=n_position)
        self.dropout=nn.Dropout(0.1)
        self.layer_stack=nn.ModuleList([EncoderLayer(d_model,d_inner,n_head,d_k,d_v,dropout=dropout) for _ in range(n_layers)]) #nn.Modulelist就像一个list，可以像调用List一样，调用其中的层。nn.Sequential()就像一个多个nn.module组成的一个nn.module，可以直接输入输出的
        self.layer_norm=nn.LayerNorm(d_model,eps=1e-6)
        
    def forward(self,src_seq,src_mask,return_attns=False):
        enc_slf_attn_list=[]
        #前向
        #word embedding
        enc_output=self.src_word_emb(src_seq)
        #scale操作
        enc_output=enc_output*self.d_model**0.5
        #position embedding，再layer_norm
        enc_output=self.dropout(self.position_enc(enc_output))
        enc_output=self.layer_norm(enc_output)
        #过encoder layers
        for enc_layer in self.layer_stack:
         enc_output,enc_slf_attn=enc_layer(enc_output,slf_attn_mask=src_mask)
         #判断要不要记录encoder的self attention
         enc_slf_attn_list+=[enc_slf_attn] if return_attns else [] #相当于.append
        #要不要返回self attention的值？？？返回这个值有啥用啊？？？
        if return_attns:
            return enc_output,enc_slf_attn_list
        return enc_output
        
        
class Decoder(nn.module):
    def __init__(self,n_trg_vocab,d_word_vec,n_layers,n_head,d_k,d_v,d_model,d_inner,pad_idx,n_position=200,dropout=0.1,scale_emb=False):  #和上面一样
        super().__init__()
        #word embedding层
        self.trg_word_emb=nn.Embedding(n_trg_vocab,d_word_vec,padding_idx=pad_idx)
        #positional embedding层
        self.position_dec=PositionalEncoding(d_word_vec,n_position=n_position)
        #layer norm
        self.layer_norm=nn.LayerNorm(d_model,eps=1e-6)
        #dropout
        self.dropout=nn.Dropout(p=dropout)
        #过几个decoder layer
        self.layer_stack=nn.ModuleList([DecoderLayer(d_model,d_inner,d_k,d_v,dropout=dropout) for _ in range(n_layers)])
        #要不要对word embedding的结果就先scale一下
        self.scale_emb=scale_emb
        #记一下d_model下面可以scale
        self.d_model=d_model
        
    #前向
    def forward(self,trg_seq,trg_mask,enc_output,src_mask,return_attns=False):
        dec_slf_attn_list=[],dec_enc_attn_list=[]
        #word embedding
        dec_output=self.trg_word_emb(trg_seq)
        #scale
        if self.scale_emb:
            dec_output=dec_output*self.d_model**0.5
        #position embedding + layernorm
        dec_output=self.layer_norm(self.position_dec(dec_output))
        #过几个decdoer layer
        for dec_layer in self.layer_stack:
            dec_output,dec_slf_attn,dec_enc_attn=dec_layer(dec_output,enc_output,enc_output,slf_attn_mask=trg_mask,dec_enc_attn_mask=src_mask)
        #记不记录attention
        dec_slf_attn_list += [dec_slf_attn] if return_attns else []
        dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        
        #是否返回attention
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output
        
 
#实现transformer 
class Transformer(nn.Module):
    #最后两个参数
    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,d_word_vec=512, d_model=512, d_inner=2048,n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):
        super().__init__()
        #记下下面forward要用的一些参数
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        
        #构造encoder和decoder
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)
            
        #输出映射到词典
        self.trg_word_prj=nn.Linear(d_model,n_trg_vocab,bias=False)
        
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
            
            