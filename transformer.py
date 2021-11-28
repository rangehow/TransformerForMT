import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math


class Transformer(nn.Module):

    def __init__(self, layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout=True):
        '''

        :param layers:
        :param embed_dim:
        :param ff_embed_dim:
        :param num_heads:
        :param dropout:
        :param weights_dropout:
        '''
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, with_external, weights_dropout))

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_memories=None, external_padding_mask=None):
        '''

        Args:
            x: 源语言或者目标语言的 token tensor变成的词向量(做embedding和位置编码)
            kv:
            self_padding_mask: 一个padding位置是false的矩阵，维度和x一致
            self_attn_mask:
            external_memories:
            external_padding_mask:

        Returns:

        '''
        for idx, layer in enumerate(self.layers):
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask, external_memories, external_padding_mask)
        return x


class TransformerLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout=True):
        '''

        Args:
            embed_dim:
            ff_embed_dim:
            num_heads:
            dropout:
            with_external: 跨层注意力
            weights_dropout:
        '''
        super(TransformerLayer, self).__init__()
        # 自注意层 多头自注意
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        # 自注意层后的层正则化
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        # 前馈神经网络层
        self.ff_layer = FeedForwardLayer(embed_dim, ff_embed_dim, dropout)
        # 前馈神经网络层后的层正则化
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.with_external = with_external
        self.dropout = dropout
        # 如果有开这个设定，额外在后面加一层多头注意、残差、层正则化
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
            self.external_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                external_memories=None, external_padding_mask=None):
        '''

        Args:
            x: q
            kv: kv,没给说明是cross-attention
            self_padding_mask: encoder的 padding矩阵 seq x bsz
            self_attn_mask:
            external_padding_mask:

        Returns:

        '''
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            # self-attention
            x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask,
                                          attn_mask=self_attn_mask)
        else:
            # cross-attention
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, key_padding_mask=self_padding_mask,
                                          attn_mask=self_attn_mask)
        # x:tgt_len, bsz, embed_dim
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 残差连接
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            residual = x
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories,
                                                  key_padding_mask=external_padding_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None

        residual = x
        x = self.ff_layer(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn, external_attn


class FeedForwardLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        '''

        Args:
            embed_dim: 词向量长度
            num_heads: 头数
            dropout: 丢弃概率
            weights_dropout: true 决定dropout的参数是softmax后的权重，false是最后出来的结果
        '''
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # 点乘注意力 x 1/根号dk ，现在每个头的维度变成headnum，所以改改
        self.scaling = self.head_dim ** -0.5
        # 用于将x映射成QKV的参数
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, attn_bias=None):
        """
        :param query: seq_len x bsz x embed_dim
        :param key: seq_len x bsz x embed_dim
        :param value: seq_len x bsz x embed_dim
        :param key_padding_mask: seq x bsz
        :param attn_mask: tgt_len x src_len
        :param attn_bias:
        :return:  attn: tgt_len, bsz, embed_dim
                  attn-mask:bsz,tgt_len,src_len

        """

        # encoder和decoder的多头自注意力
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        # decoder的跨语言注意力
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            # 在正经transformer里不会出现的情况，如果出现也被assert掉了
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        # q*=self.scaling 可能会导致反向传播时的错误
        q = q * self.scaling
        # 点乘注意力的时候bmm要在第1维和第二维操作，所以得改成batch_first
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # k,v: bsz*heads x src_len x dim
        # q: bsz*heads x tgt_len x dim

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_bias is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_bias = attn_bias.transpose(0, 1).unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights + attn_bias
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # 用负无穷把mask矩阵是1的地方填充了，第0维靠广播
        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),
                float('-inf')
            )

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # 对 k维度 也就是src_len做归一
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn, attn_weights

    # in_proj_* 是用来做切割的
    def in_proj_qkv(self, query):
        '''

        :param query: seq_len x bsz x embed_dim
        :return:
        '''
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    # 用来做映射
    def _in_proj(self, input, start=0, end=None):
        '''

        :param input: seq_len x bsz x embed_dim
        :param start: 用来截取 Weights 和 bias
        :param end: 用来截取 Weights 和 bias
        :return: input x weight' + bias  seq_len x bsz x 3*embed_dim
        '''
        weight = self.in_proj_weight  # 3 * embed_dim, embed_dim
        bias = self.in_proj_bias  # 3*embed_dim
        # 截取weights 和 bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    '''

    Args:
        num_embeddings: 词表大小
        embedding_dim:每个词向量的长度
        padding_idx:padding的词表索引

    Returns:返回一个词嵌入层，实现词与词向量的映射

    '''
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # 用正态分布随机填充m的权重张量
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    # 把padding位置的权重设置为0
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class SelfAttentionMask(nn.Module):
    def __init__(self, init_size=100):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)

    @staticmethod
    def get_mask(size):
        # 主对角线以上的元素(不包含主对角线) 转成bool类型，也就是右上角全true的方阵
        weights = torch.ones((size, size), dtype=torch.uint8).triu_(1).bool()
        return weights

    def forward(self, input):
        """Input is expected to be of size [seq_len x bsz x (...)]."""
        size = input.size(0)
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        res = self.weights[:size, :size].to(input.device).detach()
        return res


class LearnedPositionalEmbedding(nn.Module):
    """This module produces LearnedPositionalEmbedding.
    """

    def __init__(self, embedding_dim, init_size=512):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(init_size, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weights.weight, 0.)

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = input.size()
        positions = (offset + torch.arange(seq_len)).to(input.device)
        res = self.weights(positions).unsqueeze(1)
        return res


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    """

    def __init__(self, embedding_dim, init_size=512):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        '''

        Args:
            num_embeddings: 广泛意义的句子长度，这里我们多求一点，正弦位置编码不依赖数据，所以
                            求个512就够用了，这么长的话不多见
            embedding_dim: 词向量数

        Returns:正余弦位置编码二维张量

        '''

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # 其实和论文是一样的，上下同除以2，这样分子可以直接×range(i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)

        # 因为两个乘数维度不一定相同，所以在头尾补一个一维这样就可以乘了，得到一个 num_emb * half_dim
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # 在列里拼接，然后展成一维 [[ SIN,COS],[SIN,COS]...]这样，虽然和论文交错的不太一样
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果词向量是奇数，那么最后总会有一个位置还是空着的，没有处理，就用0填充意思意思
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz x (...)]."""
        seq_len, *dims = input.size()
        mx_position = seq_len + offset
        if self.weights is None or mx_position > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                mx_position,
                self.embedding_dim,
            )

        positions = offset + torch.arange(seq_len)
        shape = [1] * len(dims)
        shape = [seq_len] + shape + [-1]
        res = self.weights.index_select(0, positions).view(shape).to(input.device).detach()
        return res
