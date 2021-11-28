import torch
from torch import nn
import torch.nn.functional as F
import math

from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask, Embedding
from data import ListsToTensor, BOS, EOS, _back_to_txt_for_check
from search import Hypothesis, Beam, search_by_batch


class Generator(nn.Module):
    def __init__(self, vocabs,
                 embed_dim, ff_embed_dim, num_heads, dropout,
                 enc_layers, dec_layers, label_smoothing):
        super(Generator, self).__init__()

        self.vocabs = vocabs

        # src端
        self.encoder = Encoder(vocabs['src'], enc_layers, embed_dim, ff_embed_dim, num_heads, dropout)

        # tgt端
        self.tgt_embed = Embedding(vocabs['tgt'].size, embed_dim, vocabs['tgt'].padding_idx)
        self.tgt_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
        self.decoder = Transformer(dec_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)

        self.embed_scale = math.sqrt(embed_dim)
        self.self_attn_mask = SelfAttentionMask()
        self.output = TokenDecoder(vocabs, self.tgt_embed, label_smoothing)
        self.dropout = dropout

    def encode_step(self, inp):
        src_repr, src_mask = self.encoder(inp['src_tokens'])
        return src_repr, src_mask

    def prepare_incremental_input(self, step_seq):
        token = torch.from_numpy(ListsToTensor(step_seq, self.vocabs['tgt']))
        return token

    def decode_step(self, step_token, state_dict, mem_dict, offset, topk):
        src_repr = mem_dict['encoder_state']
        src_padding_mask = mem_dict['encoder_state_mask']
        _, bsz, _ = src_repr.size()

        new_state_dict = {}

        token_repr = self.embed_scale * self.tgt_embed(step_token) + self.tgt_pos_embed(step_token, offset)
        for idx, layer in enumerate(self.decoder.layers):
            name_i = 'decoder_state_at_layer_%d' % idx
            if name_i in state_dict:
                prev_token_repr = state_dict[name_i]
                new_token_repr = torch.cat([prev_token_repr, token_repr], 0)
            else:
                new_token_repr = token_repr

            new_state_dict[name_i] = new_token_repr
            token_repr, _, _ = layer(token_repr, kv=new_token_repr, external_memories=src_repr,
                                     external_padding_mask=src_padding_mask)
        name = 'decoder_state_at_last_layer'
        if name in state_dict:
            prev_token_state = state_dict[name]
            new_token_state = torch.cat([prev_token_state, token_repr], 0)
        else:
            new_token_state = token_repr
        new_state_dict[name] = new_token_state

        LL = self.output(token_repr, None, work=True)

        def idx2token(idx, local_vocab):
            if (local_vocab is not None) and (idx in local_vocab):
                return local_vocab[idx]
            return self.vocabs['tgt'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1)  # bsz x k

        results = []
        for s, t in zip(topk_scores.tolist(), topk_token.tolist()):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, None), score))
            results.append(res)

        return new_state_dict, results

    @torch.no_grad()
    def work(self, data, beam_size, max_time_step, min_time_step=1):
        src_repr, src_mask = self.encode_step(data)
        mem_dict = {'encoder_state': src_repr,
                    'encoder_state_mask': src_mask}
        init_hyp = Hypothesis({}, [BOS], 0.)
        bsz = src_repr.size(1)
        beams = [Beam(beam_size, min_time_step, max_time_step, [init_hyp]) for i in range(bsz)]
        search_by_batch(self, beams, mem_dict)
        return beams

    def forward(self, data):
        src_repr, src_mask = self.encode_step(data)
        tgt_in_repr = self.embed_scale * self.tgt_embed(data['tgt_tokens_in']) + self.tgt_pos_embed(
            data['tgt_tokens_in'])
        tgt_in_repr = F.dropout(tgt_in_repr, p=self.dropout, training=self.training)
        tgt_in_mask = torch.eq(data['tgt_tokens_in'], self.vocabs['tgt'].padding_idx)
        attn_mask = self.self_attn_mask(data['tgt_tokens_in'])

        tgt_out = self.decoder(tgt_in_repr,
                               self_padding_mask=tgt_in_mask, self_attn_mask=attn_mask,
                               external_memories=src_repr, external_padding_mask=src_mask)

        return self.output(tgt_out, data)


class Encoder(nn.Module):
    def __init__(self, vocab, layers, embed_dim, ff_embed_dim, num_heads, dropout):
        '''

        Args:
            vocab:是dict{‘src’：Vocab(),'tgt':Vocab()}
            layers: 几层transformer，默认是 3X
            embed_dim: 默认是512
            ff_embed_dim: 默认是2048
            num_heads: 头的数量，一般是8
            dropout: tensor置为0的概率
        '''
        super(Encoder, self).__init__()
        self.vocab = vocab
        # 词嵌入层
        self.src_embed = Embedding(vocab.size, embed_dim, vocab.padding_idx)
        # 正余弦位置编码
        self.src_pos_embed = SinusoidalPositionalEmbedding(embed_dim)
        # 嵌入的縮放大小是根号dim
        self.embed_scale = math.sqrt(embed_dim)
        # transformer
        self.transformer = Transformer(layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.dropout = dropout

    def forward(self, input_ids):
        '''

        Args:
            input_ids: 源语言或者目标语言的token tensor

        Returns:源语言的表示和padding的掩码矩阵

        '''
        # 源语言的表示= 根号d倍的放大嵌入再加上位置信息，理由如下，基本思想是让位置编码相对小。
        # https://stackoverflow.com/questions/56930821/why-does-embedding-vector-multiplied-by-a-constant-in-transformer-model
        src_repr = self.embed_scale * self.src_embed(input_ids) + self.src_pos_embed(input_ids)
        # 按照dropout的概率随机把tensor的某个元素置为0，training那一项就是要不要应用dropout
        src_repr = F.dropout(src_repr, p=self.dropout, training=self.training)
        # 拿到一个掩码矩阵，这个矩阵是padding_idx的地方会变成false，否则是TRUE
        src_mask = torch.eq(input_ids, self.vocab.padding_idx)
        #
        src_repr = self.transformer(src_repr, self_padding_mask=src_mask)
        return src_repr, src_mask


class TokenDecoder(nn.Module):
    def __init__(self, vocabs, tgt_embed, label_smoothing):
        super(TokenDecoder, self).__init__()
        # weight: 词表大小 x 词嵌入维度
        self.output_projection = nn.Linear(
            tgt_embed.weight.shape[1],
            tgt_embed.weight.shape[0],
            bias=False,
        )
        self.output_projection.weight = tgt_embed.weight
        self.vocabs = vocabs
        self.label_smoothing = label_smoothing

    def forward(self, outs, data, work=False):
        '''

        :param outs: tgt_len, bsz, embed_dim
        :param data: 一个batch的数据
        :param work:
        :return:
        '''
        # 线性变换接softmax  lprobs: tgt_len, bsz, 词表大小
        lprobs = F.log_softmax(self.output_projection(outs), -1)

        if work:
            return lprobs
        loss, nll_loss = label_smoothed_nll_loss(lprobs, data['tgt_tokens_out'], self.label_smoothing,
                                                 ignore_index=self.vocabs['tgt'].padding_idx, sum=True)
        # 取出最高可能性的词
        # lprobs :tgt_len x bsz
        top1 = torch.argmax(lprobs, -1)
        acc = torch.eq(top1, data['tgt_tokens_out']).float().sum().item()
        loss = loss / data['tgt_num_tokens']
        return loss, acc


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, sum=True):
    '''

    Args:
        lprobs: 取了对数的概率
        target: 这个是gold，其实就是一个等差数列
        epsilon: 一个较小的值，label smoothing
        ignore_index:
        sum: bool值，是否求和

    Returns:

    '''
    # 如果少了一维，就把target在最后一维展开
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    # 负对数损失是target位置的负数，因为lprobs本身是log之后了，概率被映射在负无穷到0之间
    # target位置是最可能的值，应当越靠近0越好。所以取负数可以得到nll_loss
    nll_loss = -lprobs.gather(dim=-1, index=target)
    # 对lprobs的列求和，就相当于pdf里的第五页左侧公式的分母
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if sum:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
