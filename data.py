import random, logging
import torch
from torch import nn
import numpy as np
from utils import move_to_device

PAD, UNK, BOS, EOS = '<PAD>', '<UNK>', '<BOS>', '<EOS>'

logger = logging.getLogger(__name__)


class Vocab(object):

    def __init__(self, filename, min_occur_cnt, specials=None):
        """

        Args:
            filename: 词频表文件
            min_occur_cnt: 最小词频阈值
            specials: 特殊标识字符 如BOS EOS
        """
        # idx2token= ['<PAD>', '<UNK>', '<BOS>'] / ['<PAD>', '<UNK>', '<BOS>','<EOS>']
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        # 词的总数
        num_tot_tokens = 0
        # 词频表中词频大于最小阈值的计数
        num_invocab_tokens = 0
        '''
        这个语料库是这种样子
        token \t cnt \n
        
        '''
        for line in open(filename).readlines():
            try:
                token, cnt = line.rstrip('\n').split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                logger.info("(Vocab)Illegal line:", line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_invocab_tokens += cnt

        # 有效词的覆盖率
        self.coverage = num_invocab_tokens / num_tot_tokens
        # 把dict变成 {token,下标索引} 的形式
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        # 把局部变量作为属性保存下来
        self._idx2token = idx2token
        # 存储PAD的下标，int类型
        self._padding_idx = self._token2idx[PAD]
        # 存储unk的下标，int类型
        self._unk_idx = self._token2idx[UNK]

    # 属性装饰器，把属性变成只读类型。
    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        '''

        Args:
            x: 索引列表或者单个索引

        Returns:根据x的内容从token list中取出对应的token

        '''
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        '''

        Args:
            x:token列表或者单个token

        Returns:如果x存在dict中，返回对应的下标，否则返回unk的下标

        '''
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)


def _back_to_txt_for_check(tensor, vocab, local_idx2token=None):
    '''
    把vocab中的合法token(除了PAD)取出来打印
    Args:
        tensor: 索引张量
        vocab: 单词表
        local_idx2token:未知

    Returns: NONE

    '''
    for bid, xs in enumerate(tensor.t().tolist()):
        txt = []
        for x in xs:
            if x == vocab.padding_idx:
                # 如果x是填充下标，就退出。这里这么写，可能是因为填充是在句子尾部，减少不必要的遍历。
                break
            if x >= vocab.size:
                # 如果x超出了vocab的大小，其实是一种不太正常的情况。
                # 检查以下两个是否非空，否则弹出异常
                assert local_idx2token is not None
                assert local_idx2token[bid] is not None
                tok = local_idx2token[bid][x]
            else:
                tok = vocab.idx2token(x)
            txt.append(tok)
        # txt是 [txt,tok]的形式，现在转换成"txt tok"=>txt
        txt = ' '.join(txt)
        print(txt)
        print('-' * 55)
    print('=' * 55)


def ListsToTensor(xs, vocab=None, worddrop=0.):
    '''

    Args:
        xs: 一个二维list batch_size *seqLength
        vocab: Vocab
        worddrop:按照这个概率丢弃一些词语，即改成UNK


    Returns:seqLength * batch_size 做了padding，根据vocab转成了tensor

    '''
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w):
        if vocab is None:
            return w
        if isinstance(w, list):
            # 如果传进来的是一个列表，就把列表中的每个元素依次重新处理
            return [toIdx(_) for _ in w]
        if random.random() < worddrop:
            # 随机把某个词根据worddrop替换成UNK
            return vocab.unk_idx

        # 根据TOKEN返回他在vocab里面的下标
        return vocab.token2idx(w)

    # 找出最长的句子的长度，用于填充
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        # 根据Vocab转独热编码，做padding
        y = toIdx(x) + [pad] * (max_len - len(x))
        ys.append(y)
    # 从 batch_size * seqLength -> seqLength * batch_size
    data = np.transpose(np.array(ys))
    return data


def batchify(data, vocabs, max_seq_len):
    '''

    :param data: {'src_tokens': src_tokens, 'tgt_tokens': tgt_tokens,'index': i}
    :param vocabs: {'src'/'tgt':Vocab()}
    :param max_seq_len: 默认175
    :return:
    '''
    src_tokens = [x['src_tokens'][:max_seq_len]+[EOS] for x in data]
    tgt_tokens_in = [[BOS] + x['tgt_tokens'][:max_seq_len] for x in data]
    tgt_tokens_out = [x['tgt_tokens'][:max_seq_len] + [EOS] for x in data]

    src_token = ListsToTensor(src_tokens, vocabs['src'])

    tgt_token_in = ListsToTensor(tgt_tokens_in, vocabs['tgt'])
    tgt_token_out = ListsToTensor(tgt_tokens_out, vocabs['tgt'])
    #是padding的地方是0,否则是1
    not_padding = (tgt_token_out != vocabs['tgt'].padding_idx).astype(np.int64)
    # 因为是 seqLength * batch_size ，先沿着第0维度求和，再求和，这样可以得到这批次词的数目
    tgt_lengths = np.sum(not_padding, axis=0)
    tgt_num_tokens = int(np.sum(tgt_lengths))

    # not_padding = (src_token != vocabs['src'].padding_idx).astype(np.int64)
    # src_lengths = np.sum(not_padding, axis=0)
    ret = {
        'src_tokens': src_token,
        # 'src_lengths': src_lengths,
        'tgt_tokens_in': tgt_token_in,
        'tgt_tokens_out': tgt_token_out,
        'tgt_num_tokens': tgt_num_tokens,
        # 'tgt_lengths': tgt_lengths,
        'tgt_raw_sents': [x['tgt_tokens'] for x in data],
        'indices': [x['index'] for x in data]
    }
    return ret


# 训练数据加载类
class DataLoader(object):
    def __init__(self, vocabs, filename, batch_size, for_train, max_seq_len=256, rank=0, num_replica=1):
        '''

        :param vocabs:dict(str(['src'/'tgt']):Vocab()) src和tgt的Vocab
        :param filename: 双语对齐训练集的路径
        :param batch_size:
        :param for_train:
        :param max_seq_len:
        :param rank:
        :param num_replica:
        '''
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train

        src_tokens, tgt_tokens = [], []
        src_sizes, tgt_sizes = [], []

        # 分布式训练，readlines会把所有的数据按行读进一个list里面
        # 通过控制索引把数据分发到不同的卡上
        # rank是当前GPU，num_replica是world_size，通过控制开始位置和步长，刚好可以把数据错开
        for line in open(filename).readlines()[rank::num_replica]:
            try:
                # 按照\t分给源语言和目标语言(str)
                src, tgt = line.strip().split('\t')
            except:
                continue
            # 按词分开后的一维list
            src, tgt = src.split(), tgt.split()
            # 统计句子长度的一维list
            src_sizes.append(len(src))
            tgt_sizes.append(len(tgt))

            src_tokens.append(src)
            tgt_tokens.append(tgt)


        self.src = src_tokens
        self.tgt = tgt_tokens
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)
        # 默认175
        self.max_seq_len = max_seq_len

        logger.info("(DataLoader rank %d) read %s file with %d paris. max src len: %d, max tgt len: %d", rank, filename,
                    len(self.src), self.src_sizes.max(), self.tgt_sizes.max())

    def __len__(self):
        return len(self.src)

    def __iter__(self):
        indices = np.argsort(self.src_sizes, kind='mergesort') # 保持一个batch里的数据长度尽可能相似，对源语言长度排一下序
        batches = []
        num_tokens, batch = 0, []
        for i in indices:
            # 用于计数这个batch有没有超过batch_size，统计的词数
            num_tokens += 1 + max(self.src_sizes[i], self.tgt_sizes[i])
            if num_tokens > self.batch_size:
                #这说明一个batch完成了，把batch(句子序号列表)加入batches
                batches.append(batch)
                # 重置变量
                num_tokens, batch = 1 + max(self.src_sizes[i], self.tgt_sizes[i]), [i]
            else:
                batch.append(i)

        if not self.train or num_tokens > self.batch_size / 2:
            #会出现最后一组不够batch_size的情况
            # 如果是测试，就直接加入；如果是训练，超过batch_size一半再加入
            batches.append(batch)

        if self.train:
            #如果是训练，打乱一下batches元素的排列
            random.shuffle(batches)
        #前面都是下标，现在把数据装进来
        for batch in batches:
            data = []
            for i in batch:
                src_tokens = self.src[i]
                tgt_tokens = self.tgt[i]
                item = {'src_tokens': src_tokens, 'tgt_tokens': tgt_tokens,'index': i}
                data.append(item)

            '''
            logger.info('data is ')
            logger.info(data)
            '''
            yield batchify(data, self.vocabs, self.max_seq_len)


def parse_config():
    """
    根据命令行中调用的指令解析各个参数

    Returns:一个ArgumentParser对象

    """
    import argparse
    parser = argparse.ArgumentParser()
    # --是按照词解析的一种命名规范，default是当这一项为空时填充上去的东西
    # 这一段是指导parser有几个参数要解析，每个的类型是什么
    parser.add_argument('--src_vocab', type=str, default='es.vocab')
    parser.add_argument('--tgt_vocab', type=str, default='en.vocab')
    parser.add_argument('--train_data', type=str, default='dev.mem.txt')
    parser.add_argument('--train_batch_size', type=int, default=4096)

    # 返回解析后的结果，可以直接通过add_argument的名字调用从sys.argv中得到的参数列表
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()

    vocabs = dict()
    vocabs['src'] = Vocab(args.src_vocab, 0, [EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab, 0, [BOS, EOS])

    train_data = DataLoader(vocabs, args.train_data, args.train_batch_size, for_train=True)
    for d in train_data:
        d = move_to_device(d, torch.device('cpu'))
        for k, v in d.items():
            if 'raw' in k:
                continue
            try:
                print(k, v.shape)
            except:
                print(k, v)
        _back_to_txt_for_check(d['src_tokens'][:, 5:6], vocabs['src'])
        _back_to_txt_for_check(d['tgt_tokens_in'][:, 5:6], vocabs['tgt'])
        _back_to_txt_for_check(d['tgt_tokens_out'][:, 5:6], vocabs['tgt'])
        bsz = d['tgt_tokens_out'].size(1)
        _back_to_txt_for_check(d['all_mem_tokens'][:, 5::bsz], vocabs['tgt'])
        break
