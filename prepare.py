import argparse
import logging
import os
from collections import Counter

logger = logging.getLogger(__name__)


def make_vocab(batch_seq, char_level=False):
    '''
    这个就是data.py需要的filename打开的预处理，因为他返回的格式就是TOKEN,CNT
    Args:
        batch_seq: 批量句子，\n分割
        char_level: 字符级别的统计？

    Returns:一个计数器，或者两个计数器

    '''
    cnt = Counter()
    # 这个其实等价于直接 cnt=Counter(batch_seq)。可以得到一个dict{"TOKEN"->str:CNT->int}
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    # 字符级别的计数器
    char_cnt = Counter()
    # 从cnt取出出现次数最多的（token->x,count->y),统计这个token中出现最多次的字符
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    '''
    按词频从大到小的顺序写入词频表
    Args:
        vocab: 词汇表
        path: 写入的路径

    Returns:NONE

    '''
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n' % (x, y))


def init_parser():
    '''

    return: 一个parser
    '''
    parser = argparse.ArgumentParser()
    # input
    # -----------------------------------------------------------
    # 训练源语
    parser.add_argument('--train_data_src', type=str)
    # 训练目标语
    parser.add_argument('--train_data_tgt', type=str)
    # ---------------------------------------------------------

    # output
    # --------------------------------------------------------
    # 清洗后生成的训练双语路径
    parser.add_argument('--output_file', type=str)
    # 生成的源语词汇表路径
    parser.add_argument('--vocab_src', type=str)
    # 生成的目标语词汇表路径
    parser.add_argument('--vocab_tgt', type=str)
    # --------------------------------------------------------

    # config
    # --------------------------------------------------------
    # 源语言和目标语言的长短比阈值
    parser.add_argument('--ratio', type=float, default=1.5)
    # 句子对最大长度和最小长度
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=175)
    # --------------------------------------------------------
    return parser.parse_args()


if __name__ == "__main__":
    # 初始化命令解析器
    args = init_parser()

    # 设置日志格式
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info("make vocabularies")
    if not os.path.exists(args.output_file):  # 如果路径不存在
        os.makedirs(os.path.dirname(args.output_file))
    output = open(args.output_file, 'w')
    src_lines = []
    tgt_lines = []
    tot_lines = 0
    absCnt=0
    relCnt=0
    for src_line, tgt_line in zip(open(args.train_data_src).readlines(),
                                  open(args.train_data_tgt).readlines()):
        src_line = src_line.strip().split()
        tgt_line = tgt_line.strip().split()
        tot_lines += 1

        # 检查句对的绝对长度和相对长度比例
        if args.min_len <= len(src_line) <= args.max_len and args.min_len <= len(tgt_line) <= args.max_len:
            # 这里如果写成 1/args.ratio 可能会0中断，所以还是写全一点
            if len(src_line) / len(tgt_line) > args.ratio or len(tgt_line) / len(src_line) > args.ratio:
                relCnt+=1
                continue

            output.write(' '.join(src_line) + '\t' + ' '.join(tgt_line) + '\n')
            # 把合法的句子分别存到两个列表里
            src_lines.append(src_line)
            tgt_lines.append(tgt_line)
        else:
            absCnt+=1
    output.close()

    # 得到一个词频计数器 counter(dict{token,cnt})
    src_vocab = make_vocab(src_lines)
    tgt_vocab = make_vocab(tgt_lines)
    logger.info("清洗前句对数 : %d \t 清洗后句对数: %d ",tot_lines,len(src_lines))
    logger.info("语料的有效率是： %f ",len(src_lines)/tot_lines)
    logger.info("因绝对长度和相对长度比例被清洗的数量分别为 %d %d",absCnt,relCnt)

    print('write vocabularies')
    # 按词频降序写入词频文件中
    write_vocab(src_vocab, args.vocab_src)
    write_vocab(tgt_vocab, args.vocab_tgt)

