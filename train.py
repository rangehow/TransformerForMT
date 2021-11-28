import argparse
import logging
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from data import Vocab, DataLoader, BOS, EOS
from optim import get_inverse_sqrt_schedule_with_warmup, Adam
from utils import move_to_device, set_seed, average_gradients, Statistics
from module import Generator
from work import validate

logger = logging.getLogger(__name__)


def init_parser():
    parser = argparse.ArgumentParser()
    # input
    # -----------------------------------------------------------------------------------------
    # vocabs
    parser.add_argument('--src_vocab', type=str, default='src.vocab')
    parser.add_argument('--tgt_vocab', type=str, default='tgt.vocab')
    # corpus
    parser.add_argument('--train_data', type=str, default='train.txt')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    # 验证集和测试集可以是路径
    parser.add_argument('--dev_data', type=str, default='dev.txt')
    parser.add_argument('--test_data', type=str, default='test.txt')
    # -----------------------------------------------------------------------------------------

    # config
    # -----------------------------------------------------------------------------------------
    # architecture
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--ff_embed_dim', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    # dropout / label_smoothing
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    # training
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--total_train_steps', type=int, default=100000)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=4096)
    parser.add_argument('--dev_batch_size', type=int, default=4096)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--only_save_best', action='store_true')
    # -----------------------------------------------------------------------------------------

    # output
    # -----------------------------------------------------------------------------------------
    parser.add_argument('--ckpt', type=str, default='ckpt')
    # -----------------------------------------------------------------------------------------

    # distributed training
    # -----------------------------------------------------------------------------------------
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='55555')
    parser.add_argument('--start_rank', type=int, default=0)
    # -----------------------------------------------------------------------------------------
    return parser.parse_args()


def main(args, local_rank):
    # 设置训练卡
    #os.environ['CUDA_VISIBLE_DEVICES'] = ""
    device = torch.device('cuda', local_rank)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    vocabs = dict()
    vocabs['src'] = Vocab(args.src_vocab, 0, [BOS, EOS])
    vocabs['tgt'] = Vocab(args.tgt_vocab, 0, [BOS, EOS])
    # 单进程或主进程输出args和vocab的信息
    if args.world_size == 1 or (dist.get_rank() == 0):
        logger.info(args)
        for name in vocabs:
            logger.info("vocab %s, size %d, coverage %.3f",
                        name, vocabs[name].size, vocabs[name].coverage)

    set_seed(19940117)

    model = Generator(vocabs, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout,
                      args.enc_layers, args.dec_layers, args.label_smoothing)
    if args.resume_ckpt:
        model.load_state_dict(torch.load(args.resume_ckpt)['model'])

    global_step = 0
    if args.world_size > 1:
        set_seed(19940117 + dist.get_rank())
    # 将模型加载到指定设备上
    model = model.to(device)
    optimizer = Adam([{'params': model.parameters(), 'initial_lr': args.embed_dim ** -0.5 * 0.1}], betas=(0.9, 0.98), eps=1e-9)

    # 学习率衰减，warmup
    lr_schedule = get_inverse_sqrt_schedule_with_warmup(optimizer, args.warmup_steps, args.total_train_steps)
    # 数据加载
    train_data = DataLoader(vocabs, args.train_data, args.per_gpu_train_batch_size,
                            for_train=True, rank=local_rank, num_replica=args.world_size)
    step, epoch = 0, 0
    # 参数字典,用于输出训练时的日志
    tr_stat = Statistics()
    logger.info("start training")
    model.train()
    best_dev_bleu = 0.

    # 默认训练到10万步
    while global_step <= args.total_train_steps:
        for batch in train_data:
            # step_start = time.time()
            batch = move_to_device(batch, device)
            loss, acc = model(batch)
            # 更新指标
            tr_stat.update({'loss': loss.item() * batch['tgt_num_tokens'],
                            'tokens': batch['tgt_num_tokens'],
                            'acc': acc})
            tr_stat.step()
            # 计算梯度，反向传播
            loss.backward()
            # step_cost = time.time() - step_start
            # print ('step_cost', step_cost)
            step += 1
            # 如果step取余这个玩意不等于 gradient_accumulation_steps，就跳过。
            if not (step % args.gradient_accumulation_steps == -1 % args.gradient_accumulation_steps):
                continue
            # 如果工作组大于1的话，求梯度平均值
            if args.world_size > 1:
                average_gradients(model)
            # 梯度裁剪，当梯度小于/大于阈值时，更新的梯度为阈值，抵抗梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数
            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()
            global_step += 1
            # 主进程负责输出
            if args.world_size == 1 or (dist.get_rank() == 0):
                # 输出日志信息
                if global_step % args.print_every == -1 % args.print_every:
                    logger.info("epoch %d, step %d, loss %.3f, acc %.3f", epoch, global_step,
                                tr_stat['loss'] / tr_stat['tokens'], tr_stat['acc'] / tr_stat['tokens'])
                    tr_stat = Statistics()
                # 验证
                if global_step % args.eval_every == -1 % args.eval_every:
                    model.eval()
                    # 在两倍的warmup步数之前没必要开太多
                    max_time_step = 256 if global_step > 2 * args.warmup_steps else 5
                    bleus = []
                    # dev_data是一个存放着验证集文件的列表
                    for cur_dev_data in args.dev_data:
                        # 逐个加载文件
                        dev_data = DataLoader(vocabs, cur_dev_data, args.dev_batch_size, for_train=False)
                        bleu = validate(device, model, dev_data, beam_size=5, alpha=0.6, max_time_step=max_time_step)
                        bleus.append(bleu)
                    bleu = sum(bleus) / len(bleus)
                    logger.info("epoch %d, step %d, dev bleu %.2f", epoch, global_step, bleu)
                    # 如果验证集上比以往更好就调入测试集
                    if bleu > best_dev_bleu:
                        testbleus = []
                        for cur_test_data in args.test_data:
                            test_data = DataLoader(vocabs, cur_test_data, args.dev_batch_size, for_train=False)
                            testbleu = validate(device, model, test_data, beam_size=5, alpha=0.6,
                                                max_time_step=max_time_step)
                            testbleus.append(testbleu)
                        testbleu = sum(testbleus) / len(testbleus)
                        logger.info("epoch %d, step %d, test bleu %.2f", epoch, global_step, testbleu)
                        # 覆盖以往的best.pt
                        torch.save({'args': args, 'model': model.state_dict()}, '%s/best.pt' % (args.ckpt,))
                        # 存储
                        if not args.only_save_best:
                            torch.save({'args': args, 'model': model.state_dict()},
                                       '%s/epoch%d_batch%d_devbleu%.2f_testbleu%.2f' % (
                                           args.ckpt, epoch, global_step, bleu, testbleu))
                        best_dev_bleu = bleu
                    model.train()

            if global_step > args.total_train_steps:
                break
        epoch += 1
    logger.info('rank %d, finish training after %d steps', local_rank, global_step)


def init_processes(local_rank, args, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    main(args, local_rank)


if __name__ == "__main__":
    args = init_parser()
    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)
    # 判断是不是一个路径，是的话把这个路径下的所有文件和路径拼接起来，形成一个路径list[验证集1，验证集2 ]
    if os.path.isdir(args.dev_data):
        args.dev_data = [os.path.join(args.dev_data, file) for file in os.listdir(args.dev_data)]
    # 不是路径的话，就只有一个文件了
    else:
        args.dev_data = [args.dev_data]
    # 类似，加载测试集
    if os.path.isdir(args.test_data):
        args.test_data = [os.path.join(args.test_data, file) for file in os.listdir(args.test_data)]
    else:
        args.test_data = [args.test_data]

    if args.world_size == 1:
        main(args, 0)
        exit(0)

    mp.spawn(init_processes, args=(args,), nprocs=args.gpus)
