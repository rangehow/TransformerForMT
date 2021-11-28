import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np


def move_to_device(maybe_tensor, device):
    """

    Args:
        maybe_tensor:
        device: torch.device('cuda'/'cpu', OPTIONAL[index])

    Returns:处理完后的数据

    """
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        # from_numpy是浅拷贝方法，所以这个返回的tensor是不能改变大小的，对其的操作也会反映到对应的ndarray上，反之亦然。
        # contiguous是申请把tensor内的元素排布变得在内存上连续，比如对一个3X4的tensor做转置成4X3的tensor，
        # pytorch并不会修改这个tensor在内存上的存储方式，而是只是会修改一个元信息（用于从维度索引变成能在一维的内存上访存的索引）
        # 这在flatten等操作上需要保证tensor内元素不仅语义连续，内存分布也要连续，见https://zhuanlan.zhihu.com/p/64551412
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        # 如果传进来的是dict，还是返回dict，但是要把key-value对的value的device用to方法处理一下
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        # 还是返回list，但是对其中的[x1,x2,..] 的x1 做变换
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        # 不改变结构，只是简单地对tuple中的元素做变换
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


class Statistics:
    '''
    运行状态参数的字典类，默认情况下只会保存steps
    '''

    def __init__(self, key_value_dict=None, **kwargs):
        self.statistics = {'steps': 0}
        if key_value_dict is not None:
            for x in key_value_dict:
                self.statistics[x] = key_value_dict[x]
        for x in kwargs:
            self.statistics[x] = kwargs[x]

    def update(self, key_or_dict, value=None):
        if value is None:
            assert isinstance(key_or_dict, dict)
            for key in key_or_dict:
                if key not in self.statistics:
                    self.statistics[key] = 0.
                self.statistics[key] += key_or_dict[key]
        else:
            assert isinstance(key_or_dict, str)
            if key_or_dict not in self.statistics:
                self.statistics[key_or_dict] = 0.
            self.statistics[key_or_dict] += value

    def __getitem__(self, attr):
        return self.statistics[attr]

    def step(self):
        '''

        Returns:字典里的step+1

        '''
        self.statistics['steps'] += 1


def data_proc(data, queue):
    '''

    Args:
        data: dataLoader 也就是一个[batch_size,seq_length] 这样的类
        queue: 消息共享队列

    Returns:

    '''
    #这个x就是一批批的数据，按不同的batch分
    # 最后queue里的数据是这样的(第几批次的数据，该批次的第几句话，该批次的第几个词)
    for x in data:
        queue.put(x)
    queue.put('EPOCHDONE')


def asynchronous_load(data_loader):
    '''

    Args:
        data_loader: 数据包装类

    Returns: 一个batch大小的数据

    '''
    queue = mp.Queue(10)
    # 创建对象，这个做完之后会得到一个一维的队列，里面的每个元素是一个二维数组
    # 这个二维数组的内容就是[[第一句话的第一个词，第二句话的第一个词...],[]]
    # 也就是（句子长度，batch——size）
    data_generator = mp.Process(target=data_proc, args=(data_loader, queue))
    # 生成进程
    data_generator.start()
    done = False
    while not done:
        batch = queue.get()
        # 这里为什么这么判断呢，因为在data_proc最后面加了一个'EPOCHDONE'，这是个str
        # 换句话说，这个data里本身没有字符串，因为已经全部在前面的ListToTensor变成词表独热码了
        if isinstance(batch, str):
            done = True
        else:
            #所以每次返回的都是一batch的数据,这是个二维的东西
            yield batch
    data_generator.join()
