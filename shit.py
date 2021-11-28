import math
from typing import List

import numpy as np
import torch
a = torch.randn(4, 3,2)
print(a)
print(torch.argmax(a, -1))