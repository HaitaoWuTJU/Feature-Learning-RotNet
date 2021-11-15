import numpy as np
from paddle import fluid

x = np.ones([2, 2], np.float32)
with fluid.dygraph.guard():
    inputs = []
    for _ in range(10):
        inputs.append(fluid.dygraph.to_variable(x))
    ret = fluid.layers.sums(inputs)
    print(ret.numpy())