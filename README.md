

## 模型精度：

| 模型                 | CIFAR10 Top1 acc |
| -------------------- | ---------------- |
| RotNet+conv(pytorch) | 91.16            |
| RotNet+conv(paddle)  | 91.6238          |



## 训练：

##### RotNet_NIN4blocks训练：

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks
```



##### ConvClassifier训练：

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats
```



##### 训练日志与训练模型

classifier_net_epoch92 放在./experiments/CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats

model_net_epoch102 放在./experiments/CIFAR10_RotNet_NIN4blocks

model_opt_epoch102 放在./experiments/CIFAR10_RotNet_NIN4blocks



##### [百度网盘](https://pan.baidu.com/s/1tPqxjbO6E3gWlcOMpqa02w)

提取码：k1gf



## 评估：

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats --evaluate --checkpoint=92
```

