# PearPigLin's Jupyter Notebook


### Attention-based Conv2d Pruning
1. 将 Weight data 类型转换和求绝对值: **A [C, H, W]**
2. 计算 **F(A)=∑<sub>i=1</sub><sup>C</sup> |A<sub>i</sub>|** 沿通道方向绝对值之和 
3. 计算 **||F(A)||<sup>2</sup>**  二范数的平方
4. 计算 **F(A) / ||F(A)||<sup>2</sup>**
5. 计算 **F(A<sub>j</sub>) / ||F(A<sub>j</sub>)||<sup>2</sup>** 和 **gamma = ∑ | F(A) / ||F(A)||<sup>2</sup> - F(A<sub>j</sub>) / ||F(A<sub>j</sub>)||<sup>2</sup> |**


### Attention-based Feature Pruning
1. Dataset CIFAR10
    1. one minimum iterator
    2. one data, one batch of iterator
2. Model VGG19_BN
    1. pretrained ImageNet VGG19 Model
    2. define empty VGG19 Model
    3. initialize model weights
    4. load weight data
    5. the first conv2d converting of model
    6. find BatchNorm2d & nn.ReLU converting
3. Activation-based Gramma
    1. batch size activation-based gamma
4. Prune
    1. number of channels
    2. all channels' gamma
    3. threshold
    4. prune

### Pytorch Loss Function
1. 损失函数简介
2. 损失函数的本质
3. 损失函数实例
    1. 绝对值损失
        - [nn.L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html?highlight=l1loss#torch.nn.L1Loss)
        - [nn.SmoothL1Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html?highlight=smoothl1loss#torch.nn.SmoothL1Loss)
    2. 平方损失
        - [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html?highlight=mseloss#torch.nn.MSELoss)
    3. 对数损失
        - [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html?highlight=nllloss#torch.nn.NLLLoss)
        - [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
        - [nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss)
        - [nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html?highlight=multilabelsoftmarginloss#torch.nn.MultiLabelSoftMarginLoss)
    
### Pytorch More Ways
1. list 转成 array 实现减法
2. np.reshape() 与 np.resize() 是不同的
3. vars() 提供打印变量的所有参数
4. gt() 大于操作，返回值为 True or False
5. ng.argwhere(x > 0) 返回x中大于0的数组元组的索引
6. squeeze() 与 unsqueeze()
7. nn.ReLU()
8. linalg.norm() 与 nn.functional.normalize() 的区别
9. Pytorch Variable


### Pytorch Pretrained VGG19
0.1 cuda is available()

0.2 pytorch version
1. Pretrained VGG19 Model with Batch Normalization
	- VGG19 Weight
	- VGG19 Convolution Layer
2. Pretrained VGG19 Model without Batch Normalization
	- features[0]: Conv2d
	- features[2]: Conv2d
	- Prune Conv2d Bias
	- Prune Conv2d Weight