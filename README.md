# PearPigLin's Jupyter Notebook

## Attention-Based Conv2d Pruning

1. 将 Weight data 类型转换和求绝对值: A [C, H, W]
2. 计算F(A)=∑<sub>i=1</sub><sup>C</sup> |A<sub>i</sub>| 沿通道方向绝对值之和 
3. 计算 ||F(A)||<sup>2</sup>  二范数的平方
4. 计算 F(A) / ||F(A)||<sup>2</sup> 
5. 计算 F(A<sub>j</sub>) / ||F(A<sub>j</sub>)||<sup>2</sup>  和 gamma = ∑ | F(A) / ||F(A)||<sup>2</sup> - F(A<sub>j</sub>) / ||F(A<sub>j</sub>)||<sup>2</sup> |


## Attention-based Feature Pruning
1. 提取`batchSize=1`的Cifar10数据集
2. 加载VGG19_BN并进行每一层输入输出，找到ReLU的输出

## Pytorch More Ways
1. list 转成 array 实现减法
2. np.reshape() 与 np.resize() 是不同的
3. vars() 提供打印变量的所有参数
4. gt() 大于操作，返回值为 True or False
5. ng.argwhere(x > 0) 返回x中大于0的数组元组的索引


## Pytorch Pretrained VGG19
1. Pretrained VGG19 Model with Batch Normalization
	- 1.1 VGG19 Weight
	- 1.2 VGG19 Convolution Layer
2. Pretrained VGG19 Model without Batch Normalization
	- 2.1 features[0]: Conv2d
	- 2.2 features[2]: Conv2d
	- 2.3 Prune Conv2d Bias
	- 2.4 Prune Conv2d Weight