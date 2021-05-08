# jupyter_notebook
My personal pytorch test

## Attention-Based Conv2d Pruning
### 实现 Attention-Based Conv2d Pruning

#### 1. 将 Weight data 类型转换和求绝对值
- x [N<sub>in</sub>, N<sub>out</sub>, kernel_size[0], kernel_size[1]]
- A [N<sub>in</sub>, N<sub>out</sub>, kernel_size[0] * kernel_size[1]]
- C = N<sub>in</sub>, H =  N<sub>out</sub>, W = kernel_size[0] * kernel_size[1]
- A [C, H, W]

#### 2. 沿通道方向绝对值之和 
F(A)=∑<sub>i=1</sub><sup>C</sup> |A<sub>i</sub>|

#### 3. 计算 ||F(A)||<sup>2</sup>  二范数的平方

#### 4. 计算 F(A) / ||F(A)||<sup>2</sup> 

#### 5. 计算 F(A<sub>j</sub>) / ||F(A<sub>j</sub>)||<sup>2</sup>  和 gamma = ∑ | F(A) / ||F(A)||<sup>2</sup> - F(A<sub>j</sub>) / ||F(A<sub>j</sub>)||<sup>2</sup> |


### 1. 测试 `linalg.norm` 与 `nn.functional.normalize` 的区别
- 二范数：( |x<sub>1</sub>|<sup>2</sup> + |x<sub>2</sub>|<sup>2</sup> +...+|x<sub>n</sub>|<sup>2</sup>)<sup>1/2</sup>
- `torch.norm` 已废弃，不建议使用
- [linalg.norm](https://pytorch.org/docs/stable/linalg.html?highlight=norm#torch.linalg.norm) 在计算矩阵二范数时，没有开平方根
- [nn.functional.normalize](https://pytorch.org/docs/stable/nn.functional.html?highlight=normalize#torch.nn.functional.normalize) 在计算矩阵二范数时，开了平方根
- 即 `torch.sqrt(torch.linalg.norm(data))` 等价于 `fn.normalize(data)`