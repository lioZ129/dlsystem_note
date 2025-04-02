# 归一化和正则化

## 大纲
- 归一化
- 正则化
- 优化、初始化、归一化、正则化的交互

### 归一化

- 归一化为了将特征之间的差异缩小，例如特征的计量单位不同使得其差异巨大，需要通过01或Z-score归一化使其特征的数据都控制在一定范围内，便于特征与特征之间的起作用的平均
- 详细举例：身高体重两特征，一个基本在1-2（米），一个在50以上（kg），则对损失函数（例如为差平方）的优化就会偏向增大体重对应参数的权重

> BatchNorm是对一个batch-size样本内的每个特征[分别]做归一化，LayerNorm是[分别]对每个样本的所有特征做归一化

#### 1. Layer normalization 层归一化
- 标准化（均值零和方差一）每一层的激活函数值
- $\hat{z}_{i+1} =\sigma_i\left(W_i^T z_i+b_i\right) \\ \\
z_{i+1}  =\frac{\hat{z}_{i+1}-\mathbf{E}\left[\hat{z}_{i+1}\right]}{\left(\mathbf{V a r}\left[\hat{z}_{i+1}\right]+\epsilon\right)^{1 / 2}}$

- 其能“修复”不同层激活范数的问题，效果如下图：
![alt text](image-17.png)


#### 2. Batch normalization

