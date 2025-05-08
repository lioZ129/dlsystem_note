# 神经网络库实现

## 大纲
- 实操课

### 1. Needle库基础复习
#### 参数更新优化
- **问题**：直接使用`w = w + (-lr) * grad`会构建冗余计算图
- **解决**：通过操作`w.data`进行原地更新，避免计算图膨胀
- 示例代码：
  ```py
  w.data = w.data + (-lr) * grad.data  # 直接修改数据，不构建计算图
  ```

#### 数值稳定性（基于Softmax）
- **问题**：直接计算指数可能导致数值溢出
- **解决**：减去最大值后计算，保持数值稳定
- **数学公式**：
  \[
  z_i = \frac{\exp(x_i - \max(x))}{\sum \exp(x_j - \max(x))}
  \]
- 代码示例：
  ```py
  def softmax_stable(x):
      x = x - np.max(x) # here
      z = np.exp(x)
      return z / np.sum(z)
  ```


### 2. 神经网络模块设计
#### 参数类 (Parameter)
- 功能：继承自`Tensor`，标记为可训练参数

```py
  class Parameter(ndl.Tensor):
      """可训练参数"""
```

#### 模块基类 (Module)
- 功能：管理参数和子模块，提供统一接口
- 关键方法：
  - `parameters()`: 递归收集所有参数
  - `forward()`: 定义前向传播逻辑（需子类实现）
- 代码示例：

  ```py
  class Module:
      def parameters(self):
          return _get_params(self.__dict__)  # 把当前 Module 实例的所有属性传递给 _get_params 函数,递归收集参数
      
      def __call__(self, *args, **kwargs):
          return self.forward(*args, **kwargs)
  ```


  > 辅助函数 _get_params(value):
  ```py
  def _get_params(value):
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _get_params(v)
        return params
    if isinstance(value, Module):
        return value.parameters()
    return []
  ```

#### 具体模块实现
- **ScaleAdd模块**：实现线性变换 \( y = x \cdot s + b \)
  
  ```py
  class ScaleAdd(Module):
      def __init__(self, init_s=1, init_b=0):
          self.s = Parameter([init_s], dtype="float32")
          self.b = Parameter([init_b], dtype="float32")
      
      def forward(self, x):
          return x * self.s + self.b
  ```

- **多路径模块 (MultiPathScaleAdd)**：组合多个子模块
  ```py
  class MultiPathScaleAdd(Module):
      def __init__(self):
          self.path0 = ScaleAdd()
          self.path1 = ScaleAdd()
      
      def forward(self, x):
          return self.path0(x) + self.path1(x)
  ```

- 模块示例用法：
```py
# 代码
mpath = MultiPathScaleAdd()
mpath.parameters()

# 输出
# [needle.Tensor([1.]),
#  needle.Tensor([0.]),
#  needle.Tensor([1.]),
#  needle.Tensor([0.])]
```


### 3. 损失函数与优化器
#### 均方误差损失 (L2Loss)
- **实现**：计算预测值与目标值的平方差
  ```py
  class L2Loss(Module):
      def forward(self, x, y):
          z = x - y
          return z * z  # 平方误差（任意假设的）
  ```

#### 随机梯度下降 (SGD)
- **功能**：更新参数，学习率控制步长
- **关键方法**：
  - `reset_grad()`: 清零梯度
  - `step()`: 执行参数更新
- 代码：
  ```py
  class Optimizer:
    def __init__(self, params):
        self.params = params

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplemented()


  class SGD(Optimizer):
      def __init__(self, params, lr):
          self.params = params
          self.lr = lr
      
      def step(self):
          for w in self.params:
              w.data = w.data - self.lr * w.grad  # 参数更新
  ```

### 4. 训练循环示例
- 流程：
  1. 前向传播计算损失
  2. 反向传播计算梯度
  3. 优化器更新参数

- **感受模块化的优势：可以只改变前面的模型而不改变后文批处理下降**
- 代码：
  ```py
  model = MultiPathScaleAdd()
  loss_fn = L2Loss()
  opt = SGD(model.parameters(), lr=0.01)
  
  for epoch in range(10):
      opt.reset_grad()
      pred = model(x)
      loss = loss_fn(pred, y)
      loss.backward()
      opt.step()
      print(loss.numpy())
  ```


### 5. 初始化策略（Kaiming初始化）
- 针对ReLU激活函数设计，防止梯度消失/爆炸
- 公式：权重初始化为 \(\mathcal{N}(0, \sqrt{2/n_{in}})\) （对于ReLU）
- 通过保持各层激活值方差一致，确保稳定训练


### 6. 高级功能：融合操作与元组
- **融合操作**：单个操作返回多个输出（如`fused_add_scalars`）。
- **TensorTuple**：表示元组类型的值，支持索引操作。
- 代码示例：
  ```py
  z = ndl.ops.fused_add_scalars(x, 1, 2)  # 返回包含两个结果的元组
  v0 = z[0]  # 通过索引获取第一个结果
  ```



