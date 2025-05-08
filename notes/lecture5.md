# 自动微分实现

## 大纲
- 是needle操作的一些介绍，也是完成HW1作业之后的检测部分

## autograd.py代码解析

### 核心概念
- 计算图：由操作（Op）和数据节点（Value）组成的有向无环图（DAG），用于描述计算过程。
- Tensor：表示多维数组，是计算图中的数据节点，支持梯度计算。
- Op：定义前向计算和反向梯度传播的规则。

### 关键类

#### `Op` 类
- 作用：所有操作的基类（如加法、乘法）。
- 方法：
  - `__call__`: 调用操作，生成新的 `Value`（如 `Tensor`）。
  - `compute`: 实现前向计算（基于 NumPy）。
  - `gradient`: 计算反向传播的梯度（需子类实现）。
  
```py
class Op:
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)  # 创建新 Tensor
    def compute(self, *args): ...  # 前向计算
    def gradient(self, out_grad, node): ...  # 反向梯度
```

#### `Value` 类
- 作用：计算图中的节点，存储数据和操作信息。
- 属性：
  - `op`: 生成该值的操作。
  - `inputs`: 输入节点列表。
  - `cached_data`: 缓存的实际数据（如 NumPy 数组）。
  - `requires_grad`: 是否需要计算梯度。
- 方法：
  - `realize_cached_data`: 执行计算并缓存结果（惰性计算）。
  - `is_leaf`: 判断是否为叶子节点（无 `op`）。

```py
class Value:
    def realize_cached_data(self):
        if self.cached_data is None:
            # 递归计算所有输入的数据
            self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data
```

#### `Tensor` 类
- 继承自 `Value`，表示多维数组。
- 操作符重载：如 `+`、`*`、`@`（矩阵乘法）等，触发对应 Op。
- 关键方法：
  - `backward`: 启动反向传播，计算梯度。
  - `detach`: 创建脱离计算图的新 Tensor。
  - `numpy`: 将数据转换为 NumPy 数组。

```py
class Tensor(Value):
    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)  # 调用加法操作
        else:
            return needle.ops.AddScalar(other)(self)    # 标量加法
    def backward(self, out_grad=None):
        compute_gradient_of_variables(self, out_grad)   # 反向传播
```

---

### 自动微分流程

#### 前向传播
- 示例：`c = a + b`
  1. 调用 `EWiseAdd` 操作的 `__call__` 方法。
  2. 创建新 Tensor，其 `op` 为 `EWiseAdd`，`inputs` 为 `[a, b]`。
  3. 实际计算（如 `a + b`）在 `realize_cached_data` 时执行。

#### 反向传播
- 链式法则：梯度从输出向输入传播。
- 关键函数：
  - `compute_gradient_of_variables`: 遍历计算图，累积梯度。
  - `find_topo_sort`: 生成拓扑排序，确保按正确顺序计算。

```py
def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads = {output_tensor: [out_grad]}
    reverse_topo_order = reversed(find_topo_sort([output_tensor]))
    for node in reverse_topo_order:
        grad = sum_grads(node_to_output_grads[node])
        if node.op:
            input_grads = node.op.gradient(grad, node)
            for input, input_grad in zip(node.inputs, input_grads):
                node_to_output_grads[input].append(input_grad)
    # 将梯度存入 Tensor.grad
```

---

### 待实现部分

#### 拓扑排序 (`find_topo_sort`)
- 目标：按计算依赖关系后序遍历节点
- 伪代码：
  ```py
  def find_topo_sort(nodes):
      visited = set()
      topo_order = []
      for node in nodes:
          topo_sort_dfs(node, visited, topo_order)
      return topo_order

  def topo_sort_dfs(node, visited, topo_order):
      if node in visited:
          return
      visited.add(node)
      for input in node.inputs:
          topo_sort_dfs(input, visited, topo_order)
      topo_order.append(node)
  ```

#### 梯度计算 (`compute_gradient_of_variables`)
- 步骤：
  1. 初始化输出梯度。
  2. 按拓扑逆序遍历节点。
  3. 对每个节点，计算其输入的梯度并累积。


### 扩展性
- 通过替换 `backend_numpy`，可支持其他计算后端（如 GPU）。