## 三层神经网络分类器

### 文件结构

```python
Three Layer Net (按推荐代码阅读顺序[从上往下])
├──__init__.py              # 导入相关模块的文件
├──activation_func.py       # 激活函数
├──linear_component.py      # 线性（+激活）层
├──loss.py                  # 损失函数
├──full_connect_network.py  # N层神经网络分类器
├──optimization.py          # 优化器
├──solver.py                # 求解器，包括训练、预测、保存模型、导入模型
├──utils.py                 # 可视化神经网络相关函数
├──main.py                  # 主函数，定义parser
└──data                     # 数据集
    ├──train-images-idx3-ubyte.gz   # 训练图片
    ├──train-labels-idx1-ubyte.gz   # 训练标签
    ├──t10k-images-idx3-ubyte.gz    # 测试图片
    └──t10k-labels-idx1-ubyte.gz    # 测试标签
└──readme.md                # readme文件
```