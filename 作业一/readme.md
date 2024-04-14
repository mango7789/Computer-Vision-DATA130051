## 三层神经网络分类器

### 文件结构

```python
Three Layer Net
├──__init__.py              # 导入相关模块的文件
├──linear_component.py      # 线性（+激活）层
├──activation.py            # 激活函数
├──loss.py                  # 损失函数
├──optimization.py          # 优化器
├──solver.py                # 求解器，包括训练、预测、保存模型、导入模型
├──full_connect_network.py  # N层神经网络分类器
├──utils.py                 # 可视化神经网络
├──main.py                  # 主函数，定义parser
└──readme.md                # readme
```