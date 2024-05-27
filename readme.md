
## 作业一

> [!IMPORTANT]
> 手工搭建三层神经网络分类器，在数据集[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)上进行训练以实现图像分类。

### 基本要求：

- 本次作业要求自主实现反向传播，不允许使用pytorch，tensorflow等现成的支持自动微分的深度学习框架，可以使用numpy；
- 最终提交的代码中应至少包含模型、训练、测试和参数查找四个部分，鼓励进行模块化设计；
- 其中模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；训练部分应实现SGD优化器、学习率下降、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重；参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能；测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。

### 提交要求：

- 仅提交pdf格式的实验报告，报告中除对模型、数据集和实验结果的基本介绍外，还应可视化训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy曲线；
- 报告中需包含对训练好的模型网络参数的可视化，并观察其中的模式；
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重的下载地址。


## 期中作业

> [!IMPORTANT]
> 任务1：微调在ImageNet上预训练的卷积神经网络实现鸟类识别

### 基本要求：
- 修改现有的CNN架构（如AlexNet，ResNet-18）用于鸟类识别，通过将其输出层大小设置为200以适应数据集中的类别数量，其余层使用在ImageNet上预训练得到的网络参数进行初始化；
- 在[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158)数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调；
- 观察不同的超参数，如训练步数、学习率，及其不同组合带来的影响，并尽可能提升模型性能；
- 与仅使用CUB-200-2011数据集从随机初始化的网络参数开始训练得到的结果进行对比，观察预训练带来的提升。

### 提交要求：
- 仅提交pdf格式的实验报告，报告中除对模型、数据集和实验结果的基本介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy变化；
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重的下载地址。

> [!IMPORTANT]
> 任务2：在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3

### 基本要求：
- 学习使用现成的目标检测框架——如mmdetection或detectron2——在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3；
- 挑选4张测试集中的图像，通过可视化对比训练好的Faster R-CNN第一阶段产生的proposal box和最终的预测结果。
- 搜集三张不在VOC数据集内包含有VOC中类别物体的图像，分别可视化并比较两个在VOC数据集上训练好的模型在这三张图片上的检测结果（展示bounding box、类别标签和得分）；


### 提交要求：
- 仅提交pdf格式的实验报告，报告中除对模型、数据集和实验结果的介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的mAP曲线；
- 报告中应提供详细的实验设置，如训练测试集划分、网络结构、batch size、learning rate、优化器、iteration、epoch、loss function、评价指标等。
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重的下载地址。


## 期末作业

> [!IMPORTANT]
> 任务1：对比监督学习和自监督学习在图像分类任务上的性能表现

### 基本要求：
- 实现任一自监督学习算法并使用该算法在自选的数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；
- 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；
- 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；

### 提交要求：
- 提交pdf格式的实验报告，报告中除对模型、数据集和实验结果的基本介绍外，还应包含用Tensorboard可视化的训练过程中的loss曲线变化以及Linear classification过程中accuracy的变化；
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重的下载地址。

> [!IMPORTANT]
> 任务2：在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型

### 基本要求：
- 分别基于CNN和Transformer架构实现具有相近参数量的图像分类网络；
- 在CIFAR-100数据集上采用相同的训练策略对二者进行训练，其中数据增强策略中应包含CutMix；
- 尝试不同的超参数组合，尽可能提升各架构在CIFAR-100上的性能以进行合理的比较。

### 提交要求：
- 提交pdf格式的实验报告，报告中除对模型、数据集和实验结果的介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的Accuracy曲线；
- 报告中应提供详细的实验设置，如训练测试集划分、网络结构、batch size、learning rate、优化器、iteration、epoch、loss function、评价指标等。
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重的下载地址。

> [!IMPORTANT]
> 任务3：基于NeRF的物体重建和新视图合成

### 基本要求：
- 选取身边的物体拍摄多角度图片/视频，并使用COLMAP估计相机参数，随后使用现成的框架进行训练；
- 基于训练好的NeRF渲染环绕物体的视频，并在预留的测试图片上评价定量结果。
  
### 提交要求：
- 提交pdf格式的实验报告，报告中除对模型、数据和实验结果的介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和测试集上的loss曲线，以及在测试集上的PSNR等指标；
- 报告中应提供详细的实验设置，如训练测试集划分、网络结构、batch size、learning rate、优化器、iteration、epoch、loss function、评价指标等。
- 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重和渲染的视频上传到百度云/google drive等网盘，实验报告内应包含实验代码所在的github repo链接及模型权重和视频的下载地址。