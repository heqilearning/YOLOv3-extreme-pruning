# YOLOv3-extreme-pruning

这是在[Lam1360/YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)工作基础上的一个改版，实现了所有卷积层参与剪枝。<br>
YOLOv3有23处残差连接，由于通道剪枝对下一层的输入有相应影响，shortcut接入的两个层如果发生了通道剪枝，通道数发生改变，剪的位置也不尽相同，在连接相加时并不好处理，原作没有对这些层进行剪枝，但也实现了较高剪枝率，有效降低了参数量。<br>
而这个改版尝试让所有卷积层参与剪枝，为此在shortcut处添加了一个零值基板，以保持不同通道的对应连接，这个做法可能是低效的，导致shortcut层计算量增加，特别是剪枝率低的时候；剪枝率提高，这种代价会降低，而且带来卷积层更多的参数量和计算量下降，85%的剪枝率能够去掉98%的参数。除了输入层的输入维度，输出层的输出维度不变，其他卷积层都参与了卷积。<br>
[PengyiZhang/SlimYOLOv3](https://github.com/PengyiZhang/SlimYOLOv3)是一种折中的做法，同一个残差块不同残差连接进行同步剪枝，保证通道一致。<br>


## 训练和剪枝
训练步骤可以参考[Lam1360/YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)，注意在cfg文件中新增了'old_filters'和'remain'项来记录剪枝前后的状态，因为shortcut的计算被重新定义了，剪枝后的模型需要保证不同通道的连接。<br>
工作的重点仍然是稀疏训练，通过训练策略的调整让模型达到较高稀疏度并保持较高mAP值，之后再进行剪枝和finetune。这个改版也支持迭代式剪枝。
