### 继承文件

#### **resnet18.py**

https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/resnet18.py

```python
# model settings
model = dict(
    type='ImageClassifier',                       # 分类器类型
    backbone=dict(
        type='ResNet',                            # 主干网络类型
        depth=18,                                 # 主干网网络深度， ResNet 一般有18, 34, 50, 101, 152 可以选择
        num_stages=4,                             # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(3, ),                        # 输出的特征图输出索引。越远离输入图像，索引越大
        style='pytorch'                           # 网络微调时，冻结网络的stage（训练时不执行反相传播算法），若num_stages=4，backbone包含stem 与 4 个 stages。frozen_stages为-1时，不冻结网络； 为0时，冻结 stem； 为1时，冻结 stem 和 stage1；为4时，冻结整个backbone
    ),
                                 
    neck=dict(type='GlobalAveragePooling'),       # 颈网络类型
    head=dict(
        type='LinearClsHead',                     # 线性分类头，
        num_classes=1000,                         # 输出类别数，这与数据集的类别数一致
        in_channels=512,                          # 输入通道数，这与 neck 的输出通道一致
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  # 损失函数配置信息
        topk=(1, 5),                              # 评估指标，Top-k 准确率， topk=(1, 5)为 top1 与 top5 准确率
    )
)
```

#### **imagenet_bs32.py**

https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/datasets/imagenet_bs32.py

```python
# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),                 # 读取图片
    dict(type='RandomResizedCrop', size=224),       # 随机缩放抠图
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'), # 以概率为0.5随机水平翻转图片
    dict(type='Normalize', **img_norm_cfg),         # 归一化
    dict(type='ImageToTensor', keys=['img']),       # image 转为 torch.Tensor
    dict(type='ToTensor', keys=['gt_label']),       # gt_label 转为 torch.Tensor
    dict(type='Collect', keys=['img', 'gt_label'])  # 决定数据中哪些键应该传递给检测器的流程
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']) # test 时不传递 gt_label
]
data = dict(
    samples_per_gpu=32,                             # 单个 GPU 的 Batch size
    workers_per_gpu=2,                              # 单个 GPU 的 线程数
    train=dict(
        type=dataset_type,                          # 数据集名称
        data_prefix='data/imagenet/train',          # 数据集目录，当不存在 ann_file 时，类别信息从文件夹自动获取
        pipeline=train_pipeline),                   # 数据集需要经过的 数据流水线
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',      # 标注文件路径，存在 ann_file 时，不通过文件夹自动获取类别信息
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
```

#### **default_runtime.py**

https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/default_runtime.py

```python
# checkpoint saving
checkpoint_config = dict(interval=1)     # 保存的间隔是 1，单位会根据 runner 不同变动，可以为 epoch 或者 iter。
# yapf:disable
log_config = dict(
    interval=100,                        # 打印日志的间隔， 单位 iters
    hooks=[
        dict(type='TextLoggerHook'),     # 用于记录训练过程的文本记录器(logger)。
        # dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ])
# yapf:enable

dist_params = dict(backend='nccl')       # 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'                             # 日志的输出级别
load_from = None                         # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
resume_from = None
workflow = [('train', 1)]                # runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次

```



### 