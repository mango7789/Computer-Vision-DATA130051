_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# set the number of output classes of this model
model = dict(
    roi_head = dict(
        bbox_head = dict(
            num_classes = 20
        )
    )
)

# set the path of the dataset
data = dict(
    train = dict(
        type = 'VOCDataset',
        ann_file = 'data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
        img_prefix = 'data/VOCdevkit/VOC2007'
    ),
    val = dict(
        type = 'VOCDataset',
        ann_file = 'data/VOCdevkit/VOC2007/ImageSets/Main/val.txt',
        img_prefix = 'data/VOCdevkit/VOC2007'
    ),
    test = dict(
        type = 'VOCDataset',
        ann_file = 'data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix = 'data/VOCdevkit/VOC2007'
    ),
)

# set hyper-parameters
optimizer = dict(
    type = 'SGD',
    lr = 0.001,
    momentum = 0.9,
    weight_decay = 0.0005
)

lr_config = dict(
    policy = 'step',
    step = [8_000, 100_000],
    gamma = 0.9
)

runner = dict(
    type = 'EpochBasedRunner',
    max_epochs = 20
)