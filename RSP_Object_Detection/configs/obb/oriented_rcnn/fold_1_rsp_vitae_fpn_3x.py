_base_ = [
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

dataset_type = 'DOTADataset'
data_root = '/home/ahmadreza/Downloads/Research/M2SODAI/baseline_official/fold_1_obb/dota/'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1  # ship only
        )
    ),
    pretrained='pretrained/rsp_vitaev2_s_e300.pth'
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelTxt/',
        img_prefix=data_root + 'train/images/',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/labelTxt/',
        img_prefix=data_root + 'val/images/',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/labelTxt/',
        img_prefix=data_root + 'val/images/',
    )
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
evaluation = dict(interval=1, metric='mAP')
