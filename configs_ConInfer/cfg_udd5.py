_base_ = './base_config2.py'

# model settings
model = dict(
    name_path='./prompts/cls_udd5.txt',
    prob_thd=0.6,
    bg_idx=4,
    dataset_type = 'UDD5Dataset',
)

# dataset settings
dataset_type = 'UDD5Dataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=50,
    num_workers=32,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/UDD5/val/src',
            seg_map_path='data/UDD5/val/gt'),
        pipeline=test_pipeline))