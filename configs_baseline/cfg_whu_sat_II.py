_base_ = './base_config1.py'

# model settings
model = dict(
    name_path='./prompts/cls_whu_sat.txt',
    prob_thd=0.1,
    dataset_type = 'WHUDataset'
)

# dataset settings
dataset_type = 'WHUDataset'
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
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        data_prefix=dict(
            img_path='data/WHU_Sat_II/test/image', 
            seg_map_path='data/WHU_Sat_II/test/label_cvt'),
        pipeline=test_pipeline))