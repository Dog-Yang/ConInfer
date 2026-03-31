_base_ = './base_config3.py'

# model settings
model = dict(
    name_path='./prompts/cls_wbs-si.txt',
    prob_thd=0.5,
    dataset_type = 'WaterDataset'
)

# dataset settings
dataset_type = 'WaterDataset'
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
    batch_size=1,
    num_workers=32,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='tools/dataset_converters/wbs-si_val.txt',
        data_prefix=dict(
            img_path='data/water_body_segmentation/WaterBodiesDatasetPreprocessed/Images',
            seg_map_path='data/water_body_segmentation/WaterBodiesDatasetPreprocessed/Masks_cvt'),
        pipeline=test_pipeline))