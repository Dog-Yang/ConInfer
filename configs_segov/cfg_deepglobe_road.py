_base_ = './base_config3.py'

# model settings
model = dict(
    name_path='./prompts/cls_deepglobe_road.txt',
    prob_thd=0.3,
    dataset_type = 'RoadValDataset'
)

# dataset settings
dataset_type = 'RoadValDataset'
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
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/GlobalRoadSet_Val/DeepGlobe_test_1530/image_cvt',
            seg_map_path='data/GlobalRoadSet_Val/DeepGlobe_test_1530/label_cvt'),
        pipeline=test_pipeline))