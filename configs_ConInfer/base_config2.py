# base configurations
model = dict(
    type='ConInferSegmentation',
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    # model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
    # ignore_residual=True,    # 'True', 'Flase'
    
    # maskclip
    # model_type='MaskCLIP',
    # feature_up=False,
    # cls_token_lambda=0,
    # ignore_residual=False,

    # clearclip
    # model_type='ClearCLIP',
    # feature_up=False,
    # cls_token_lambda=0,
    # ignore_residual=True,

    # SCLIP
    # model_type='SCLIP',
    # feature_up=False,
    # cls_token_lambda=0,
    # ignore_residual=False,

    # # GEM
    # model_type='GEM',
    # feature_up=False,
    # cls_token_lambda=0,
    # ignore_residual=False,

    # ConInfer
    model_type='SegEarth',
    feature_up=False,
    cls_token_lambda=0,
    ignore_residual=True,
    attn_lambda=0.6,

    # # MaskCLIP + ConInfer
    # model_type='MaskCLIP',
    # feature_up=False,
    # cls_token_lambda=0,
    # ignore_residual=False,
    # attn_lambda=0.0,

    # # SCLIP + ConInfer
    # model_type='SCLIP',
    # feature_up=False,
    # cls_token_lambda=0,
    # ignore_residual=False,
    # attn_lambda=0.0,

    # # ClearCLIP + ConInfer
    # model_type='ClearCLIP',
    # feature_up=False,
    # cls_token_lambda=0,
    # ignore_residual=True,
    # attn_lambda=0.0,

    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),

    slide_stride=0,
    slide_crop=0,

    logit_scale=5,
    gmm_temp=50,
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.5, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))