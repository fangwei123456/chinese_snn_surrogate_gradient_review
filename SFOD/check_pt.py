import torch


a = '/home/wfang/SFOD/debug_pt_detection/lif_bn/ckpt-od-gen1-densenet-121_16-train/gen1-epoch=00-train_loss=5.0255.ckpt'
b = '/home/wfang/SFOD/pt_detection/lif_bn/ckpt-od-gen1-densenet-121_16-train/gen1-epoch=49-train_loss=1.1350.ckpt'


a = torch.load(a, map_location='cpu')
b = torch.load(b, map_location='cpu')
'''
dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'MixedPrecisionPlugin', 'hparams_name', 'hyper_parameters'])
'''
'''
{'args': Namespace(dataset='gen1', path='/home/pkumdy/SNNDet/Gen/Gen', num_classes=2, device=1, b=16, sample_size=100000, T=5, tbin=2, image_shape=(240, 304), epochs=1, lr=0.001, wd=0.0001, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, num_workers=4, train=True, test=True, precision='16-mixed', save_ckpt=True, early_stopping=False, model='densenet-121_16', norm='bn', sn='lif', pretrained_backbone='/home/wfang/SFOD/pt_classification/lif_bn/ckpt-ncars-densenet-121_16-val/ncars-epoch=08-val_acc=0.9169.ckpt', pretrained=None, extras=[640, 320, 320], fusion=False, fusion_layers=3, mode='res', min_ratio=0.05, max_ratio=0.8, aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], box_coder_weights=[10.0, 10.0, 5.0, 5.0], iou_threshold=0.5, score_thresh=0.01, nms_thresh=0.45, topk_candidates=200, detections_per_img=100, sop=False, debug=True)}
{'args': Namespace(dataset='gen1', path='/home/pkumdy/SNNDet/Gen/Gen', num_classes=2, device=0, b=16, sample_size=100000, T=5, tbin=2, image_shape=(240, 304), epochs=50, lr=0.001, wd=0.0001, limit_train_batches=1.0, limit_val_batches=1.0, limit_test_batches=1.0, num_workers=4, train=True, test=True, precision='16-mixed', save_ckpt=True, early_stopping=False, model='densenet-121_16', norm='bn', sn='lif', pretrained_backbone='/home/wfang/SFOD/pt_classification/lif_bn/ckpt-ncars-densenet-121_16-val/ncars-epoch=08-val_acc=0.9169.ckpt', pretrained=None, extras=[640, 320, 320], fusion=False, fusion_layers=3, mode='res', min_ratio=0.05, max_ratio=0.8, aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], box_coder_weights=[10.0, 10.0, 5.0, 5.0], iou_threshold=0.5, score_thresh=0.01, nms_thresh=0.45, topk_candidates=200, detections_per_img=100)}

'''