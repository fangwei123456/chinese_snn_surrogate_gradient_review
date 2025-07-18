from os.path import join
import sys
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn.functional as F

from datasets.gen1_od_dataset import GEN1DetectionDataset
from object_detection_module import DetectionLitModule
from lightning.pytorch.loggers import TensorBoardLogger

from numpy import random

import copy

_seed_ = 2025
import random
random.seed(2025)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def horizontal_flip_boxes(boxes, width):
    """
    Performs a horizontal flip of the bounding box

    :param boxes: Bounding box with shape [number_boxes, 4] ([xmin, ymin, xmax, ymax])
    :param width: Width of the image
    :return: Bounding box after flipping
    """
    boxes_flipped = boxes.clone()
    boxes_flipped[:, 0] = width - boxes[:, 2]
    boxes_flipped[:, 2] = width - boxes[:, 0]
    return boxes_flipped


def augmentation_collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)
    batch_size, num_steps, channels, height, width = samples.size()
    targets = [item[1] for item in batch]

    augmented_samples = []
    augmented_targets = []

    PATIENCE = 2
    patience = PATIENCE
    i = 0
    while i < batch_size:
        transform_t = torch.eye(3)  # 3x3unit matrix
        flip = False
        if random.random() > 0.5:
            flip = True
            transform_t[0, 0] *= -1

        transform_t_single = transform_t[:2, :].unsqueeze(0).repeat(num_steps, 1, 1).to(torch.float32)
        affine_t = F.affine_grid(transform_t_single.view(-1, 2, 3), [num_steps, channels, height, width],
                                 align_corners=False)

        sample_augmented = F.grid_sample(samples[i], affine_t, padding_mode='border', align_corners=False)
        augmented_samples.append(sample_augmented)

        real_boxes = targets[i]['boxes'].clone()
        if flip:
            targets[i]['boxes'] = horizontal_flip_boxes(targets[i]['boxes'], width)

        augmented_targets.append(copy.deepcopy(targets[i]))
        targets[i]['boxes'] = real_boxes

        if targets[i]['labels'].sum() > 0:
            patience -= 1
            if patience != 0:
                i -= 1
            else:
                patience = PATIENCE

        i += 1

    assert len(augmented_samples) == len(augmented_targets), "length is wrong"
    augmented_samples = torch.stack(augmented_samples, 0)  # Stack them together
    return [augmented_samples, augmented_targets]


def collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)

    targets = [item[1] for item in batch]
    return [samples, targets]

'''
python object_detection.py -device 0 -num_workers 4 -test -save_ckpt -backbone densenet-121_16 -b 16 -fusion_layers 3 -mode res -sn lif -norm bn -pretrained_backbone /home/wfang/SFOD/pt_classification/lif_bn/ckpt-ncars-densenet-121_16-val/ncars-epoch=08-val_acc=0.9169.ckpt

python object_detection.py -device 1 -num_workers 4 -test -save_ckpt -backbone densenet-121_16 -b 16 -fusion_layers 3 -mode res -sn if -norm bn -pretrained_backbone /home/wfang/SFOD/pt_classification/if_bn/ckpt-ncars-densenet-121_16-val/ncars-epoch=03-val_acc=0.8100.ckpt

python object_detection.py -device 2 -num_workers 4 -test -save_ckpt -backbone densenet-121_16 -b 16 -fusion_layers 3 -mode res -sn lif -norm tebn -pretrained_backbone /home/wfang/SFOD/pt_classification/lif_tebn/ckpt-ncars-densenet-121_16-val/ncars-epoch=09-val_acc=0.9096.ckpt

python object_detection.py -device 3 -num_workers 4 -test -save_ckpt -backbone densenet-121_16 -b 16 -fusion_layers 3 -mode res -sn spsn3 -norm bn -pretrained_backbone /home/wfang/SFOD/pt_classification/spsn3_bn/ckpt-ncars-densenet-121_16-val/ncars-epoch=12-val_acc=0.9141.ckpt

python object_detection.py -device 2 -num_workers 4 -test -save_ckpt -backbone densenet-121_16 -b 16 -fusion_layers 3 -mode res -sn clif -norm bn -pretrained_backbone /home/wfang/SFOD/pt_classification/clif_bn/ckpt-ncars-densenet-121_16-val/ncars-epoch=06-val_acc=0.8534.ckpt -no_train -pretrained /home/wfang/SFOD/pt_detection/clif_bn/ckpt-od-gen1-densenet-121_16-train/gen1-epoch=49-train_loss=6.4587.ckpt -lr 1e-4 -fusion

python object_detection.py -device 1 -num_workers 4 -test -save_ckpt -backbone densenet-121_16 -b 16 -fusion_layers 3 -mode res -sn blockalif -norm bn -pretrained_backbone /home/wfang/SFOD/pt_classification/blockalif_bn/ckpt-ncars-densenet-121_16-val/ncars-epoch=11-val_acc=0.6814.ckpt

python object_detection.py -device 1 -num_workers 4 -test -save_ckpt -backbone densenet-121_16 -b 16 -fusion_layers 3 -mode res -sn osr -norm osr -pretrained_backbone /home/wfang/SFOD/pt_classification/osr_osr/ckpt-ncars-densenet-121_16-val/ncars-epoch=01-val_acc=0.5623.ckpt -lr 1e-4 -fusion

'''
def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    # Dataset
    parser.add_argument('-dataset', default='gen1', type=str, help='dataset used {GEN1}')
    parser.add_argument('-path', default='/home/pkumdy/SNNDet/Gen/Gen', type=str,
                        help='path to dataset location')
    parser.add_argument('-num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('-device', default='0', type=int)

    # Data
    parser.add_argument('-b', default=4, type=int, help='batch size')#64
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(240, 304), type=tuple, help='spatial resolution of events')

    # Training
    parser.add_argument('-epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate used')
    parser.add_argument('-wd', default=1e-4, type=float, help='weight decay used')
    parser.add_argument('-limit_train_batches', default=1., type=float, help='train batches limit')
    parser.add_argument('-limit_val_batches', default=1., type=float, help='val batches limit')
    parser.add_argument('-limit_test_batches', default=1., type=float, help='test batches limit')
    parser.add_argument('-num_workers', default=2, type=int, help='number of workers for dataloaders')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')
    parser.add_argument('-precision', default='16-mixed', type=str, help='whether to use AMP {16, 32, 64}')
    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-early_stopping', action='store_true', help='early stopping')

    # Backbone
    parser.add_argument('-backbone', default='densenet-121_24', type=str,
                        help='model used {densenet-v}', dest='model')
    parser.add_argument('-norm', type=str)
    parser.add_argument('-sn', type=str)
    parser.add_argument('-pretrained_backbone', default=None, type=str, help='path to pretrained backbone model')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-extras', type=int, default=[640, 320, 320], nargs=3,
                        help='number of channels for extra layers after the backbone')#[640, 320, 320]
    parser.add_argument('-fusion', action='store_true', help='if to fusion the features')

    # Neck
    parser.add_argument('-fusion_layers', default=4, type=int, help='number of fusion layers')
    parser.add_argument('-mode', type=str, default='norm', help='The mode of detection_pyramid')

    # Priors
    parser.add_argument('-min_ratio', default=0.05, type=float, help='min ratio for priors\' box generation')
    parser.add_argument('-max_ratio', default=0.80, type=float, help='max ratio for priors\' box generation')
    parser.add_argument('-aspect_ratios', default=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], type=int,  #[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
                        help='aspect ratios for priors\' box generation')

    # Loss parameters
    parser.add_argument('-box_coder_weights', default=[10.0, 10.0, 5.0, 5.0], type=float, nargs=4,
                        help='weights for the BoxCoder class')
    parser.add_argument('-iou_threshold', default=0.50, type=float,
                        help='intersection over union threshold for the SSDMatcher class')
    parser.add_argument('-score_thresh', default=0.01, type=float,
                        help='score threshold used for postprocessing the detections')
    parser.add_argument('-nms_thresh', default=0.45, type=float,
                        help='NMS threshold used for postprocessing the detections')
    parser.add_argument('-topk_candidates', default=200, type=int, help='number of best detections to keep before NMS')
    parser.add_argument('-detections_per_img', default=100, type=int,
                        help='number of best detections to keep after NMS')




    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-resume', default=None)


    args = parser.parse_args()
    print(args)

    torch.set_float32_matmul_precision('medium')

    if args.dataset == "gen1":
        dataset = GEN1DetectionDataset
        torch.multiprocessing.set_sharing_strategy('file_system')
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    module = DetectionLitModule(args)

    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
        # ckpt_path = join(f"ckpt-od-{args.dataset}-{args.model}-val", args.pretrained)
        ckpt_path = args.pretrained
        print('loaded pretrained model:', ckpt_path)
        # with torch.no_grad():
        #     print('before load')
        #     for i, p in enumerate(module.backbone.parameters()):
        #         print(i, p.mean(), p.std())
        #         if i == 3:
        #             break
        module = DetectionLitModule.load_from_checkpoint(ckpt_path, strict=True)
        # with torch.no_grad():
        #     print('after load')
        #     for i, p in enumerate(module.backbone.parameters()):
        #         print(i, p.mean(), p.std())
        #         if i == 3:
        #             exit()
    # print(module)
    callbacks = []
    if args.debug:
        default_root_dir = './debug_pt_detection/' + args.sn + '_' + args.norm

    else:
        default_root_dir = './pt_detection/' + args.sn + '_' + args.norm
    if args.fusion:
        default_root_dir += '_fusion'
    import os
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir)


    if args.save_ckpt:
        ckpt_callback_val = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join(default_root_dir, f"ckpt-od-{args.dataset}-{args.model}-val/"),
            filename=f"{args.dataset}" + "-{epoch:02d}-{val_loss:.4f}",
            save_top_k=5,
            mode='min',
        )
        ckpt_callback_train = ModelCheckpoint(
            monitor='train_loss',
            dirpath=os.path.join(default_root_dir, f"ckpt-od-{args.dataset}-{args.model}-train/"),
            filename=f"{args.dataset}" + "-{epoch:02d}-{train_loss:.4f}",
            save_top_k=-1,
            mode='min',
        )
        callbacks.append(ckpt_callback_val)
        callbacks.append(ckpt_callback_train)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
        )
        callbacks.append(early_stopping)


    trainer = pl.Trainer(
        default_root_dir=os.path.join(default_root_dir, f"ckpt-od-{args.dataset}-{args.model}-train/"),
        devices=[args.device], #args.devices,
        accelerator="gpu",
        logger=TensorBoardLogger(save_dir="./debug_tb_detection_logs/" if args.debug else "./tb_detection_logs/", name=args.sn + '_' + args.norm + ('_fusion' if args.fusion else '')),
        gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        check_val_every_n_epoch=1,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_true'
    )

    if args.sn == 'osr':
        from models import review_modules
        def osr_init_hook(m, input):
            for sm in m.modules():
                if isinstance(sm, (review_modules.OnlineLIFNode, review_modules.OSR)):
                    sm.init = True

        module.register_forward_pre_hook(osr_init_hook)



    if args.train:
        train_dataset = dataset(args, mode="train")
        val_dataset = dataset(args, mode="val")

        train_dataloader = DataLoader(train_dataset, batch_size=args.b, collate_fn=augmentation_collate_fn,
                                      num_workers=args.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers)


        trainer.fit(module, train_dataloader, val_dataloader, ckpt_path=args.resume)
    if args.test:
        import energy
        import torch.nn as nn

        def record_input_shape_hook(module, input, output):
            if hasattr(module, 'input_shape'):
                return
            else:
                module.input_shape = input[0][0:1].shape
                module.output_shape = output[0:1].shape
                # print(module, module.input_shape, module.output_shape)

        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(record_input_shape_hook)

        energy.set_record_spike(module)

        test_dataset = dataset(args, mode="test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.b, collate_fn=collate_fn,
                                     num_workers=args.num_workers)

        trainer.test(module, test_dataloader)

        energy.set_flops(module)

        print('sop=', (energy.get_sops(module)) / 1e6)





if __name__ == '__main__':
    main()
