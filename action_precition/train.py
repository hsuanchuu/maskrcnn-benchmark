# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:08:43 2019

@author: epyir
"""
import argparse
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list

from baseline import baseline
from dataloader import BatchLoader

import torch.optim as optim

def train(cfg, args):
    detector = build_detection_model(cfg)
    # print(detector)
    detector.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    detector.to(device)
    outdir = cfg.OUTPUT_DIR
    
    # Initialize the network
    model = baseline()
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.initLR, momentum=0.9, weight_decay=5e-4)
    
    # Initialize image batch
    # imBatch = Variable(torch.FloatTensor(args.batch_size, 3, args.imHeight, args.imWidth))
    imBatch = Variable(torch.FloatTensor(args.batch_size, 3, 736, 1280))
    targetBatch = Variable(torch.LongTensor(args.batch_size, 1))
    
    # Move network and batch to gpu
    imBatch = imBatch.cuda(device)
    targetBatch = targetBatch.cuda(device)
    model = model.cuda(device)
    
    # Initialize dataloader
    Dataset = BatchLoader(
            imageRoot = args.imageroot,
            gtRoot = args.gtroot,
            cropSize = (args.imWidth, args.imHeight)
            )
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    
    lossArr = []
    AccuracyArr = []
    accuracy = 0
    iteration = 0
    
    for epoch in range(0, 10):
        trainingLog = open(outdir + ('trainingLog_{0}.txt'.format(epoch)), 'w')
        for i, dataBatch in enumerate(dataloader):
            iteration += 1
            
            # Read data, under construction
            img_cpu = dataBatch['img']
            # if args.batch_size == 1:
            #     img_list = to_image_list(img_cpu[0,:,:], cfg.DATALOADER.SIZE_DIVISIBILITY)
            # else:
            #     img_list = to_image_list(img_cpu, cfg.DATALOADER.SIZE_DIVISIBILITY)
            img_list = to_image_list(img_cpu[0,:,:], cfg.DATALOADER.SIZE_DIVISIBILITY)
            imBatch.data.copy_(img_list.tensors) # Tensor.shape(BatchSize, 3, Height, Width)

            
            target_cpu = dataBatch['target']
            # print(target_cpu)
            targetBatch.data.copy_(target_cpu)
            
            # Train network
            RoIPool_module = detector.roi_heads.box.feature_extractor.pooler
            Backbone = detector.backbone
            hook_roi = SimpleHook(RoIPool_module)
            hook_backbone = SimpleHook(Backbone)
            out_detector = detector(imBatch)
            features_roi = hook_roi.output.data
            features_backbone = hook_backbone.output[0].data # only use the bottom one
            # features = detector(imBatch) # features extracted by ROI pooling/align
            optimizer.zero_grad()
            
            # pred = model(features_roi, features_backbone)
            pred = model(features_roi, features_backbone)
            print('prediction:', pred)
            # print('target:', targetBatch[0,:][0])
            loss = criterion(pred, targetBatch[0,:])
            action = pred.cpu().argmax().data.numpy()

            print('predicted action:', action)
            print('ground truth:', target_cpu.data.numpy()[0])

            loss.backward()
            
            optimizer.step()
            if action == target_cpu.data.numpy()[0]:
                accuracy += 1

            lossArr.append(loss.cpu().data.item())
            AccuracyArr.append(accuracy/iteration)

            meanLoss = np.mean(np.array(lossArr))

            print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (epoch, iteration, lossArr[-1], meanLoss ))
            trainingLog.write('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (epoch, iteration, lossArr[-1], meanLoss ))

            print('Epoch %d Iteration %d: Accumulated Accuracy %.5f' % (epoch, iteration, AccuracyArr[-1]))
            trainingLog.write('Epoch %d Iteration %d: Accumulated Accuracy %.5f' % (epoch, iteration, AccuracyArr[-1]))
            
            
            if iteration in [20000,40000]:
                print('The learning rate is being decreased at Iteration %d', iteration)
                trainingLog.write('The learning rate is being decreased at Iteration %d', iteration)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
                    
            if iteration == args.MaxIteration:
                torch.save(model.state_dict(), 'netFinal_%d.pth' % (epoch+1))
                break
        if iteration >= args.MaxItertion:
            break
        
        if (epoch+1) % 2 == 0:
            torch.save(model.state_dict(), 'netFinal_%d.pth' % (epoch+1))
            
class SimpleHook(object):
    """
    A simple hook function to extract features.
    :return:
    """
    def __init__(self, module, backward=False):
        # super(SimpleHook, self).__init__()
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input_, output_):
        self.input = input_
        self.output = output_

    def close(self):
        self.hook.remove()


def run_test():
    pass

def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_101_FPN_1x.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--MaxIteration",        
        help="the iteration to end training",
        type=int,
        default=90000,      
    )
    parser.add_argument(
        "--initLR",
        help="Initial learning rate",
        default=0.001
    )
    parser.add_argument(
        "--imageroot",
        type=str,
        help="Directory of the images",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/images/100k/train"
    )
    parser.add_argument(
        "--gtroot",
        type=str,
        help="Directory of the groundtruth",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/annotations/train_gt_action.json"
    )
    parser.add_argument(
        "--imWidth",
        type=int,
        help="Crop to width",
        default=1280
    )
    parser.add_argument(
        "--imHeight",
        type=int,
        help="Crop to height",
        default=720
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Size",
        default=1
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Give this experiment a name",
        default=str(datetime.datetime.now())
    )
    
    
    args = parser.parse_args()
    print(args)
    cfg.MODEL.WEIGHT = "/data6/SRIP_SelfDriving/Outputs/model_final.pth"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    if torch.cuda.is_available():
        print("CUDA device is available.")
    
    # output directory
    outdir = cfg.OUTPUT_DIR
    print("Save path:", outdir)
    if outdir:
        mkdir(outdir)
    
#    logger = setup_logger("training", outdir)
    
    train(cfg, args)
    
    # if validate
    if not args.skip_test:
        run_test()

if __name__ == "__main__":
    main()