import argparse
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model

from DataLoader_together import BatchLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def train(cfg, args):
    # torch.cuda.set_device(2)

    # Initialize the network
    model = build_detection_model(cfg)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    # model load weights
    # print(torch.load(cfg.MODEL.WEIGHT))
    # checkpoint = torch.load(cfg.MODEL.WEIGHT)['model']
    print("Load from checkpoint?", bool(args.from_checkpoint))
    if not bool(args.from_checkpoint):
        # path = '/data6/SRIP19_SelfDriving/bdd100k/trained_model/Outputs/model_final_apt.pth'
        path = args.model_root
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint, strict=False)
    else:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
    # model.train()

    print("Freeze faster rcnn?", bool(args.freeze))
    for i, child in enumerate(model.children()):
        # print(i)
        # print(child)
        if i < 3:
            for param in child.parameters():
                param.requires_grad = False
                # param.requires_grad = not (bool(args.freeze))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outdir = cfg.OUTPUT_DIR

    class_weights = [1, 2, 2, 2]
    w = torch.FloatTensor(class_weights).cuda()
    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    criterion2 = nn.BCEWithLogitsLoss().cuda()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(args.initLR), weight_decay=float(args.weight_decay))

    # Initialize DataLoader
    Dataset = BatchLoader(
        imageRoot=args.imageroot,
        gtRoot=args.gtroot,
        reasonRoot=args.reasonroot,
        cropSize=(args.imHeight, args.imWidth)
    )
    dataloader = DataLoader(Dataset, batch_size=int(args.batch_size), num_workers=0, shuffle=True)

    # lossArr = []
    # AccuracyArr = []

    for epoch in range(0, args.num_epoch):
        trainingLog = open(outdir + ('trainingLogTogether_{0}.txt'.format(epoch)), 'w')
        trainingLog.write(str(args))
        lossArr = []
        AccuracyArr = []
        AccSideArr = []
        for i, dataBatch in enumerate(dataloader):

            # Read data
            img_cpu = dataBatch['img']
            imBatch = img_cpu.to(device)

            target_cpu = dataBatch['target']
            targetBatch = target_cpu.to(device)
            # ori_img_cpu = dataBatch['ori_img']
            if cfg.MODEL.SIDE:
                reason_cpu = (dataBatch['reason']).type(torch.FloatTensor)
                reasonBatch = reason_cpu.to(device)

            optimizer.zero_grad()
            if cfg.MODEL.SIDE:
                pred, pred_reason = model(imBatch)
                # Joint loss
                loss1 = criterion(pred, targetBatch)
                loss2 = criterion2(pred_reason, reasonBatch)
                loss = loss1 + loss2
            else:
                pred = model(imBatch)
                loss = criterion(pred, targetBatch)

            # torch.cuda.empty_cache()
            # pred, selected_boxes = model(imBatch)
            # DrawBbox(ori_img_cpu[0], selected_boxes[0])
            # plt.clf()
            # plt.close()

            # print(pred)
            # print(targetBatch)

            loss.backward()
            optimizer.step()
            loss_cpu = loss.cpu().data.item()

            lossArr.append(loss_cpu)
            meanLoss = np.mean(np.array(lossArr))

            # Calculate accuracy
            predict = torch.sigmoid(pred) > 0.5

            f1 = f1_score(target_cpu.data.numpy(), predict.cpu().data.numpy(), average='samples')
            AccuracyArr.append(f1)
            meanAcc = np.mean(np.array(AccuracyArr))

            if cfg.MODEL.SIDE:
                predict_reason = torch.sigmoid(pred_reason) > 0.5
                f1_side = f1_score(reason_cpu.data.numpy(), predict_reason.cpu().data.numpy(), average='samples')
                AccSideArr.append(f1_side)

            if i % 50 == 0:
                print('prediction logits:', pred)

                print('ground truth:', targetBatch.cpu().data.numpy())
                print('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f' % (
                    epoch, i, loss_cpu, meanLoss))
                trainingLog.write('Epoch %d Iteration %d: Loss %.5f Accumulated Loss %.5f \n' % (
                    epoch, i, loss_cpu, meanLoss))
                print('Epoch %d Iteration %d Action Prediction: F1 %.5f Accumulated F1 %.5f' % (
                    epoch, i, AccuracyArr[-1], meanAcc))
                if cfg.MODEL.SIDE:
                    meanAccSide = np.mean(AccSideArr)
                    print('Epoch %d Iteration %d Side Task: F1 %.5f Accumulated F1 %.5f' % (
                        epoch, i, AccSideArr[-1], meanAccSide))

            if epoch in [int(0.4 * args.num_epoch), int(0.7 * args.num_epoch)] and i == 0:
                print('The learning rate is being decreased at Iteration %d', i)
                trainingLog.write('The learning rate is being decreased at Iteration %d \n' % i)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), (outdir + 'net_%d.pth' % (epoch + 1)))
        # if args.val and epoch % 10 == 0:
        #    print("Validation...")
        #    run_test(cfg, args)
    print("Saving final model...")
    torch.save(model.state_dict(), (outdir + 'net_Final.pth'))
    print("Done!")


def run_test(cfg, args):
    pass

def DrawBbox(img, boxlist):
    plt.imshow(img)
    currentAxis = plt.gca()
    for i in range(boxlist.shape[0]):
        bbox = boxlist[i]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect)

    plt.show()

def main():
    # Build a parser for arguments
    parser = argparse.ArgumentParser(description="Action Prediction Training")
    parser.add_argument(
        "--config-file",
        default="/home/SelfDriving/maskrcnn/maskrcnn-benchmark/configs/baseline.yaml",
        metavar="FILE",
        help="path to maskrcnn_benchmark config file",
        type=str,
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--initLR",
        help="Initial learning rate",
        default=0.0001
    )
    parser.add_argument(
        "--freeze",
        default=False,
        help="If freeze faster rcnn",
    )
    parser.add_argument(
        "--imageroot",
        type=str,
        help="Directory to the images",
        default="/data6/SRIP19_SelfDriving/bdd12k/data1/"
    )
    parser.add_argument(
        "--gtroot",
        type=str,
        help="Directory to the groundtruth",
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/12k_gt_train_5_actions.json"
    )
    parser.add_argument(
        "--reasonroot",
        type=str,
        help="Directory to the explanations",
        default="/data6/SRIP19_SelfDriving/bdd12k/annotations/train_reason_img.json"
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
    parser.add_argument(
        "--model_root",
        type=str,
        help="Directory to the trained model",
        default="/data6/SRIP19_SelfDriving/bdd100k/trained_model/Outputs/model_final_apt.pth"
    )
    parser.add_argument(
        "--val",
        action='store_true',
        default=False,
        help='Validation or not'
    )
    parser.add_argument(
        "--num_epoch",
        default=20,
        help="The number of epoch for training",
        type=int
    )
    parser.add_argument(
        "--from_checkpoint",
        default=False,
        help="If we need load weights from checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=".",
        help="The path to the checkpoint weights.",
        type=str,
    )

    args = parser.parse_args()
    print(args)
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


if __name__ == "__main__":
    main()
