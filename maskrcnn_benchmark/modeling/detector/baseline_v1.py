"""
Baseline v1.
"""

import torch
import logging
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

# from maskrcnn_benchmark.modeling.poolers import Pooler
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from test_all import DrawBbox
# import numpy as np

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads_v1 import build_roi_heads



# def DrawBbox(img, boxlist):
#     plt.imshow(img)
#     currentAxis = plt.gca()
#     # folder = '/data6/SRIP19_SelfDriving/bdd12k/Outputs/'
#     for i in range(boxlist.shape[0]):
#         bbox = boxlist[i]
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
#         # rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='r', facecolor='none')
#         currentAxis.add_patch(rect)
#
#     plt.show()


class Baseline(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(Baseline, self).__init__()
        self.side = cfg.MODEL.SIDE
        print("side:{}".format(self.side))

        self.training = False
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        if int(cfg.MODEL.PREDICTOR_NUM) == 1:
            from .predictor2_2 import build_predictor
        else:
            from .predictor2 import build_predictor
        self.predictor = build_predictor(cfg, side=self.side)

    def forward(self, images, targets=None, get_feature=True): #TODO: Remember to set get_feature as False if there are any changes later
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # self.count += 1
        images = to_image_list(images)
        features = self.backbone(images.tensors) # global features
        
        # print(features[0].shape)
        # print(type(features))
        
        logger = logging.getLogger("tmp")
        logger.info(len(features))
        proposals,_ = self.rpn(images, features, targets)
        # print(proposals)
        b_size = len(proposals)
        bbox_num = [proposals[i].__len__() for i in range(b_size)]
        # print(bbox_num)
        if self.roi_heads:

            x, result, _ = self.roi_heads(features, proposals, targets)
            result, idx = result # result is a list of box, idx is a list of index, length is the same as the batch size.
            # print(result)
            # print(idx[0].shape, idx[1].shape)
            xx = []
            tmp = 0
            for i in range(b_size):
                xx.append(x[tmp: tmp + bbox_num[i]][idx[i]])# list of (N, 2048, 7, 7), N is the number of detected boxes.
                tmp = bbox_num[i]

            del x, tmp, bbox_num
            # print(xx[0].shape)
            preds = []
            preds_reason = []
            # selected_boxes=[]
            for i in range(b_size):
                results = {'roi_features':xx[i],
                           'glob_feature':features[0][i].unsqueeze(0), }

                hook_selector = SimpleHook(self.predictor.selector)
                if self.side:
                    pred, pred_reason = self.predictor(results)
                else:
                    pred = self.predictor(results)

                # scores = hook_selector.output.data
                # hook_selector.close()
                # scores, idx = torch.sort(scores, dim=0, descending=True)
                # idx = idx.reshape(-1,)
                # selected_boxes.append(result[i].bbox)
                # selected_boxes.append(result[i].bbox[idx[:15]])

                preds.append(pred)
                if self.side:
                    preds_reason.append(pred_reason)
            # print(torch.stack(preds,dim=0).squeeze(1))
            return (torch.stack(preds,dim=0).squeeze(1), torch.stack(preds_reason,dim=0).squeeze(1)) if self.side else torch.stack(preds,dim=0).squeeze(1)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        return result

# Self-made hook
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
