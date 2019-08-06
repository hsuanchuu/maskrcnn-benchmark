# This file is intended to flip the image with action "stop" to balance the data
import numpy as np
import matplotlib.pyplot as plt
import json

imageroot = "/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/images/100k/train/"
gtroot = "/home/SelfDriving/maskrcnn/maskrcnn-benchmark/datasets/bdd100k/annotations/train_gt_action.json"
with open(gtroot, 'r') as json_file:
    data = json.load(json_file)

labels = data['annotations']
imgs = data['images']
for i, label in enumerate(labels):
    filename = imgs[i]['file_name']
    image = plt.imread(imageroot + filename)
    image_flipped = image[:,::-1,:]
    plt.imsave(imageroot+filename)