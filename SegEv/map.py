# -*- coding:utf-8 -*-
import window
from eval import use_char
import cv2
import torch
from utils.augmentations import BaseTransform, FastBaseTransform
import datetime
import glob
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from minisegnet import Yolact, FPN
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from pytorch_grad_cam import GradCAM,EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

global image_paths
def Heat_Map1(image_path,weights_path = None,save_name = None):
    image = cv2.imread(image_path)
    rgb_img = np.float32(cv2.resize(image, (550, 550))) / 255
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    model = use_char(weights_path)
    output = model(batch)
    target_layers = [model.backbone.layers[-1]]
    # target_layers = [model.proto_net]
    targets = output
    cam = EigenCAM(model,
                   target_layers,
                   use_cuda=torch.cuda.is_available())

    grayscale_cam = cam(batch, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    output_dir = "./heat_testfiles"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir , "heat_map_" + save_name + ".png"),cam_image)


def Feature_Map1(image_path,weights_path = None):
    weights_path = 'weights/yolact_coco_custom_49_2450.pth'
    # image_path = "testfile/27_12.jpg"
    # image_path = window.Ui_From().select_image_ennew()
    net = use_char(weights_path)
    image = cv2.imread(image_path)
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    output = net(batch)

def Graph_Map():
    logs_path = "./logs_dir"
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    log_dir = os.path.join(logs_path, "graph_b_" + str(time_str))
    writer1 = SummaryWriter(log_dir, comment='graph_b_comment', filename_suffix="_graph_b_suffix")
    log_dir = os.path.join(logs_path, "graph_f_" + str(time_str))
    writer2 = SummaryWriter(log_dir, comment='graph_f_comment', filename_suffix="_graph_f_suffix")
    fake_img = torch.randn(1, 3, 550, 550)
    model1 = Yolact().backbone
    model2 = FPN([128,256,512])
    writer1.add_graph(model1, fake_img)
    writer2.add_graph(model2, fake_img)
    writer1.close()
    writer2.close()
