from minisegnet import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from layers.output_utils import postprocess
from labelme import utils
from data import cfg
import torch
import cv2
import json
import os
import base64
import numpy as np
from eval import use_net


classes = ["LuanShu", "RongShu", "MuMian", "YangTiJia", "ShuiSha", "ZongLvShu"]
colors = [[0, 0, 0], [192, 128, 0], [0, 64, 128], [192, 128, 0],
          [192, 0, 128], [0, 128, 0], [128, 128, 0],[0, 0, 128], [128, 0, 128],
          [0, 128, 128], [128, 128, 128],[64, 0, 128], [128, 0, 0],
          [64, 0, 0], [192, 0, 0],[64, 128, 0], [128, 64, 0], [0, 192, 0],
          [128, 192, 0],[64, 128, 128]]
classes_idx = {}
for pos in range(len(classes)):
    classes_idx[pos] = classes[pos]

save_path_json = './testfile'


def pro_mask(dets_out, img, h, w, image,undo_transform=True,image_path =None):
    img_gpu = img / 255.0
    h, w, _ = img.shape
    t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0.15)
    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:15]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes_in, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]

    num_dets_to_consider = min(15, len(classes_in))
    for j in range(num_dets_to_consider):
        if scores[j] < 0.15:
            num_dets_to_consider = j
            break
    name = image_path.split('\\')[-1]
    name = name.split('.')[0]
    if(num_dets_to_consider > 0):
        masks = masks[:num_dets_to_consider, :, :, None]
        classes_in = classes_in[:num_dets_to_consider]
        img_numpy = (masks).byte().cpu().numpy()
        # 将所有图层的mask进行二值化
        for pos in range(len(img_numpy)):
            mask = img_numpy[pos, :, :, :]
            mask[mask != 0] = 1
            mask[mask == 0] = 0
            img_numpy[pos, :, :, :] = mask

        contours_demo(img_numpy, classes_in, image_path, name, image)
    else:
        print(str(name) + "未预测到掩码")

def img_to_base64(img_array):
    # 传入图片为RGB格式numpy矩阵，传出的base64也是通过RGB的编码
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) #RGB2BGR，用于cv2编码
    encode_image = cv2.imencode(".jpg", img_array)[1] #用cv2压缩/编码，转为一维数组
    byte_data = encode_image.tobytes() #转换为二进制
    base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
    return base64_str

def contours_demo(image,label,path,image_name,image_origin):
    '''
       image : 预测的mask
       label : mask对应的类别
       path : 对应的图片地址
       image_name : 图片名字
       image_origin : 原图片
    '''
    out = {"version": "5.3.1","flags": {},"shapes" : []}
    for pos in range(len(image)):
        dst = cv2.GaussianBlur(image[pos], (3, 3), 0)
        ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for num in range(len(contours)):
            if(len(contours[num]) < 10):
                continue
            shape_per = {}
            shape_per['label'] = classes[int(label[pos])]
            point = contours[num].reshape(-1, 2)
            point = np.array(point, dtype=float).tolist()
            shape_per['points'] = point
            shape_per["group_id"] = 'null'
            shape_per["shape_type"] = "polygon"
            shape_per["flags"] = {}
            out['shapes'].append(shape_per)
    out['imageData'] = img_to_base64(image_origin)
    out['imagePath'] = path
    out['imageHeight'] = image_origin.shape[0]
    out['imageWidth'] = image_origin.shape[1]
    with open(os.path.join(save_path_json,image_name.split('.')[0] + '.json'), 'w') as f:
        json.dump(out, f , indent=4, ensure_ascii=False)

def read_json(path):
    with open(path, 'r') as f:
        content = f.read()
    return json.loads(content)

def read_image(path):
    image = cv2.imread(path)
    return image

def pro_json(net:Yolact,path:str):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    pro_mask(preds, frame, None, None, undo_transform=False, image_path=path,image = cv2.imread(path))


def use_pro_json(net,path):
    for item in os.listdir(path):
        if (item.split('.')[1] == 'json'):
            continue
        image_path = os.path.join('testfile',item)
        pro_json(net, image_path)
        print(image_path + " : json文件转换完成")


if __name__ == '__main__':
    weights_path = 'weights/minisegnet_coco_custom_24_1200_zhiwu.pth'
    image_path_pro_json = 'testfile'
    net = use_net(weights_path)
    use_pro_json(net,image_path_pro_json)




