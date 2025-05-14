import sys

from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QTextEdit, QApplication
#from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

from utils.augmentations import FastBaseTransform
from utils import timer
from layers.output_utils import postprocess
from labelme import utils
from data import cfg
import torch
import cv2
import json
import numpy as np
from eval import use_net
from PIL import Image
import csv
import matplotlib.pyplot as plt
import window
from torch import Tensor
from mask import per_class_iu,per_class_PA_Recall,per_class_Precision,per_class_Dice,per_class_mPA_Recall,per_F1,per_Accuracy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def read_json(path):
    with open(path, 'r') as f:
        content = f.read()
    return json.loads(content)

def read_image(path):
    image = cv2.imread(path)
    return image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

class Get_miou():

    def __init__(self, path, weights, net, classes):
        super().__init__()
        self.net = net
        self.path = path
        self.weights = weights
        self.classes = classes
        self.classes_idx = {}
        self.top_k = 15
        self.thresh = 0.25
        self.lables = []
        self.scores = []


        self.all_lables = np.array([])     # 初始化一个空的 numpy 数组用于存储所有的 scores
        self.all_scores = []     # 初始化一个空的 numpy 数组用于存储所有的 scores
        for pos in range(len(classes)):
            self.classes_idx[self.classes[pos]] = pos
        self.roc_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown']
        self.colors = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))
        self.msg_box=QMessageBox()


    proce = window.Ui_From()
    def pre_color(self, image):
        frame = torch.from_numpy(image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)
        h, w, _ = image.shape
        t = postprocess(preds, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=self.thresh)
        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.top_k]

            if cfg.eval_mask_branch:
                masks = t[3][idx]
            classes_in, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.top_k, len(classes_in))
        for j in range(num_dets_to_consider):
            if scores[j] < self.thresh:
                num_dets_to_consider = j
                break

        if(num_dets_to_consider > 0):
            masks = masks[:num_dets_to_consider, :, :, None]
            masks = masks[:, :, :, 0]
            classes_in = classes_in[:num_dets_to_consider]
            classes_in = classes_in + 1
            # img_numpy.shape = [class,640,640]
            img_numpy = (masks).byte().cpu().numpy()
            class_all = []
            show_image = np.zeros([masks.shape[1], masks.shape[2], 3])
            pre_inx = np.zeros([h, w]).astype(int)
            for pos in range(len(classes_in)):
                class_all.append(self.classes[classes_in[pos]])
                pre_inx[img_numpy[pos] != 0] = classes_in[pos]

            for pos in classes_in:
                show_image[:, :, 0] += ((pre_inx == pos) * self.colors[pos][0])
                show_image[:, :, 1] += ((pre_inx == pos) * self.colors[pos][1])
                show_image[:, :, 2] += ((pre_inx == pos) * self.colors[pos][2])


        else:
            show_image = np.zeros([image.shape[0], image.shape[1], 3])
            class_all = []

        return show_image, class_all

    def label_image(self, image, shapes, label, classes_idx):
        '''
           classess_idx : 字典类型的标签名对应索引
        '''
        lbl = utils.shapes_to_label(image.shape, shapes, classes_idx)
        data_lable = np.array(lbl)
        show_image = np.zeros([image.shape[0], image.shape[1], 3])
        for item in label:
            '''
                由于minisegnet没有预测背景类别，所以 0 类 background 是没有的，在创建 show_image
                时候是默认全 0 的填充也就是与类别 0（background）同名,在给像素按照类别索引赋值的时候
                要给索引值加 1 ,也就是说从 1 开始（0变1，1变2......） ，这样的话就不会因为show_image
                的时候是全 0 而图片全都是索引0对应的颜色，而且好处是背景默认为 0。
                classes = ["background","LuanShu", "RongShu", "MuMian", "YangTiJia", "ShuiSha", "ZongLvShu"]
            '''
            index = self.classes.index(item)
            show_image[:, :, 0] += ((data_lable == index) * self.colors[index][0])
            show_image[:, :, 1] += ((data_lable == index) * self.colors[index][1])
            show_image[:, :, 2] += ((data_lable == index) * self.colors[index][2])
        return show_image

    def blend_image(self,image, image_pre, image_label, image_name, labels, pre_label, name, flag):
        '''
           image : 原图片
           image_pre : 预测后的图片
           image_label : 标注的图片
           image_name : 图片名字
           labels : 标签类别
           name : [path , 图片名称]
           flag : 是否保存blend后的图片
        '''
        image = Image.fromarray(np.uint8(image))
        image_pre = Image.fromarray(np.uint8(image_pre))
        image_label = Image.fromarray(np.uint8(image_label))
        image1 = Image.blend(image, image_pre, 0.5)
        image2 = Image.blend(image, image_label, 0.5)
        image_all = np.hstack([image1, image2])
        if (flag):
            assert len(name) == 2, "name应该是两个值文件夹和文件名称"
            cv2.imwrite(os.path.join(name[0], name[1] + '_blend.jpg'), image_all)
            print("保存 " + name[1] + ".jpg 完成")
        else:
            print("Show image ---" + image_name + '.jpg ,' + ' pred classes ----', format(pre_label))
            print("Show image ---" + image_name + '.jpg ,' + ' json classes ----', format(labels))
            print("\n")
            cv2.namedWindow("pre_and_label", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("pre_and_label", 800, 400)
            cv2.imshow("pre_and_label", np.array(image_all))
            cv2.waitKey(0)
            cv2.destroyWindow("pre_and_label")

    def show_blend_image(self, flag):
        '''
           net: 预加载完成的网络
           path : 图片和json文件所在文件夹
           flag : 是否下载组合的图片
        '''
        assert os.path.isdir(self.path), "path 需要输入文件地址"

        name_flag = ""
        for item in os.listdir(self.path):
            if (item.split('.')[0].split('_')[-1] == "blend"):
                continue
            name = item.split('.')[0]
            if (not os.path.exists(os.path.join(self.path, name + '.jpg')) or
                    not os.path.exists(os.path.join(self.path, name + '.json'))):
                print(name + ' 的 jpg / json  不存在')
                continue
            # 由于json文件的名字与图片的名字一样所有这里用于判断是否同名
            if (name_flag == name or item.split('.')[-1] == 'json'):
                continue
            else:
                name_flag = name

            content_json = read_json(os.path.join(self.path, name + '.json'))
            image = read_image(self.path + name + '.jpg')
            labels = [item['label'] for item in content_json['shapes']]
            image_label = self.label_image(image, content_json['shapes'], label=labels, classes_idx=self.classes_idx)
            # print(image_label)
            pre_image,pre_label = self.pre_color(image=cv2.imread(self.path + item))
            ##组合预测和标签图片并且显示
            self.blend_image(image, pre_image, image_label, image_name=name,
                             labels=labels, pre_label = pre_label,name=[self.path, name], flag=flag)

    def pre_idx(self, image):
        frame = torch.from_numpy(image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)
        h, w, _ = image.shape
        t = postprocess(preds, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=self.thresh)
        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.top_k]

            if cfg.eval_mask_branch:
                masks = t[3][idx]
            classes_in, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]

        self.scores = scores
        print(classes_in)
        print(scores)
        score = [0] * 7
        for i in range(len(scores)):

            score[classes_in[i] + 1] = scores[i]
        self.all_scores.append(score)
        print(self.all_scores)
        num_dets_to_consider = min(self.top_k, len(classes_in))
        for j in range(num_dets_to_consider):
            if scores[j] < self.thresh:
                num_dets_to_consider = j
                break

        if(num_dets_to_consider > 0):
            masks = masks[:num_dets_to_consider, :, :, None]
            masks = masks[:, :, :, 0]
            classes_in = classes_in[:num_dets_to_consider]
            img_numpy = (masks).byte().cpu().numpy()
            pre_inx = np.zeros([h, w]).astype(int)
            pre_seg = {}
            for pos in range(len(classes_in)):
                if(classes_in[pos] in pre_seg):
                    pre_seg[classes_in[pos]].append(img_numpy[pos])
                else:
                    pre_seg[classes_in[pos]] = [img_numpy[pos]]
            class_set = [item for item in pre_seg]

            for pos in range(len(class_set)):
                for mask in pre_seg[class_set[pos]]:
                    pre_inx[mask == 1] = class_set[pos] + 1
            return Image.fromarray(pre_inx)
        return 0

    def label_idx(self, image, shapes):
        label_name_to_value = {'background': 0}
        for shape in shapes:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))
        lbl = utils.shapes_to_label(image.shape, shapes, label_name_to_value)
        new = np.zeros([np.shape(image)[0], np.shape(image)[1]])
        for name in label_names:
            index_json = label_names.index(name)
            index_all = self.classes.index(name)
            new = new + index_all * (np.array(lbl) == index_json)
        return new

    def save_idx(self, save_pre, save_label):
        # self.msg_box.show()
        i = 0
        for item in os.listdir(self.path):
            count = int(len(os.listdir(self.path))/2)
            # print(count)

            if(item.split('.')[-1] != 'json'):
                continue
            i+=1
            name = item.split('.')[0]
            content_json = read_json(os.path.join(self.path, name + '.json'))
            image = read_image(self.path + name + '.jpg')
            pre_idx = self.pre_idx(image)
            if(isinstance(pre_idx,int)):
                print(name + " 生成索引图像中未预测出结果")
                pre_idx = np.zeros([image.shape[0], image.shape[1]]).astype(int)
                pre_idx = Image.fromarray(pre_idx)
            label_idx = self.label_idx(image, content_json['shapes'])
            unique_values = set()
            lable = [0]*7
            print(label_idx)
            for row in label_idx:
                for element in row:
                    # if element >= 0 and element <= 6:
                    #     unique_values.add(element)
                    if (element >= 0).all() and (element <= 6).all():
                        unique_values.update(element)  # 添加所有元素到集合
            print(unique_values)
            for value in unique_values:
                lable[int(value)] = 1
            self.lables.append(lable)
            pre_idx.save(os.path.join(save_pre,name + '.png'))
            utils.lblsave(os.path.join(save_label, name + '.png'), label_idx)
            # utils.lblsave(os.path.join(save_label, name + '.png'), label_idx[0])
            # label_single_channel = np.argmax(label_idx, axis=0)  # 变成 (640, 640)
            # utils.lblsave(os.path.join(save_label, name + '.png'), label_single_channel)
            print(str(i)+"/"+str(count),name + " ： 索引图像保存成功")

            #  进度提示框
            # self.msg_box.setText(str(i) + "/" + str(count)  + " "  + name + " ： 索引图像保存成功")
            self.msg_box.setText(str(i) + "/" + str(count) + "\n" + name + " ： 索引图像保存成功")

            self.msg_box.show()
            QApplication.processEvents()

        # self.proce.showFinishedMessage()
        self.msg_box.exec_()


    def compute_iou(self, pre_idx, label_idx, num_class):
        matrix = np.zeros([num_class, num_class])
        if (len(os.listdir(pre_idx)) == 0 or len(os.listdir(label_idx)) == 0):
            print("文件夹中没有索引图片文件")
            return
        for item in os.listdir(pre_idx):
            if item in os.listdir(label_idx):
                pred = np.array(Image.open(os.path.join(pre_idx, item)))
                label = np.array(Image.open(os.path.join(label_idx, item)))
                if len(label.flatten()) != len(pred.flatten()):
                    print(item + "存在像素值不匹配问题")
                    continue
                matrix += fast_hist(pred.flatten(), label.flatten(),num_class)

            else:
                print(item + " 不存在与 " + label_idx + " 中")
                continue
        IoUs = self.per_class_iu(matrix)
        PA_Recall = self.per_class_PA_Recall(matrix)
        Precision = self.per_class_Precision(matrix)
        Dice = self.per_class_Dice(matrix)
        mPA_Recall = self.per_class_mPA_Recall(matrix)
        F1 = self.per_F1(matrix)
        # ROC_x = self.per_ROC_x(matrix)
        Accuracy = round(self.per_Accuracy(matrix) * 100, 2)
        print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) +
              ' ; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) +
              ' ; Accuracy: ' + str(round(self.per_Accuracy(matrix) * 100, 2)) + "%")

        return np.array(matrix, np.int32), IoUs, mPA_Recall, Precision, Dice, F1,  Accuracy,  self.classes

    def per_class_iu(self,hist):
        return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

    def per_class_PA_Recall(self,hist):
        return np.diag(hist) / np.maximum(hist.sum(1), 1)

    def per_class_Precision(self,hist):
        return np.diag(hist) / np.maximum(hist.sum(0), 1)

    def per_Accuracy(self,hist):
        return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

    def per_class_Dice(self,hist):
        return (2 * np.diag(hist)) / (np.maximum(hist.sum(0), 1) + np.maximum(hist.sum(1), 1))

    def per_Accuracy(self,hist):
        return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

    def per_class_mPA_Recall(self,hist):
        return np.diag(hist) / np.maximum(hist.sum(1), 1)

    def per_F1(self,hist):
        return (2 * self.per_class_Precision(hist) * self.per_class_mPA_Recall(hist)) / (
                    self.per_class_Precision(hist) + self.per_class_mPA_Recall(hist))


    def show_results_1(self, miou_out_path_1, hist, IoUs, PA_Recall, Precision, Dice, F1, name_classes,   tick_font_size = 12):
        # self.draw_pr_func(Precision, PA_Recall, os.path.join(miou_out_path, "PR.png"), plt_show=False)
        self.draw_roc_auc(self.lables, self.all_scores, name_classes,"ROC", os.path.join(miou_out_path_1, "ROC_A.png"), plt_show=False)
        self.draw_pr(self.lables, self.all_scores, name_classes, "PR", os.path.join(miou_out_path_1, "PR_A.png"),plt_show=False)
        # self.draw_ROC_func(PA_Recall, ROC_x, os.path.join(miou_out_path, "ROC.png"), plt_show=False)
        #self.draw_pr(self.lables, self.scores, "pr", os.path.join(miou_out_path, "PR.png"), plt_show=False)
        self.draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                       os.path.join(miou_out_path_1, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
        print("Save mIoU out to " + os.path.join(miou_out_path_1, "mIoU.png"))

        self.draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                       os.path.join(miou_out_path_1, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save mPA out to " + os.path.join(miou_out_path_1, "mPA.png"))

        self.draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                       os.path.join(miou_out_path_1, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save Recall out to " + os.path.join(miou_out_path_1, "Recall.png"))

        self.draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
                       "Precision", \
                       os.path.join(miou_out_path_1, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save Precision out to " + os.path.join(miou_out_path_1, "Precision.png"))

        self.draw_plot_func(Dice, name_classes, "mDice = {0:.2f}%".format(np.nanmean(Dice) * 100), "Dice", \
                       os.path.join(miou_out_path_1, "Dice.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save Dice out to " + os.path.join(miou_out_path_1, "Dice.png"))

        self.draw_plot_func(F1, name_classes, "mF1 = {0:.2f}%".format(np.nanmean(Dice) * 100), "F1", \
                            os.path.join(miou_out_path_1, "F1.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save F1 out to " + os.path.join(miou_out_path_1, "F1.png"))

        # self.proce.showFinishedMessage()


        with open(os.path.join(miou_out_path_1, "confusion_matrix.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer_list = []
            writer_list.append([' '] + [str(c) for c in name_classes])
            for i in range(len(hist)):
                writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
            writer.writerows(writer_list)
        print("Save confusion_matrix out to " + os.path.join(miou_out_path_1, "confusion_matrix.csv"))

    def show_results_2(self, miou_out_path_2, hist, IoUs, PA_Recall, Precision, Dice, F1, name_classes,   tick_font_size = 12):
        # self.draw_pr_func(Precision, PA_Recall, os.path.join(miou_out_path, "PR.png"), plt_show=False)
        self.draw_roc_auc(self.lables, self.all_scores, name_classes,"ROC", os.path.join(miou_out_path_2, "ROC.png"), plt_show=False)
        self.draw_pr(self.lables, self.all_scores, name_classes, "PR", os.path.join(miou_out_path_2, "PR.png"),plt_show=False)
        # self.draw_ROC_func(PA_Recall, ROC_x, os.path.join(miou_out_path, "ROC.png"), plt_show=False)
        #self.draw_pr(self.lables, self.scores, "pr", os.path.join(miou_out_path, "PR.png"), plt_show=False)
        self.draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                       os.path.join(miou_out_path_2, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
        print("Save mIoU out to " + os.path.join(miou_out_path_2, "mIoU.png"))

        self.draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                       os.path.join(miou_out_path_2, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save mPA out to " + os.path.join(miou_out_path_2, "mPA.png"))

        self.draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                       os.path.join(miou_out_path_2, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save Recall out to " + os.path.join(miou_out_path_2, "Recall.png"))

        self.draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
                       "Precision", \
                       os.path.join(miou_out_path_2, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save Precision out to " + os.path.join(miou_out_path_2, "Precision.png"))

        self.draw_plot_func(Dice, name_classes, "mDice = {0:.2f}%".format(np.nanmean(Dice) * 100), "Dice", \
                       os.path.join(miou_out_path_2, "Dice.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save Dice out to " + os.path.join(miou_out_path_2, "Dice.png"))

        self.draw_plot_func(F1, name_classes, "mF1 = {0:.2f}%".format(np.nanmean(Dice) * 100), "F1", \
                            os.path.join(miou_out_path_2, "F1.png"), tick_font_size=tick_font_size, plt_show=False)
        print("Save F1 out to " + os.path.join(miou_out_path_2, "F1.png"))

        # self.proce.showFinishedMessage()


        with open(os.path.join(miou_out_path_2, "confusion_matrix.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer_list = []
            writer_list.append([' '] + [str(c) for c in name_classes])
            for i in range(len(hist)):
                writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
            writer.writerows(writer_list)
        print("Save confusion_matrix out to " + os.path.join(miou_out_path_2, "confusion_matrix.csv"))

    def adjust_axes(self,r, t, fig, axes):
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1] * propotion])

    def draw_plot_func(self, values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
        plt.figure()
        fig = plt.gcf()
        axes = plt.gca()
        plt.barh(range(len(values)), values, color='royalblue')
        plt.title(plot_title, fontsize=tick_font_size + 2)
        plt.xlabel(x_label, fontsize=tick_font_size)
        plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
        r = fig.canvas.get_renderer()
        for i, val in enumerate(values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
            if i == (len(values) - 1):
                self.adjust_axes(r, t, fig, axes)

        fig.tight_layout()
        fig.savefig(output_path)
        if plt_show:
            plt.show()
        plt.close()


    def draw_roc_auc(self, all_lables:Tensor, all_scores:Tensor ,name_classes, title, output_path, x_label="False Positive Rate",
                        y_label="True Positive Rate",plt_show=True):
        plt.figure()
        fig = plt.gcf()
        fpr = dict()
        tpr = dict()
        # roc_auc = dict()
        all_lables = np.array(all_lables)
        all_scores = np.array(all_scores)
        for i in range(len(name_classes)):
            all_lables_flattened = all_lables[:, i].ravel()
            all_scores_flattened = all_scores[:, i].ravel()
            fpr_value, tpr_value, _ = roc_curve(all_lables_flattened, all_scores_flattened)
            # print(fpr_value)
            fpr[i] = fpr_value
            tpr[i] = tpr_value
            #roc_auc[i] = auc(fpr[i], tpr[i])

        for i, color in zip(range(len(name_classes)), self.roc_colors):
            # print(i)
            if i == 0:
                continue
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                #label="ROC curve of class {0} (area = {1:0.2f})".format(self.name_classes[i], roc_auc[i]),
                label=name_classes[i],
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path)
        if plt_show:
            plt.show()
        plt.close()
    def draw_pr(self, all_lables:Tensor, all_scores:Tensor ,name_classes, title, output_path, x_label="Recall",
                        y_label = "Precision", plt_show=True):
        plt.figure()
        fig = plt.gcf()
        precision = dict()
        recall = dict()
        aps = dict()
        all_lables = np.array(all_lables)
        all_scores = np.array(all_scores)
        for i in range(len(name_classes)):
            all_lables_flattened = all_lables[:, i].ravel()
            all_scores_flattened = all_scores[:, i].ravel()
            precision[i], recall[i], thresholds = precision_recall_curve(all_lables_flattened, all_scores_flattened)
            aps[i] = average_precision_score(all_lables_flattened, all_scores_flattened)

        for i, color in zip(range(len(name_classes)), self.roc_colors):
            if i == 0:
                continue
            plt.plot(
                recall[i],
                precision[i],
                color=color,
                lw=2,
                label=name_classes[i],
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path)
        if plt_show:
            plt.show()
        plt.close()

    def compute_iou(self, pre_idx, label_idx, num_class):
        matrix = np.zeros([num_class, num_class])
        if (len(os.listdir(pre_idx)) == 0 or len(os.listdir(label_idx)) == 0):
            print("文件夹中没有索引图片文件")
            return
        for item in os.listdir(pre_idx):
            if item in os.listdir(label_idx):
                pred = np.array(Image.open(os.path.join(pre_idx, item)))
                label = np.array(Image.open(os.path.join(label_idx, item)))
                if len(label.flatten()) != len(pred.flatten()):
                    print(item + "存在像素值不匹配问题")
                    continue
                matrix += fast_hist(pred.flatten(), label.flatten(),num_class)

            else:
                print(item + " 不存在与 " + label_idx + " 中")
                continue
        IoUs = self.per_class_iu(matrix)
        PA_Recall = self.per_class_PA_Recall(matrix)
        Precision = self.per_class_Precision(matrix)
        Dice = self.per_class_Dice(matrix)
        mPA_Recall = self.per_class_mPA_Recall(matrix)
        F1 = self.per_F1(matrix)
        # ROC_x = self.per_ROC_x(matrix)
        Accuracy = round(self.per_Accuracy(matrix) * 100, 2)
        print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) +
              ' ; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) +
              ' ; Accuracy: ' + str(round(self.per_Accuracy(matrix) * 100, 2)) + "%")

        return np.array(matrix, np.int32), IoUs, mPA_Recall, Precision, Dice, F1,  Accuracy,  self.classes


def compute_iou(pre_idx, label_idx, num_class):
    matrix = np.zeros([num_class, num_class])
    if (len(os.listdir(pre_idx)) == 0 or len(os.listdir(label_idx)) == 0):
        print("文件夹中没有索引图片文件")
        return
    for item in os.listdir(pre_idx):
        if item in os.listdir(label_idx):
            pred = np.array(Image.open(os.path.join(pre_idx, item)))
            label = np.array(Image.open(os.path.join(label_idx, item)))
            if len(label.flatten()) != len(pred.flatten()):
                print(item + "存在像素值不匹配问题")
                continue
            matrix += fast_hist(pred.flatten(), label.flatten(),num_class)

        else:
            print(item + " 不存在与 " + label_idx + " 中")
            continue
    IoUs_2 = per_class_iu(matrix)
    PA_Recall_2 = per_class_PA_Recall(matrix)
    Precision_2 = per_class_Precision(matrix)
    Dice_2 = per_class_Dice(matrix)
    mPA_Recall_2 = per_class_mPA_Recall(matrix)
    F1_2 = per_F1(matrix)
    # ROC_x = self.per_ROC_x(matrix)
    Accuracy_2 = round(per_Accuracy(matrix) * 100, 2)
    print('===> mIoU: ' + str(round(np.nanmean(IoUs_2) * 100, 2)) +
          ' ; mPA: ' + str(round(np.nanmean(PA_Recall_2) * 100, 2)) +
          ' ; Accuracy: ' + str(round(per_Accuracy(matrix) * 100, 2)) + "%")

    return np.array(matrix, np.int32), IoUs_2, mPA_Recall_2, Precision_2, Dice_2, F1_2,  Accuracy_2

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


def read_json(path):
    with open(path, 'r') as f:
        content = f.read()
    return json.loads(content)

def read_image(path):
    image = cv2.imread(path)
    return image

class show_comparison:
    def __init__(self, path, weights, net, classes):
        self.net = net
        self.path = path
        self.weights = weights
        self.classes = classes
        self.classes_idx = {}
        self.top_k = 15
        self.thresh = 0.35
        self.scores_all = []
        self.label_all = []
        for pos in range(len(classes)):
            self.classes_idx[self.classes[pos]] = pos
        self.colors = ((244, 67, 54),
                       (233, 30, 99),
                       (156, 39, 176),
                       (103, 58, 183),
                       (63, 81, 181),
                       (33, 150, 243),
                       (3, 169, 244),
                       (0, 188, 212),
                       (0, 150, 136),
                       (76, 175, 80),
                       (139, 195, 74),
                       (205, 220, 57),
                       (255, 235, 59),
                       (255, 193, 7),
                       (255, 152, 0),
                       (255, 87, 34),
                       (121, 85, 72),
                       (158, 158, 158),
                       (96, 125, 139))
        self.roc_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown']

    def pre_color(self,image):
        frame = torch.from_numpy(image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)
        h, w, _ = image.shape
        t = postprocess(preds, w, h, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=self.thresh)
        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.top_k]

            if cfg.eval_mask_branch:
                masks = t[3][idx]
            classes_in, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.top_k, len(classes_in))
        for j in range(num_dets_to_consider):
            if scores[j] < self.thresh:
                num_dets_to_consider = j
                break

        if(num_dets_to_consider > 0):
            masks = masks[:num_dets_to_consider, :, :, None]
            masks = masks[:, :, :, 0]
            classes_in = classes_in[:num_dets_to_consider]
            scores = scores[:num_dets_to_consider]
            boxes = boxes[:num_dets_to_consider]
            classes_in = classes_in + 1
            # img_numpy.shape = [class,640,640]
            img_numpy = (masks).byte().cpu().numpy()
            class_all = []
            show_image = np.zeros([masks.shape[1], masks.shape[2], 3])
            pre_inx = np.zeros([h, w]).astype(int)
            for pos in range(len(classes_in)):
                class_all.append(self.classes[classes_in[pos]])
                pre_inx[img_numpy[pos] != 0] = classes_in[pos]
            for pos in classes_in:
                show_image[:, :, 0] += ((pre_inx == pos) * self.colors[pos][0])
                show_image[:, :, 1] += ((pre_inx == pos) * self.colors[pos][1])
                show_image[:, :, 2] += ((pre_inx == pos) * self.colors[pos][2])
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale =1
            font_thickness = 2
            for pos in range(len(scores)):
                x1, y1, x2, y2 = boxes[pos]
                text = self.classes[classes_in[pos]] + ":" + str(round(scores[pos],2))
                cv2.putText(show_image,text,(x1,y1), font_face, font_scale,
                            self.colors[pos], font_thickness, cv2.LINE_AA)
                cv2.rectangle(show_image, (x1, y1), (x2, y2), self.colors[pos], 2)
        else:
            show_image = np.zeros([image.shape[0], image.shape[1], 3])
            class_all = []
            img_numpy = []
            boxes = []
        return show_image, class_all,img_numpy,boxes

    def label_image(self,image, shapes, label, classes_idx):
        '''
           classess_idx : 字典类型的标签名对应索引
        '''
        lbl = utils.shapes_to_label(image.shape, shapes, classes_idx)
        data_lable = np.array(lbl)
        show_image = np.zeros([image.shape[0], image.shape[1], 3])

        show = {}
        for shape in shapes:
            points = np.array(shape["points"])
            label_show = shape["label"]
            x1,y1,x2,y2 = min(points[:,0]),min(points[:,1]),max(points[:,0]),max(points[:,1])
            if(label_show in show):
                show[label_show].append([x1,y1,x2,y2])
            else:
                show[label_show] = [[x1,y1,x2,y2]]

        for item in label:
            index = self.classes.index(item)
            show_image[:, :, 0] += ((data_lable == index) * self.colors[index][0])
            show_image[:, :, 1] += ((data_lable == index) * self.colors[index][1])
            show_image[:, :, 2] += ((data_lable == index) * self.colors[index][2])

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        font_thickness = 2
        for label in show:
            for box in show[label]:
                x1, y1, x2, y2 = box
                cv2.putText(show_image, label, (int(x1), int(y1)), font_face, font_scale,
                        self.colors[self.classes_idx[label]], font_thickness, cv2.LINE_AA)

                cv2.rectangle(show_image, (int(x1), int(y1)), (int(x2), int(y2)),
                              self.colors[classes_idx[label]], 2)

        return show_image

    def get_label_mask(self,image,shapes):

        mask_all = []
        for shape in shapes:
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            points = shape['points']
            label = shape['label']
            shape_type = shape.get('shape_type', None)
            bool_mask = utils.shape_to_mask(image.shape[:2], points, shape_type)
            mask[bool_mask] = 1
            mask_all.append(mask)
        return np.array(mask_all)

    def compute_instance_iou(self,image,pre_mask_instance,label_mask_instance):
        iou_matrix = np.zeros((len(pre_mask_instance),len(label_mask_instance)),dtype=np.float)
        for pos_pre in range(len(pre_mask_instance)):
            for pos_label in range(len(label_mask_instance)):
                matrix_over = np.zeros((image.shape[0],image.shape[1]),dtype=np.int32)
                matrix_uni = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
                matrix_over[pre_mask_instance[pos_pre] == 1] = 1
                matrix_over[label_mask_instance[pos_label] == 1] = 1
                matrix_uni[pre_mask_instance[pos_pre] == 1] = 1
                matrix_uni[label_mask_instance[pos_label] == 1] += 1
                iou = np.sum(matrix_uni == 2) / np.sum(matrix_over == 1)
                iou_matrix[pos_pre][pos_label] = round(iou,2)

        return np.amax(iou_matrix, axis=1)

    def show_blend_image(self,flag = False):
        '''
           net: 预加载完成的网络
           path : 图片和json文件所在文件夹
           flag : 是否下载组合的图片
        '''
        assert os.path.isdir(self.path), "path 需要输入文件地址"

        name_flag = ""
        for item in os.listdir(self.path):
            if (item.split('.')[0].split('_')[-1] == "blend"):
                continue
            name = item.split('.')[0]
            if (not os.path.exists(os.path.join(self.path, name + '.jpg')) or
                    not os.path.exists(os.path.join(self.path, name + '.json'))):
                print(name + ' 的 jpg / json  不存在')
                continue
            # 由于json文件的名字与图片的名字一样所有这里用于判断是否同名
            if (name_flag == name or item.split('.')[-1] == 'json'):
                continue
            else:
                name_flag = name

            content_json = read_json(os.path.join(self.path, name + '.json'))
            image = read_image(self.path + name + '.jpg')
            labels = [item['label'] for item in content_json['shapes']]
            image_label = self.label_image(image, content_json['shapes'], label=labels, classes_idx=self.classes_idx)
            pre_image,pre_label ,pre_mask_instance,pre_boxes= self.pre_color(image=cv2.imread(self.path + item))
            label_mask_instance = self.get_label_mask(image,content_json['shapes'])
            iou = self.compute_instance_iou(image,pre_mask_instance,label_mask_instance)

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            font_thickness = 2
            for pos in range(len(pre_label)):
                x1, y1, x2, y2 = pre_boxes[pos]
                text = "Iou : " + str(iou[pos])
                cv2.putText(pre_image, text, (x1, y1 - 30), font_face, font_scale,
                            self.colors[pos], font_thickness, cv2.LINE_AA)

            ##组合预测和标签图片并且显示
            image = Image.fromarray(np.uint8(image))
            pre_image = Image.fromarray(np.uint8(pre_image))
            image_label = Image.fromarray(np.uint8(image_label))
            image1 = Image.blend(image, pre_image, 0.5)
            image2 = Image.blend(image, image_label, 0.5)
            image_all = np.hstack([image1, image2])

            if (flag):
                assert len(name) == 2, "name应该是两个值文件夹和文件名称"
                cv2.imwrite(os.path.join(name[0], name[1] + '_blend.jpg'), image_all)
                print("保存 " + name[1] + ".jpg 完成")
            else:
                cv2.namedWindow("pre_and_label", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow("pre_and_label", 800, 400)
                cv2.imshow("pre_and_label", np.array(image_all))
                cv2.waitKey(0)
                cv2.destroyWindow("pre_and_label")

    def draw_roc_auc(self, all_lables, all_scores ,name_classes, title, output_path, x_label="False Positive Rate",
                        y_label="True Positive Rate",plt_show=True):
        plt.figure()
        fig = plt.gcf()
        fpr = dict()
        tpr = dict()
        # roc_auc = dict()
        all_lables = np.array(all_lables)
        all_scores = np.array(all_scores)
        for i in range(len(name_classes)):
            all_lables_flattened = all_lables[:, i].ravel()
            all_scores_flattened = all_scores[:, i].ravel()
            fpr_value, tpr_value, _ = roc_curve(all_lables_flattened, all_scores_flattened)
            # print(fpr_value)
            fpr[i] = fpr_value
            tpr[i] = tpr_value
            #roc_auc[i] = auc(fpr[i], tpr[i])

        for i, color in zip(range(len(name_classes)), self.roc_colors):
            # print(i)
            if i == 0:
                continue
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                #label="ROC curve of class {0} (area = {1:0.2f})".format(self.name_classes[i], roc_auc[i]),
                label=name_classes[i],
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path)
        if plt_show:
            plt.show()
        plt.close()
    def draw_pr(self, all_lables, all_scores ,name_classes, title, output_path, x_label="Recall",
                        y_label = "Precision", plt_show=True):
        plt.figure()
        fig = plt.gcf()
        precision = dict()
        recall = dict()
        aps = dict()
        all_lables = np.array(all_lables)
        all_scores = np.array(all_scores)
        for i in range(len(name_classes)):
            all_lables_flattened = all_lables[:, i].ravel()
            all_scores_flattened = all_scores[:, i].ravel()
            precision[i], recall[i], thresholds = precision_recall_curve(all_lables_flattened, all_scores_flattened)
            aps[i] = average_precision_score(all_lables_flattened, all_scores_flattened)

        for i, color in zip(range(len(name_classes)), self.roc_colors):
            if i == 0:
                continue
            plt.plot(
                recall[i],
                precision[i],
                color=color,
                lw=2,
                label=name_classes[i],
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path)
        if plt_show:
            plt.show()
        plt.close()

    def label_idx(self, image, shapes):
        label_name_to_value = {'background': 0}
        for shape in shapes:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))
        lbl = utils.shapes_to_label(image.shape, shapes, label_name_to_value)
        new = np.zeros([np.shape(image)[0], np.shape(image)[1]])
        for name in label_names:
            index_json = label_names.index(name)
            index_all = self.classes.index(name)
            new = new + index_all * (np.array(lbl) == index_json)
        return new

    def data_roc_pr(self,save_path,save_roc_name,save_pr_name):
        for image_name in os.listdir(self.path):
            if(image_name.split(".")[-1] == "json"):
                json_data = read_json(os.path.join(self.path,image_name))
                if(os.path.exists(os.path.join(self.path,image_name.split(".")[0] + ".jpg"))):
                    image = cv2.imread(os.path.join(self.path,image_name.split(".")[0] + ".jpg"))
                else:
                    continue
                label_idx = self.label_idx(image, json_data['shapes'])
                unique_values = set()
                label = [0] * 7

                max_class_idx = label_idx.max()
                min_class_idx = label_idx.min()

                for pos in range(int(min_class_idx),int(max_class_idx) + 1):
                    if(pos in label_idx):
                        unique_values.add(float(pos))

                # for row in label_idx:
                #     for element in row:
                #         if element >= 0 and element <= 6:
                #             unique_values.add(element)
                for value in unique_values:
                    label[int(value)] = 1
                self.label_all.append(label)
            else:
                path = os.path.join(self.path, image_name)
                image = cv2.imread(path)
                frame = torch.from_numpy(image).cuda().float()
                batch = FastBaseTransform()(frame.unsqueeze(0))
                preds = self.net(batch)
                h, w, _ = image.shape
                t = postprocess(preds, w, h, visualize_lincomb=False,
                                crop_masks=True,
                                score_threshold=self.thresh)
                with timer.env('Copy'):
                    idx = t[1].argsort(0, descending=True)[:self.top_k]

                    if cfg.eval_mask_branch:
                        masks = t[3][idx]
                    classes_in, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]

                score = [0] * len(self.classes)
                for pos in range(len(scores)):
                    score[classes_in[pos] + 1] = scores[pos]
                self.scores_all.append(score)

        self.draw_roc_auc(self.label_all, self.scores_all, self.classes,"ROC", os.path.join(save_path, save_roc_name), plt_show=False)
        self.draw_pr(self.label_all, self.scores_all, self.classes, "PR", os.path.join(save_path, save_pr_name),plt_show=False)


# #
# if __name__ == "__main__":
#
#     # Precision.png  Recall.png  F1.png  mPA.png    mIoU.png    Dice.png
#     classes = ["background", "LuanShu", "MuMian", "RongShu", "ShuiSha", "YangTiJia", "ZongLvShu"]
#     weights_path = 'weights/minisegnet_with_effo.pth'
#     path = "testfile"
#     net = use_net(weights_path)
#     miou = show_comparison(path=path, weights=weights_path, classes=classes, net=net)
#     miou.data_roc_pr("miou_out_1","ROC.png","PR.png")


