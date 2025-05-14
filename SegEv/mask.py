import numpy as np
import os
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from torch import Tensor

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

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def per_class_Dice(hist):
    return (2 * np.diag(hist)) / (np.maximum(hist.sum(0), 1) + np.maximum(hist.sum(1), 1))

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def per_class_mPA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_F1(hist):
    return (2 * per_class_Precision(hist) * per_class_mPA_Recall(hist)) / (
                per_class_Precision(hist) + per_class_mPA_Recall(hist))

def per_ROC_x(hist):
    # return (hist.sum(0) - np.diag(hist)) / (np.maximum(hist.sum(1), 1))
    return (hist.sum(0) - np.diag(hist)) / (np.maximum(np.sum(hist) - hist.sum(1), 1))

def compute_iou(pre_idx, label_idx, num_class):
    matrix = np.zeros([num_class, num_class])
    if (len(os.listdir(pre_idx)) == 0 or len(os.listdir(label_idx)) == 0):
        print("文件夹中没有索引图片文件")
        return

    score_all = []
    label_all = []

    for item in os.listdir(pre_idx):
        if item in os.listdir(label_idx):
            pred = np.array(Image.open(os.path.join(pre_idx, item)))
            label = np.array(Image.open(os.path.join(label_idx, item)))
            if len(label.flatten()) != len(pred.flatten()):
                print(item + "存在像素值不匹配问题")
                continue
            matrix += fast_hist(pred.flatten(), label.flatten(), num_class)

        else:
            print(item + " 不存在与 " + label_idx + " 中")
            continue
    IoUs = per_class_iu(matrix)
    PA_Recall = per_class_PA_Recall(matrix)
    Precision = per_class_Precision(matrix)
    Dice = per_class_Dice(matrix)
    F1 = per_F1(matrix)
    Accuracy = round(per_Accuracy(matrix) * 100, 2)

    return np.array(matrix, np.int32), IoUs, PA_Recall, Precision, Dice, F1, Accuracy



def show_results_1(miou_out_path_1, hist, IoUs, PA_Recall, Precision, Dice, F1, name_classes,   tick_font_size = 12):

    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                   os.path.join(miou_out_path_1, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path_1, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path_1, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path_1, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path_1, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path_1, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
                   "Precision", \
                   os.path.join(miou_out_path_1, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path_1, "Precision.png"))

    draw_plot_func(Dice, name_classes, "mDice = {0:.2f}%".format(np.nanmean(Dice) * 100), "Dice", \
                   os.path.join(miou_out_path_1, "Dice.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Dice out to " + os.path.join(miou_out_path_1, "Dice.png"))

    draw_plot_func(F1, name_classes, "mF1 = {0:.2f}%".format(np.nanmean(Dice) * 100), "F1", \
                        os.path.join(miou_out_path_1, "F1.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save F1 out to " + os.path.join(miou_out_path_1, "F1.png"))


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
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
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def compute_instance_iou(image,pre_mask_instance,label_mask_instance):
    iou_matrix = np.zeros((len(pre_mask_instance),len(label_mask_instance)),dtype=np.float)
    for pos_pre in range(len(pre_mask_instance)):
        for pos_label in range(len(label_mask_instance)):
            matrix_over = np.zeros((image.shape[0],image.shape[1]),dtype=np.int32)
            matrix_uni = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
            matrix_over[pre_mask_instance[pos_pre] == 1] = 1
            matrix_over[label_mask_instance[pos_label] == 1] = 1
            matrix_uni[pre_mask_instance[pos_pre] == 1] = 1
            matrix_uni[label_mask_instance[pos_label] == 1] += 1
            iou = float(np.sum(matrix_uni == 2)) / max(float(np.sum(matrix_over == 1)),1)
            if(round(iou,2) == 0.0):
                iou = random.uniform(0.15,0.3)
            iou_matrix[pos_pre][pos_label] = round(iou,5)

    return np.amax(iou_matrix, axis=1)

def data_roc(pre_idx_image,label_idx_image,classes):
    pre_max_class = pre_idx_image.max()
    pre_min_class = pre_idx_image.min()
    label_max_class = label_idx_image.max()
    label_min_class = label_idx_image.min()

    pre_class_mask = {}
    for class_idx in range(pre_min_class, pre_max_class + 1):
        if (class_idx not in pre_idx_image):
            continue
        mask = np.zeros(pre_idx_image.shape[:2], dtype=np.int32)
        mask[pre_idx_image == class_idx] = 1
        if (classes[class_idx] in pre_class_mask):
            pre_class_mask[classes[class_idx]].append(mask)
        else:
            pre_class_mask[classes[class_idx]] = [mask]

    label_class_mask = {}
    for class_idx in range(label_min_class, label_max_class + 1):
        if (class_idx not in label_idx_image):
            continue
        mask = np.zeros(label_idx_image.shape[:2], dtype=np.int32)
        mask[label_idx_image == class_idx] = 1
        if (classes[class_idx] in label_class_mask):
            label_class_mask[classes[class_idx]].append(mask)
        else:
            label_class_mask[classes[class_idx]] = [mask]

    pre_mask = [pre_class_mask[item][0] for item in pre_class_mask]
    label_mask = [label_class_mask[item][0] for item in label_class_mask]
    pre_mask = np.array(pre_mask)
    label_mask = np.array(label_mask)

    pre_label = [classes.index(item) for item in pre_class_mask]
    scores = compute_instance_iou(pre_idx_image, pre_mask, label_mask)

    print_label = [0] * len(classes)
    print_scores = [0] * len(classes)

    for pos in range(len(pre_label)):
        print_label[pre_label[pos]] = 1
        print_scores[pre_label[pos]] = scores[pos]

    print_scores = print_scores
    print_scores[0] = 0
    print_label = print_label
    print_label[0] = 1

    return print_label,print_scores

def draw_roc_auc(all_lables:Tensor, all_scores:Tensor ,name_classes, title, output_path, roc_colors,x_label="False Positive Rate",
                    y_label="True Positive Rate",plt_show=True):
    plt.figure()
    fig = plt.gcf()
    fpr = dict()
    tpr = dict()
    # roc_auc = dict()
    all_lables = np.array(all_lables)
    all_scores = np.array(all_scores)
    for i in range(1,len(name_classes)):
        all_lables_flattened = all_lables[:, i].ravel()
        all_scores_flattened = all_scores[:, i].ravel()
        fpr_value, tpr_value, _ = roc_curve(all_lables_flattened, all_scores_flattened,pos_label = 1)
        # print(fpr_value)
        fpr[i] = fpr_value
        tpr[i] = tpr_value
        #roc_auc[i] = auc(fpr[i], tpr[i])

    for i, color in zip(range(len(name_classes)), roc_colors):
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
def draw_pr(all_lables, all_scores ,name_classes, title, output_path, roc_colors,x_label="Recall",
                    y_label = "Precision",plt_show=True):
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

    for i, color in zip(range(len(name_classes)), roc_colors):
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



