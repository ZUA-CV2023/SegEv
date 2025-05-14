import os
import shutil
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QGridLayout, QMessageBox, QGraphicsView, QGraphicsPixmapItem, QLabel
import get_miou
from PyQt5.QtGui import QDesktopServices, QFont
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from mask import compute_iou,show_results_1,draw_roc_auc,draw_pr
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
import window
from eval import use_net
import cv2
from PIL import Image
import numpy as np
from map import Feature_Map1, Heat_Map1

# ②

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = window.Ui_From()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())

class Ui_From(QThread):
    finished = pyqtSignal()
    select_folders_path = []
    select_A_mask_path = []
    select_B_mask_path = []
    select_A_path = []
    select_B_path = []
    select_floders_ennew = []
    selected_files_ennew = []
    selected_file = []
    time = 0
    accuracy = 0
    T = 0
    path = ''
    path_ennew = ''
    result_dir = ''
    lables = []
    scores = []
    def setupUi(self, From):

        From.setObjectName("From")
        From.resize(1000, 600)
        From.setStyleSheet("background-color: rgb(248, 248, 251);")
        self.centralwidget = QtWidgets.QWidget(From)
        self.centralwidget.setObjectName("centralwidget")

        # 添加布局管理器
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.image_show = QtWidgets.QGraphicsView(self.centralwidget)
        self.image_show.setObjectName("image_show")
        self.image_show.setStyleSheet(("""
                                    QGraphicsView {
                                        background-color: white;
                                        border: 3px solid white;
                                        border-radius: 5px; 
                                    }
                                """))
        self.image_show.setAlignment(Qt.AlignLeft)
        self.image_show.setAlignment(Qt.AlignVCenter)
        self.image_show_two = QtWidgets.QGraphicsView()
        self.image_show_two.setObjectName("image_show_two")
        self.image_show_two.setStyleSheet(("""
                                            QGraphicsView {
                                                background-color: white;
                                                border: 3px solid white;
                                                border-radius: 5px;
                                            }
                                        """))
        self.image_show_two.setAlignment(Qt.AlignLeft)
        self.image_show_two.setAlignment(Qt.AlignVCenter)

        # 创建QGraphicsScene对象
        self.scene = QtWidgets.QGraphicsScene()
        self.scene_two = QtWidgets.QGraphicsScene()
        # 创建QGraphicsPixmapItem对象
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()

        self.strat = QtWidgets.QPushButton(self.centralwidget)
        self.strat.setGeometry(QtCore.QRect(530, 550, 93, 35))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.strat.setFont(font)
        self.strat.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.strat.setObjectName("strat")
        self.strat.clicked.connect(self.process)
        self.strat.clicked.connect(self.ben_strat)
        self.strat.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.strat.setStyleSheet("""
                                    QPushButton {
                                        background-color: rgb(26, 115, 232);
                                        border: 3px solid white;
                                        border-radius: 5px;
                                    }
                                """)


        self.classes = ["background", "LuanShu", "MuMian", "RongShu", "ShuiSha", "YangTiJia", "ZongLvShu"]
        self.roc_pr_color = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown']
        self.classes_idx = {}
        for pos in range(len(self.classes)):
            self.classes_idx[self.classes[pos]] = pos
        self.step_A = 0
        self.step_B = 0
        self.file_len_B = 0
        self.file_len_A = 0
        self.Accuracy_1 = 0
        self.Accuracy_2 = 0
        self.iterator_index_A = 0
        self.iterator_index_B = 0
        self.weights_path_A = None
        self.weights_path_B = None
        self.json_path_B = None
        self.json_path_A = None
        self.url_path = None
        self.path = None
        self.current_text = ""
        self.weights_select_floders = None
        if (os.path.exists("miou_out_1")):
            shutil.rmtree("miou_out_1")
        os.makedirs("miou_out_1")
        if (os.path.exists("miou_out_2")):
            shutil.rmtree("miou_out_2")
        os.makedirs("miou_out_2")
        if (os.path.exists("feature_testfiles")):
            shutil.rmtree("feature_testfiles")
        os.makedirs("feature_testfiles")
        if (os.path.exists("heat_testfiles")):
            shutil.rmtree("heat_testfiles")
        os.makedirs("heat_testfiles")
        if (os.path.exists("loss_testfiles")):
            shutil.rmtree("loss_testfiles")
        os.makedirs("loss_testfiles")
        def on_hover(event):
            self.strat.setStyleSheet("""  
                            QPushButton {  
                                background-color: white;  
                                border: 3px solid white; 
                                border-radius: 5px;
                            }  
                        """)

        def on_leave(event):
            self.strat.setStyleSheet(("""  
                            QPushButton {  
                                background-color: rgb(26, 115, 232) ;  
                                border: 3px solid white;
                                border-radius: 10px;
                            }  
                        """))

        self.strat.enterEvent = on_hover
        self.strat.leaveEvent = on_leave

        self.ennew = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.ennew.setFont(font)
        self.ennew.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.ennew.setObjectName("re_input")
        self.ennew.clicked.connect(self.openURL)
        self.ennew.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.ennew.setStyleSheet("""
                                            QPushButton {
                                                background-color: rgb(252,248,212);
                                                border: 3px solid white;
                                                border-radius: 5px;
                                            }
                                        """)

        def on_hover(event):
            self.ennew.setStyleSheet("""  
                                    QPushButton {  
                                        background-color: white;  
                                        border: 3px solid white;
                                        border-radius: 5px;
                                    }  
                                """)

        def on_leave(event):
            self.ennew.setStyleSheet(("""  
                                    QPushButton {  
                                        background-color: rgb(252,248,212) ;  
                                        border: 3px solid white;
                                        border-radius: 10px;
                                    }  
                                """))


        self.ennew.enterEvent = on_hover
        self.ennew.leaveEvent = on_leave


        self.re_input = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.re_input.setFont(font)
        self.re_input.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.re_input.setObjectName("re_input")
        self.re_input.clicked.connect(self.re_input_cb)
        self.re_input.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.re_input.setStyleSheet("""
                                                   QPushButton {
                                                       background-color: rgb(252,248,212);
                                                       border: 3px solid white; 
                                                       border-radius: 5px;
                                                   }
                                               """)
        def on_hover(event):
            self.re_input.setStyleSheet("""  
                                           QPushButton {  
                                               background-color: white;  
                                               border: 3px solid white; 
                                               border-radius: 5px;
                                           }  
                                       """)

        def on_leave(event):
            self.re_input.setStyleSheet(("""  
                                           QPushButton {  
                                               background-color: rgb(252,248,212) ;  
                                               border: 3px solid white;
                                               border-radius: 10px;
                                           }  
                                       """))
        self.re_input.enterEvent = on_hover
        self.re_input.leaveEvent = on_leave


        self.get_dir = QtWidgets.QPushButton(self.centralwidget)
        self.get_dir.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.get_dir.setObjectName("get_dir")
        self.get_dir.clicked.connect(self.select_floders)
        self.get_dir.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.get_dir.setStyleSheet("""
                            QPushButton {
                                background-color: rgb(240, 240, 240);
                                border: 3px solid white; 
                                border-radius: 5px;  
                            }
                        """)
        def on_hover(event):
            self.get_dir.setStyleSheet("""  
                            QPushButton {  
                                background-color: white;  
                                border: 3px solid white; 
                                border-radius: 5px; 
                            }  
                        """)
        def on_leave(event):
            self.get_dir.setStyleSheet(("""  
                            QPushButton {  
                                background-color: rgb(240, 240, 240) ;  
                                border: 3px solid white;   
                                border-radius: 10px;  
                            }  
                        """))

        self.get_dir.enterEvent = on_hover
        self.get_dir.leaveEvent = on_leave

        self.get_A_com_dir = QtWidgets.QPushButton(self.centralwidget)
        self.get_A_com_dir.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.get_A_com_dir.setObjectName("get_A_com_dir")
        self.get_A_com_dir.clicked.connect(self.select_A_mask)
        self.get_A_com_dir.setFont(QFont("Microsoft YaHei", 12))
        self.get_A_com_dir.setStyleSheet("""
                                   QPushButton {
                                       background-color: rgb(240, 240, 240);
                                       border: 3px solid white; 
                                       border-radius: 5px; 
                                   }
                               """)
        def on_hover(event):
            self.get_A_com_dir.setStyleSheet("""  
                                   QPushButton {  
                                       background-color: white;  
                                       border: 3px solid white;
                                       border-radius: 5px;  
                                   }  
                               """)
        def on_leave(event):
            self.get_A_com_dir.setStyleSheet(("""  
                                   QPushButton {  
                                       background-color: rgb(240, 240, 240) ;  
                                       border: 3px solid white;
                                       border-radius: 10px; 
                                   }  
                               """))
        self.get_A_com_dir.enterEvent = on_hover
        self.get_A_com_dir.leaveEvent = on_leave

        self.get_B_com_dir = QtWidgets.QPushButton(self.centralwidget)
        self.get_B_com_dir.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.get_B_com_dir.setObjectName("get_B_com_dirr")
        self.get_B_com_dir.clicked.connect(self.select_B_mask)
        self.get_B_com_dir.setFont(QFont("Microsoft YaHei", 12))
        self.get_B_com_dir.setStyleSheet("""
                                           QPushButton {
                                               background-color: rgb(240, 240, 240);
                                               border: 3px solid white; 
                                               border-radius: 5px; 
                                           }
                                       """)
        def on_hover(event):
            self.get_B_com_dir.setStyleSheet("""  
                                           QPushButton {  
                                               background-color: white;  
                                               border: 3px solid white;  
                                               border-radius: 5px;  
                                           }  
                                       """)

        def on_leave(event):
            self.get_B_com_dir.setStyleSheet(("""  
                                           QPushButton {  
                                               background-color: rgb(240, 240, 240) ;  
                                               border: 3px solid white;  
                                               border-radius: 10px; 
                                           }  
                                       """))

        self.get_B_com_dir.enterEvent = on_hover
        self.get_B_com_dir.leaveEvent = on_leave


        self.get_comparison_A = QtWidgets.QPushButton(self.centralwidget)
        self.get_comparison_A.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.get_comparison_A.setObjectName("get_B_com_dirr")
        self.get_comparison_A.clicked.connect(self.select_A)
        self.get_comparison_A.setFont(QFont("Microsoft YaHei", 12))
        self.get_comparison_A.setStyleSheet("""
                                           QPushButton {
                                               background-color: rgb(240, 240, 240);
                                               border: 3px solid white;  /* 设置边框颜色和宽度 */
                                               border-radius: 5px;  /* 设置圆角半径 */
                                           }
                                       """)
        def on_hover(event):
            self.get_comparison_A.setStyleSheet("""  
                                           QPushButton {  
                                               background-color: white;  
                                               border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                               border-radius: 5px;  /* 设置圆角半径 */  
                                           }  
                                       """)

        def on_leave(event):
            self.get_comparison_A.setStyleSheet(("""  
                                           QPushButton {  
                                               background-color: rgb(240, 240, 240) ;  
                                               border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                               border-radius: 10px;  /* 设置圆角半径 */  
                                           }  
                                       """))
        self.get_comparison_A.enterEvent = on_hover
        self.get_comparison_A.leaveEvent = on_leave

        self.get_comparison_B = QtWidgets.QPushButton(self.centralwidget)
        self.get_comparison_B.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.get_comparison_B.setObjectName("get_comparison_B ")
        self.get_comparison_B.clicked.connect(self.select_B)
        self.get_comparison_B.setFont(QFont("Microsoft YaHei", 12))
        self.get_comparison_B.setStyleSheet("""
                                                   QPushButton {
                                                       background-color: rgb(240, 240, 240);
                                                       border: 3px solid white;  /* 设置边框颜色和宽度 */
                                                       border-radius: 5px;  /* 设置圆角半径 */
                                                   }
                                               """)
        def on_hover(event):
            self.get_comparison_B.setStyleSheet("""  
                                                   QPushButton {  
                                                       background-color: white;  
                                                       border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                       border-radius: 5px;  /* 设置圆角半径 */  
                                                   }  
                                               """)

        def on_leave(event):
            self.get_comparison_B.setStyleSheet(("""  
                                                   QPushButton {  
                                                       background-color: rgb(240, 240, 240) ;  
                                                       border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                       border-radius: 10px;  /* 设置圆角半径 */  
                                                   }  
                                               """))
        self.get_comparison_B.enterEvent = on_hover
        self.get_comparison_B.leaveEvent = on_leave


        self.heat_map = QtWidgets.QPushButton(self.centralwidget)
        self.heat_map.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.heat_map.setObjectName("heat_map")
        self.heat_map.clicked.connect(self.heat_map_cb)
        self.heat_map.setFont(QFont("Microsoft YaHei", 12))
        self.heat_map.setStyleSheet("""
                                    QPushButton {
                                        background-color: rgb(240, 240, 240);
                                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                                        border-radius: 5px;  /* 设置圆角半径 */
                                    }
                                """)
        def on_hover(event):
            self.heat_map.setStyleSheet("""  
                                    QPushButton {  
                                        background-color: white;  
                                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                        border-radius: 5px;  /* 设置圆角半径 */  
                                    }  
                                """)

        def on_leave(event):
            self.heat_map.setStyleSheet(("""  
                                    QPushButton {  
                                        background-color: rgb(240, 240, 240) ;  
                                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                        border-radius: 10px;  /* 设置圆角半径 */  
                                    }  
                                """))

        self.heat_map.enterEvent = on_hover
        self.heat_map.leaveEvent = on_leave

        self.feature_map = QtWidgets.QPushButton(self.centralwidget)
        self.feature_map.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.feature_map.setObjectName("get_dir")
        self.feature_map.clicked.connect(self.feature_map_cb)
        self.feature_map.setFont(QFont("Microsoft YaHei", 12))
        self.feature_map.setStyleSheet("""
                                            QPushButton {
                                                background-color: rgb(240, 240, 240);
                                                border: 3px solid white;  /* 设置边框颜色和宽度 */
                                                border-radius: 5px;  /* 设置圆角半径 */
                                            }
                                        """)
        def on_hover(event):
            self.feature_map.setStyleSheet("""  
                                            QPushButton {  
                                                background-color: white;  
                                                border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                border-radius: 5px;  /* 设置圆角半径 */  
                                            }  
                                        """)
        def on_leave(event):
            self.feature_map.setStyleSheet(("""  
                                            QPushButton {  
                                                background-color: rgb(240, 240, 240) ;  
                                                border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                border-radius: 10px;  /* 设置圆角半径 */  
                                            }  
                                        """))

        self.feature_map.enterEvent = on_hover
        self.feature_map.leaveEvent = on_leave

        self.loss_map = QtWidgets.QPushButton(self.centralwidget)
        self.loss_map.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.loss_map.setObjectName("get_dir")
        self.loss_map.clicked.connect(self.loss_map_cb)
        self.loss_map.setFont(QFont("Microsoft YaHei", 12))
        self.loss_map.setStyleSheet("""
                                                    QPushButton {
                                                        background-color: rgb(240, 240, 240);
                                                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                                                        border-radius: 5px;  /* 设置圆角半径 */
                                                    }
                                                """)
        def on_hover(event):
            self.loss_map.setStyleSheet("""  
                                                    QPushButton {  
                                                        background-color: white;  
                                                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                        border-radius: 5px;  /* 设置圆角半径 */  
                                                    }  
                                                """)

        def on_leave(event):
            self.loss_map.setStyleSheet(("""  
                                                    QPushButton {  
                                                        background-color: rgb(240, 240, 240) ;  
                                                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                        border-radius: 10px;  /* 设置圆角半径 */  
                                                    }  
                                                """))
        self.loss_map.enterEvent = on_hover
        self.loss_map.leaveEvent = on_leave

        self.structura_map = QtWidgets.QPushButton(self.centralwidget)
        self.structura_map.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.structura_map.setObjectName("get_dir")
        self.structura_map.clicked.connect(self.structura_map_cb)
        self.structura_map.setFont(QFont("Microsoft YaHei", 12))
        self.structura_map.setStyleSheet("""
                                                            QPushButton {
                                                                background-color: rgb(240, 240, 240);
                                                                border: 3px solid white;  /* 设置边框颜色和宽度 */
                                                                border-radius: 5px;  /* 设置圆角半径 */
                                                            }
                                                        """)
        def on_hover(event):
            self.structura_map.setStyleSheet("""  
                                                            QPushButton {  
                                                                background-color: white;  
                                                                border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                                border-radius: 5px;  /* 设置圆角半径 */  
                                                            }  
                                                        """)

        def on_leave(event):
            self.structura_map.setStyleSheet(("""  
                                                            QPushButton {  
                                                                background-color: rgb(240, 240, 240) ;  
                                                                border: 3px solid white;  /* 设置边框颜色和宽度 */  
                                                                border-radius: 10px;  /* 设置圆角半径 */  
                                                            }  
                                                        """))

        self.structura_map.enterEvent = on_hover
        self.structura_map.leaveEvent = on_leave

        self.eval_cb = QtWidgets.QPushButton(self.centralwidget)
        self.eval_cb.setGeometry(QtCore.QRect(790, 120, 93, 28))
        self.eval_cb.setObjectName("Precision_show")
        self.eval_cb.setFont(QFont("Microsoft YaHei", 12,QFont.Bold))
        self.eval_cb.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(222, 233, 253);
                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                        border-radius: 5px;  /* 设置圆角半径 */
                    }
                """)
        def on_hover(event):
            self.eval_cb.setStyleSheet("""  
                    QPushButton {  
                        background-color: rgb(222,233,253);  
                        border: 3px solid rgb(222,233,253);  /* 设置边框颜色和宽度 */  
                        border-radius: 5px;  /* 设置圆角半径 */  
                    }  
                """)

        def on_leave(event):
            self.eval_cb.setStyleSheet(("""  
                    QPushButton {  
                        background-color: rgb(222, 233, 253) ;  
                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                        border-radius: 10px;  /* 设置圆角半径 */  
                    }  
                """))

        self.eval_cb.enterEvent = on_hover
        self.eval_cb.leaveEvent = on_leave

        self.jump_cb = QtWidgets.QPushButton(self.centralwidget)
        self.jump_cb.setGeometry(QtCore.QRect(790, 120, 93, 28))
        self.jump_cb.setObjectName("jump_cb")
        self.jump_cb.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.jump_cb.clicked.connect(self.key_strat)
        self.jump_cb.setStyleSheet("""
                    QPushButton {
                        background: rgb(26, 115, 232);
                        border: 3px solid white;
                        border-radius: 20px;  /* 设置为按钮宽度的一半，使其成为圆形 */
                    }
                """)


        self.Precision_show = QtWidgets.QPushButton(self.centralwidget)
        self.Precision_show.setGeometry(QtCore.QRect(790, 120, 93, 28))
        self.Precision_show.setObjectName("Precision_show")
        self.Precision_show.clicked.connect(self.precision_cb)
        self.Precision_show.setFont(QFont("Microsoft YaHei", 12))
        self.Precision_show.setStyleSheet("""
            QPushButton {
                background-color: rgb(240, 240, 240);
                border: 3px solid white;  /* 设置边框颜色和宽度 */
                border-radius: 5px;  /* 设置圆角半径 */
            }
        """)

        def on_hover(event):
            self.Precision_show.setStyleSheet("""  
            QPushButton {  
                background-color: white;  
                border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */  
                border-radius: 5px;  /* 设置圆角半径 */  
            }  
        """)

        def on_leave(event):
            self.Precision_show.setStyleSheet(("""  
            QPushButton {  
                background-color: rgb(240, 240, 240) ;  
                border: 3px solid white;  /* 设置边框颜色和宽度 */  
                border-radius: 10px;  /* 设置圆角半径 */  
            }  
        """))

        self.Precision_show.enterEvent = on_hover
        self.Precision_show.leaveEvent = on_leave


        self.Recall_show = QtWidgets.QPushButton(self.centralwidget)
        self.Recall_show.setGeometry(QtCore.QRect(790, 160, 93, 28))
        self.Recall_show.setObjectName("Recall_show")
        self.Recall_show.clicked.connect(self.recall_cb)
        self.Recall_show.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Recall_show.setFont(QFont("Microsoft YaHei", 12))
        self.Recall_show.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(240, 240, 240);
                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                        border-radius: 5px;  /* 设置圆角半径 */
                    }
                """)
        def on_hover(event):
            self.Recall_show.setStyleSheet("""  
                    QPushButton {  
                        background-color: white;  
                        border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */  
                        border-radius: 5px;  /* 设置圆角半径 */  
                    }  
                """)

        def on_leave(event):
            self.Recall_show.setStyleSheet(("""  
                    QPushButton {  
                        background-color: rgb(240, 240, 240) ;  
                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                        border-radius: 10px;  /* 设置圆角半径 */  
                    }  
                """))

        self.Recall_show.enterEvent = on_hover
        self.Recall_show.leaveEvent = on_leave

        self.F1_show = QtWidgets.QPushButton(self.centralwidget)
        self.F1_show.setGeometry(QtCore.QRect(790, 200, 93, 28))
        self.F1_show.setObjectName("F1_show")
        self.F1_show.clicked.connect(self.F1_cb)
        self.F1_show.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.F1_show.setFont(QFont("Microsoft YaHei", 12))
        self.F1_show.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(240, 240, 240);
                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                        border-radius: 5px;  /* 设置圆角半径 */
                    }
                """)
        def on_hover(event):
            self.F1_show.setStyleSheet("""  
                    QPushButton {  
                        background-color: white;  
                        border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */  
                        border-radius: 5px;  /* 设置圆角半径 */  
                    }  
                """)
        def on_leave(event):
            self.F1_show.setStyleSheet(("""  
                    QPushButton {  
                        background-color: rgb(240, 240, 240) ;  
                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                        border-radius: 10px;  /* 设置圆角半径 */  
                    }  
                """))

        self.F1_show.enterEvent = on_hover
        self.F1_show.leaveEvent = on_leave

        self.mPA_show = QtWidgets.QPushButton(self.centralwidget)
        self.mPA_show.setGeometry(QtCore.QRect(790, 240, 93, 28))
        self.mPA_show.setObjectName("mPA_show")
        self.mPA_show.clicked.connect(self.mpa_cb)
        self.mPA_show.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.mPA_show.setFont(QFont("Microsoft YaHei", 12))
        self.mPA_show.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(240, 240, 240);
                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                        border-radius: 5px;  /* 设置圆角半径 */
                    }
                """)
        def on_hover(event):
            self.mPA_show.setStyleSheet("""  
                    QPushButton {  
                        background-color: white;  
                        border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */  
                        border-radius: 5px;  /* 设置圆角半径 */  
                    }  
                """)
        def on_leave(event):
            self.mPA_show.setStyleSheet(("""  
                    QPushButton {  
                        background-color: rgb(240, 240, 240) ;  
                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                        border-radius: 10px;  /* 设置圆角半径 */  
                    }  
                """))

        self.mPA_show.enterEvent = on_hover
        self.mPA_show.leaveEvent = on_leave


        self.mIou_show = QtWidgets.QPushButton(self.centralwidget)
        self.mIou_show.setGeometry(QtCore.QRect(790, 280, 93, 28))
        self.mIou_show.setObjectName("mIou_show")
        self.mIou_show.clicked.connect(self.miou_cb)
        self.mIou_show.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.mIou_show.setFont(QFont("Microsoft YaHei", 12))
        self.mIou_show.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(240, 240, 240);
                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                        border-radius: 5px;  /* 设置圆角半径 */
                    }
                """)
        def on_hover(event):
            self.mIou_show.setStyleSheet("""  
                    QPushButton {  
                        background-color: white;  
                        border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */  
                        border-radius: 5px;  /* 设置圆角半径 */  
                    }  
                """)
        def on_leave(event):
            self.mIou_show.setStyleSheet(("""  
                    QPushButton {  
                        background-color: rgb(240, 240, 240) ;  
                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                        border-radius: 10px;  /* 设置圆角半径 */  
                    }  
                """))

        self.mIou_show.enterEvent = on_hover
        self.mIou_show.leaveEvent = on_leave


        self.Dice_show = QtWidgets.QPushButton(self.centralwidget)
        self.Dice_show.setGeometry(QtCore.QRect(790, 320, 93, 28))
        self.Dice_show.setObjectName("Dice_show")
        self.Dice_show.clicked.connect(self.dice_cb)
        self.Dice_show.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Dice_show.setFont(QFont("Microsoft YaHei", 12))
        self.Dice_show.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(240, 240, 240);
                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                        border-radius: 5px;  /* 设置圆角半径 */
                    }
                """)

        def on_hover(event):
            self.Dice_show.setStyleSheet("""  
                    QPushButton {  
                        background-color: white;  
                        border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */  
                        border-radius: 5px;  /* 设置圆角半径 */  
                    }  
                """)
        def on_leave(event):
            self.Dice_show.setStyleSheet(("""  
                    QPushButton {  
                        background-color: rgb(240, 240, 240) ;  
                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                        border-radius: 10px;  /* 设置圆角半径 */  
                    }  
                """))

        self.Dice_show.enterEvent = on_hover
        self.Dice_show.leaveEvent = on_leave
        self.Accuracy = QtWidgets.QPushButton(From)
        self.Accuracy.setGeometry(QtCore.QRect(790, 440, 93, 28))
        self.Accuracy.setObjectName("Accuracy")
        self.Accuracy.clicked.connect(self.accuracy_cb)
        self.Accuracy.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Accuracy.setFont(QFont("Microsoft YaHei", 12))
        self.Accuracy.setStyleSheet("""
                    QPushButton {
                        background-color: rgb(240, 240, 240);
                        border: 3px solid white;  /* 设置边框颜色和宽度 */
                        border-radius: 5px;  /* 设置圆角半径 */
                    }
                """)

        def on_hover(event):
            self.Accuracy.setStyleSheet("""  
                    QPushButton {  
                        background-color: white;  
                        border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */  
                        border-radius: 5px;  /* 设置圆角半径 */  
                    }  
                """)

        def on_leave(event):
            self.Accuracy.setStyleSheet(("""  
                    QPushButton {  
                        background-color: rgb(240, 240, 240) ;  
                        border: 3px solid white;  /* 设置边框颜色和宽度 */  
                        border-radius: 10px;  /* 设置圆角半径 */  
                    }  
                """))

        self.Accuracy.enterEvent = on_hover
        self.Accuracy.leaveEvent = on_leave

        self.Roc_show = QtWidgets.QPushButton(self.centralwidget)
        self.Roc_show.setGeometry(QtCore.QRect(790, 360, 93, 28))
        self.Roc_show.setObjectName("Roc_show")
        self.Roc_show.clicked.connect(self.roc_cb)
        self.Roc_show.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Roc_show.setFont(QFont("Microsoft YaHei", 12))
        self.Roc_show.setStyleSheet("""
                           QPushButton {
                               background-color: rgb(240, 240, 240);
                               border: 3px solid white;  /* 设置边框颜色和宽度 */
                               border-radius: 5px;  /* 设置圆角半径 */
                           }
                       """)

        def on_hover(event):
            self.Roc_show.setStyleSheet("""
                           QPushButton {
                               background-color: white;
                               border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */
                               border-radius: 5px;  /* 设置圆角半径 */
                           }
                       """)

        def on_leave(event):
            self.Roc_show.setStyleSheet(("""
                           QPushButton {
                               background-color: rgb(240, 240, 240) ;
                               border: 3px solid white;  /* 设置边框颜色和宽度 */
                               border-radius: 10px;  /* 设置圆角半径 */
                           }
                       """))

        self.Roc_show.enterEvent = on_hover
        self.Roc_show.leaveEvent = on_leave

        self.PR_show = QtWidgets.QPushButton(self.centralwidget)
        self.PR_show.setGeometry(QtCore.QRect(790, 400, 93, 28))
        self.PR_show.setObjectName("PR_show")
        self.PR_show.clicked.connect(self.pr_cb)
        self.PR_show.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.PR_show.setFont(QFont("Microsoft YaHei", 12))
        self.PR_show.setStyleSheet("""
                           QPushButton {
                               background-color: rgb(240, 240, 240);
                               border: 3px solid white;  /* 设置边框颜色和宽度 */
                               border-radius: 5px;  /* 设置圆角半径 */
                           }
                       """)

        def on_hover(event):
            self.PR_show.setStyleSheet("""
                           QPushButton {
                               background-color: white;
                               border: 3px solid rgb(26,115,232);  /* 设置边框颜色和宽度 */
                               border-radius: 5px;  /* 设置圆角半径 */
                           }
                       """)

        def on_leave(event):
            self.PR_show.setStyleSheet(("""
                           QPushButton {
                               background-color: rgb(240, 240, 240) ;
                               border: 3px solid white;  /* 设置边框颜色和宽度 */
                               border-radius: 10px;  /* 设置圆角半径 */
                           }
                       """))

        self.PR_show.enterEvent = on_hover
        self.PR_show.leaveEvent = on_leave
        self.acc_show_two = QtWidgets.QTextBrowser(self.centralwidget)
        self.acc_show_two.setGeometry(QtCore.QRect(790, 480, 91, 31))
        self.acc_show_two.setObjectName("acc_show_two")
        self.acc_show_two.setStyleSheet("""  
                            QTextBrowser {  
                                background-color: white;  
                                border: 3px solid rgb(255, 255, 255);  /* 设置边框颜色和宽度 */  
                                border-radius: 5px;  /* 设置圆角半径 */  
                            }  
                        """)
        self.acc_show = QtWidgets.QTextBrowser(self.centralwidget)
        self.acc_show.setGeometry(QtCore.QRect(790, 480, 91, 31))
        self.acc_show.setObjectName("acc_show")
        self.acc_show.setStyleSheet("""  
                                    QTextBrowser {  
                                        background-color: white;  
                                        border: 3px solid rgb(255, 255, 255);  /* 设置边框颜色和宽度 */  
                                        border-radius: 5px;  /* 设置圆角半径 */  
                                    }  
                                """)
        self.jump = QtWidgets.QTextBrowser(self.centralwidget)
        self.jump.setGeometry(QtCore.QRect(790, 480, 91, 31))
        self.jump.setObjectName("acc_show")
        self.jump.setStyleSheet("""
                                                    QTextBrowser {
                                                        background-color: white;
                                                        border: 3px solid rgb(255, 255, 255);  /* 设置边框颜色和宽度 */
                                                        border-radius: 5px;  /* 设置圆角半径 */
                                                    }
                                                """)
        self.jump.installEventFilter(self)


        self.label_A = QtWidgets.QLabel(self.centralwidget)
        self.label_A.setGeometry(QtCore.QRect(1, 1, 20, 20))
        self.label_A.setText("A")
        self.label_A.setFont(QFont("Microsoft YaHei", 18))
        self.label_B = QtWidgets.QLabel(self.centralwidget)
        self.label_B.setGeometry(QtCore.QRect(1, 1, 20, 20))
        self.label_B.setText("B")
        self.label_B.setFont(QFont("Microsoft YaHei", 18))
        self.label_C = QtWidgets.QLabel(self.centralwidget)
        self.label_C.setGeometry(QtCore.QRect(1, 1, 20, 20))
        self.label_C.setText("A")
        self.label_C.setFont(QFont("Microsoft YaHei", 18))
        self.label_D = QtWidgets.QLabel(self.centralwidget)
        self.label_D.setGeometry(QtCore.QRect(1, 1, 20, 20))
        self.label_D.setText("B")
        self.label_D.setFont(QFont("Microsoft YaHei", 18))

        #界面大小自适应
        self.gridLayout.addWidget(self.label_A, 1, 2)
        # 创建一个水平布局
        horizontal_layout = QtWidgets.QHBoxLayout()
        # 在水平布局中添加 label_A 标签
        horizontal_layout.addWidget(self.label_A)
        # 添加水平间隔
        horizontal_layout.addSpacing(10)  # 你可以根据需要设置水平间隔的大小
        # 在水平布局中添加新标签 true_label
        self.true_label = QtWidgets.QLabel("prediction")  # 你可以根据需要设置标签的文本
        self.true_label.setFont(QFont("Microsoft YaHei", 18))
        horizontal_layout.addWidget(self.true_label)
        self.true_label = QtWidgets.QLabel("label")  # 你可以根据需要设置标签的文本
        self.true_label.setFont(QFont("Microsoft YaHei", 18))
        horizontal_layout.addWidget(self.true_label)
        # 将水平布局添加到网格布局中，确保在 image_show 的正上方
        self.gridLayout.addLayout(horizontal_layout, 1, 2)
        self.gridLayout.addWidget(self.label_B, 10, 2)
        horizontal_layout = QtWidgets.QHBoxLayout()
        # 在水平布局中添加 label_A 标签
        horizontal_layout.addWidget(self.label_B)
        # 添加水平间隔
        horizontal_layout.addSpacing(10)  # 你可以根据需要设置水平间隔的大小
        # 在水平布局中添加新标签 true_label
        self.true_label = QtWidgets.QLabel("prediction")  # 你可以根据需要设置标签的文本
        self.true_label.setFont(QFont("Microsoft YaHei", 18))
        horizontal_layout.addWidget(self.true_label)
        self.true_label = QtWidgets.QLabel("label")  # 你可以根据需要设置标签的文本
        self.true_label.setFont(QFont("Microsoft YaHei", 18))
        horizontal_layout.addWidget(self.true_label)
        # 将水平布局添加到网格布局中，确保在 image_show 的正上方
        self.gridLayout.addLayout(horizontal_layout, 10, 2)
        self.gridLayout.addWidget(self.get_dir, 3, 1)
        self.gridLayout.addWidget(self.get_A_com_dir, 4, 1)
        self.gridLayout.addWidget(self.get_B_com_dir, 5, 1)
        self.gridLayout.addWidget(self.get_comparison_A, 6, 1)
        self.gridLayout.addWidget(self.get_comparison_B, 7, 1)
        self.gridLayout.addWidget(self.heat_map, 12, 1)
        self.gridLayout.addWidget(self.feature_map, 13, 1)
        self.gridLayout.addWidget(self.loss_map, 14, 1)
        self.gridLayout.addWidget(self.structura_map, 15, 1)
        self.gridLayout.addWidget(self.jump, 19, 1)
        self.jump_cb.setFixedSize(70, 70)  # 设置按钮的固定大小
        self.gridLayout.addWidget(self.jump_cb, 22, 1, alignment=Qt.AlignCenter)
        self.gridLayout.addWidget(self.ennew, 11, 1)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 2, 0, 1, 1)
        self.gridLayout.addWidget(self.image_show, 2, 2, 8, 1)
        self.gridLayout.addWidget(self.image_show_two, 11, 2, 16, 1)
        self.gridLayout.addWidget(self.re_input, 19, 4, 1, 1, alignment=QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.strat, 22, 4, 1, 1, alignment=QtCore.Qt.AlignCenter)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 14, 4, 1, 1)
        self.gridLayout.addWidget(self.eval_cb, 2, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 1, 4, 1, 6)
        self.gridLayout.addWidget(self.Precision_show, 3, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 1, 4, 1, 6)
        self.gridLayout.addWidget(self.Recall_show, 4, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 2, 4, 1, 6)
        self.gridLayout.addWidget(self.F1_show, 5, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 3, 4, 1, 6)
        self.gridLayout.addWidget(self.mPA_show, 6, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 4, 4, 1, 6)
        self.gridLayout.addWidget(self.mIou_show, 7, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 5, 4, 1, 6)
        self.gridLayout.addWidget(self.Dice_show, 8, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 6, 4, 1, 6)
        self.gridLayout.addWidget(self.Accuracy, 11, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 9, 4, 1, 6)
        self.gridLayout.addWidget(self.label_C, 12, 4)
        self.gridLayout.addWidget(self.label_D, 14, 4)
        self.gridLayout.addWidget(self.acc_show, 13, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 10, 4, 1, 6)
        self.gridLayout.addWidget(self.acc_show_two, 15, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 10, 4, 1, 6)
        self.gridLayout.addWidget(self.Roc_show, 9, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 7, 4, 1, 6)
        self.gridLayout.addWidget(self.PR_show, 10, 4, 1, 6)
        self.gridLayout.addItem(
            QtWidgets.QSpacerItem(5, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding), 8, 4, 1, 6)
        self.strat.setFixedWidth(150)
        self.strat.setFixedHeight(50)
        self.eval_cb.setFixedWidth(150)
        self.eval_cb.setFixedHeight(50)
        self.Precision_show.setFixedWidth(150)
        self.Precision_show.setFixedHeight(50)
        self.Recall_show.setFixedWidth(150)
        self.Recall_show.setFixedHeight(50)
        self.F1_show.setFixedWidth(150)
        self.F1_show.setFixedHeight(50)
        self.mPA_show.setFixedWidth(150)
        self.mPA_show.setFixedHeight(50)
        self.mIou_show.setFixedWidth(150)
        self.mIou_show.setFixedHeight(50)
        self.Dice_show.setFixedWidth(150)
        self.Dice_show.setFixedHeight(50)
        self.Accuracy.setFixedWidth(150)
        self.Accuracy.setFixedHeight(50)
        self.acc_show.setFixedHeight(50)
        self.acc_show.setFixedWidth(150)
        self.acc_show_two.setFixedHeight(50)
        self.acc_show_two.setFixedWidth(150)
        self.Roc_show.setFixedWidth(150)
        self.Roc_show.setFixedHeight(50)
        self.PR_show.setFixedWidth(150)
        self.PR_show.setFixedHeight(50)
        self.ennew.setFixedWidth(180)
        self.ennew.setFixedHeight(50)
        self.loss_map.setFixedHeight(50)
        self.structura_map.setFixedHeight(50)
        self.get_dir.setFixedWidth(180)
        self.get_dir.setFixedHeight(50)
        self.get_A_com_dir.setFixedWidth(180)
        self.get_A_com_dir.setFixedHeight(50)
        self.get_B_com_dir.setFixedWidth(180)
        self.get_B_com_dir.setFixedHeight(50)
        self.get_comparison_A.setFixedWidth(180)
        self.get_comparison_A.setFixedHeight(50)
        self.get_comparison_B.setFixedWidth(180)
        self.get_comparison_B.setFixedHeight(50)
        self.heat_map.setFixedHeight(50)
        self.heat_map.setFixedWidth(180)
        self.feature_map.setFixedWidth(180)
        self.structura_map.setFixedWidth(180)
        self.loss_map.setFixedWidth(180)
        self.jump.setFixedWidth(180)
        self.jump.setFixedHeight(60)
        self.feature_map.setFixedHeight(50)
        self.re_input.setFixedWidth(150)
        self.re_input.setFixedHeight(50)


        self.resizeEvent = self.update_progress_bar_position

        From.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(From)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 673, 26))
        self.menubar.setObjectName("menubar")
        From.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(From)
        self.statusbar.setObjectName("statusbar")
        From.setStatusBar(self.statusbar)

        self.retranslateUi(From)
        QtCore.QMetaObject.connectSlotsByName(From)

    def eventFilter(self, obj, event):
        if obj == self.jump and event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            text = event.text()

            if key == QtCore.Qt.Key_Backspace:
                # 处理 Backspace 键，删除最后一个字符
                self.current_text = self.jump.toPlainText()
                self.current_text = self.current_text[:-1]
                self.jump.setPlainText(self.current_text)
            else:
                # 追加输入的文本
                self.current_text = self.jump.toPlainText()
                self.current_text += text
                self.jump.setPlainText(self.current_text)

        return super().eventFilter(obj, event)

    def key_strat(self):
        if(self.current_text.isdigit()):
            if (self.current_text != "" or self.file_len_A == 0):
                if (1 < int(self.current_text) < self.file_len_A + 1):
                    if (self.weights_path_A is not None):
                        self.step_A = int(self.current_text)
                        self.process_images_A(self.step_A - 1)
                    if (self.weights_path_B is not None):
                        self.step_B = int(self.current_text)
                        self.process_images_B(self.step_B - 1)

    def select_floders(self):
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if folder_dialog.exec_():
            self.time = self.time + 1
            self.path = folder_dialog.selectedUrls()[0].toLocalFile() + "/"
            self.select_folders_path.append(self.path)
            self.weights_select_floders = QFileDialog.getOpenFileName(None, "选择权重文件",  self.json_path_A,
                                                      "权重文件 (*.pth);;所有文件 (*)")[0]

    def select_A_mask(self):
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if folder_dialog.exec_():
            self.time = self.time + 1
            path = folder_dialog.selectedUrls()[0].toLocalFile() + "/"
            self.select_A_mask_path.append(path)
            save_label = "miou_out/label_idx"
            hist_2, IoUs_2, PA_Recall_2, Precision_2, Dice_2, F1_2, Accuracy_2 = compute_iou(path, save_label,
                                                                                             len(self.classes))
            miou_out_path_1 = "miou_out_1"
            show_results_1(miou_out_path_1, hist_2, IoUs_2, PA_Recall_2, Precision_2, Dice_2, F1_2,
                                     self.classes)
            self.Accuracy_1 = Accuracy_2
            self.precision_cb()

    def select_B_mask(self):
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if folder_dialog.exec_():
            self.time = self.time + 1
            path = folder_dialog.selectedUrls()[0].toLocalFile() + "/"
            self.select_B_mask_path.append(path)
            save_label = "miou_out/label_idx"
            hist_2, IoUs_2, PA_Recall_2, Precision_2, Dice_2, F1_2, Accuracy_2 = compute_iou(path, save_label,
                                                                                             len(self.classes))
            miou_out_path_2 = "miou_out_2"
            show_results_1(miou_out_path_2, hist_2, IoUs_2, PA_Recall_2, Precision_2, Dice_2, F1_2,
                           self.classes)
            self.Accuracy_2 = Accuracy_2
            self.precision_cb()

    def setup_key_events_A(self):
        # 连接键盘事件处理函数
        self.image_show.keyPressEvent = self.key_press_event_A
    def setup_key_events_B(self):
        # 连接键盘事件处理函数
        self.image_show_two.keyPressEvent = self.key_press_event_B

    def key_press_event_A(self, event):
        # 捕获键盘事件
        if event.key() == Qt.Key_A:
            # 处理'A'键按下事件，处理上一张图片
            self.step_A = self.step_A - 1
            if self.step_A < 0:
                self.step_A = self.file_len_A - 1
                
            if (self.weights_path_B is not None):
                self.step_B = self.step_A
                self.process_images_B(self.step_B)
            self.process_images_A(self.step_A)

        elif event.key() == Qt.Key_D:
            # 处理'D'键按下事件，处理下一张图片
            self.step_A = (self.step_A + 1) % self.file_len_A

            if (self.weights_path_B is not None):
                self.step_B = self.step_A
                self.process_images_B(self.step_B)
            self.process_images_A(self.step_A)

    def key_press_event_B(self, event):
        # 捕获键盘事件
        if event.key() == Qt.Key_A:
            # 处理'A'键按下事件，处理上一张图片
            self.step_B = self.step_B - 1
            if self.step_B < 0:
                self.step_B = self.file_len_B - 1

            if(self.weights_path_A is not None):
                self.step_A = self.step_B
                self.process_images_A(self.step_B)
            self.process_images_B(self.step_B)

        elif event.key() == Qt.Key_D:
            # 处理'D'键按下事件，处理下一张图片
            self.step_B = (self.step_B + 1) % self.file_len_B
            if (self.weights_path_A is not None):
                self.step_A = self.step_B
                self.process_images_A(self.step_B)
            self.process_images_B(self.step_B)
    def select_A(self):
        self.time = self.time + 1
        if(self.json_path_A is None):
            folder_dialog = QFileDialog()
            folder_dialog.setFileMode(QFileDialog.Directory)
            folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
            if folder_dialog.exec_():
                self.json_path_A = folder_dialog.selectedUrls()[0].toLocalFile() + "/"

        self.weights_path_A, _ = QFileDialog.getOpenFileName(None, "选择权重文件",  self.json_path_A,
                                                      "权重文件 (*.pth);;所有文件 (*)")
        if(self.weights_path_A == ''):
            self.weights_path_A = None
        if(self.weights_path_A is not None):
            if (self.json_path_B is None):
                self.json_path_B = self.json_path_A
            net = use_net(self.weights_path_A)
            self.miou_A = get_miou.show_comparison(path=self.json_path_A, weights=self.weights_path_A,
                                                   classes=self.classes,
                                                   net=net)

            self.miou_A.data_roc_pr("miou_out_1","ROC_A.png","PR_A.png")

            self.setup_key_events_A()
            if (self.weights_path_B is not None):
                self.step_A = self.step_B
            self.process_images_A(self.step_A)

    def process_images_A(self, step):

        if self.weights_path_A is not None:
            self.scene.clear()
            if(self.json_path_B is None or self.weights_path_B is None):
                self.scene_two.clear()
                self.acc_show.clear()
            files = [item for item in os.listdir(self.json_path_A) if item.endswith('.json')]
            self.file_len_A = len(files)
            self.iterator_index_A = step
            item = files[self.iterator_index_A]
            name = item.split('.')[0]

            if not os.path.exists(os.path.join(self.json_path_A, name + '.jpg')) or not os.path.exists(
                    os.path.join(self.json_path_A, name + '.json')):
                print(name + ' 的 jpg / json  不存在')

            else:
                content_json = get_miou.read_json(os.path.join(self.json_path_A, name + '.json'))
                image = get_miou.read_image(os.path.join(self.json_path_A, name + '.jpg'))
                labels = [item['label'] for item in content_json['shapes']]
                image_label = self.miou_A.label_image(image, content_json['shapes'], label=labels,
                                                      classes_idx=self.classes_idx)
                pre_image, pre_label,pre_mask_instance,pre_boxes = self.miou_A.pre_color(image=image)
                label_mask_instance = self.miou_A.get_label_mask(image, content_json['shapes'])
                iou = self.miou_A.compute_instance_iou(image, pre_mask_instance, label_mask_instance)

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 1
                font_thickness = 2
                for pos in range(len(pre_label)):
                    x1, y1, x2, y2 = pre_boxes[pos]
                    text = "Iou : " + str(iou[pos])
                    cv2.putText(pre_image, text, (x1, y1 - 30), font_face, font_scale,
                                self.miou_A.colors[pos], font_thickness, cv2.LINE_AA)
                # 组合预测和标签图片并且显示
                image = Image.fromarray(np.uint8(image))
                pre_image = Image.fromarray(np.uint8(pre_image))
                image_label = Image.fromarray(np.uint8(image_label))
                image1 = Image.blend(image, pre_image, 0.5)
                image2 = Image.blend(image, image_label, 0.5)
                image_all = np.hstack([image1, image2])
                image_all = cv2.resize(image_all, (750, 350))
                height, width, channel = image_all.shape
                bytes_per_line = 3 * width
                q_image = QtGui.QImage(image_all.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)

                pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                self.scene.addItem(pixmap_item)
                self.image_show.setAlignment(Qt.AlignHCenter)
                self.image_show.setAlignment(Qt.AlignVCenter)
                self.image_show.setScene(self.scene)
                self.label_A.setText("①  " + name + '.jpg')
                self.select_A_path.append(self.json_path_A)
        else:
            self.label_A.setText("①   ")

    def select_B(self):

        self.time = self.time + 1
        if(self.json_path_B is None):
            folder_dialog = QFileDialog()
            folder_dialog.setFileMode(QFileDialog.Directory)
            folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
            if folder_dialog.exec_():
                self.json_path_B = folder_dialog.selectedUrls()[0].toLocalFile() + "/"

        self.weights_path_B, _ = QFileDialog.getOpenFileName(None, "选择权重文件", self.json_path_B,
                                                         "权重文件 (*.pth);;所有文件 (*)")
        if (self.weights_path_B == ''):
            self.weights_path_B = None
        if (self.weights_path_B is not None):
            if(self.json_path_A is None):
                self.json_path_A = self.json_path_B
            net = use_net(self.weights_path_B)
            self.miou_B = get_miou.show_comparison(path=self.json_path_B, weights=self.weights_path_B,
                                                   classes=self.classes,
                                                   net=net)
            self.miou_B.data_roc_pr("miou_out_2","ROC_B.png","PR_B.png")

            self.setup_key_events_B()
            if (self.weights_path_A is not None):
                self.step_B = self.step_A
            self.process_images_B(self.step_B)
    def process_images_B(self, step):
        if self.weights_path_B:
            self.scene_two.clear()
            if (self.json_path_A is None or self.weights_path_A is None):
                self.scene.clear()
                self.acc_show.clear()
            files = [item for item in os.listdir(self.json_path_B) if item.endswith('.json')]
            self.iterator_index_B = step
            self.file_len_B = len(files)
            item = files[self.iterator_index_B]
            name = item.split('.')[0]

            if not os.path.exists(os.path.join(self.json_path_B, name + '.jpg')) or not os.path.exists(
                    os.path.join(self.json_path_B, name + '.json')):
                print(name + ' 的 jpg / json  不存在')

            else:
                content_json = get_miou.read_json(os.path.join(self.json_path_B, name + '.json'))
                image = get_miou.read_image(os.path.join(self.json_path_B, name + '.jpg'))
                labels = [item['label'] for item in content_json['shapes']]
                image_label = self.miou_B.label_image(image, content_json['shapes'], label=labels,
                                                      classes_idx=self.classes_idx)
                pre_image, pre_label ,pre_mask_instance,pre_boxes= self.miou_B.pre_color(image=image)
                label_mask_instance = self.miou_B.get_label_mask(image, content_json['shapes'])
                iou = self.miou_B.compute_instance_iou(image, pre_mask_instance, label_mask_instance)

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 1
                font_thickness = 2
                for pos in range(len(pre_label)):
                    x1, y1, x2, y2 = pre_boxes[pos]
                    text = "Iou : " + str(iou[pos])
                    cv2.putText(pre_image, text, (x1, y1 - 30), font_face, font_scale,
                                self.miou_B.colors[pos], font_thickness, cv2.LINE_AA)
                # 组合预测和标签图片并且显示
                image = Image.fromarray(np.uint8(image))
                pre_image = Image.fromarray(np.uint8(pre_image))
                image_label = Image.fromarray(np.uint8(image_label))
                image1 = Image.blend(image, pre_image, 0.5)
                image2 = Image.blend(image, image_label, 0.5)
                image_all = np.hstack([image1, image2])
                image_all = cv2.resize(image_all, (750, 350))
                height, width, channel = image_all.shape
                bytes_per_line = 3 * width
                q_image = QtGui.QImage(image_all.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                self.scene_two.addItem(pixmap_item)
                self.image_show_two.setAlignment(Qt.AlignHCenter)
                self.image_show_two.setAlignment(Qt.AlignVCenter)
                self.image_show_two.setScene(self.scene_two)
                self.label_B.setText("B  " + name + '.jpg')
                self.select_B_path.append(self.json_path_B)
        else:
            self.label_B.setText("B   ")

    def select_image_ennew(self):
        image_dialog = QFileDialog()
        image_dialog.setFileMode(QFileDialog.ExistingFile)
        image_dialog.setNameFilter("Image Files (*.png *.jpg *.bmp)")
        if image_dialog.exec_():
            selected_file = image_dialog.selectedFiles()[0]
            self.time += 1
            self.path_ennew = selected_file
            self.selected_files_ennew.append(selected_file)
        return self.path_ennew
    def ben_strat(self):
        if(self.path is not None):
            weights_path = self.weights_select_floders
            self.net = use_net(weights_path)
            miou = get_miou.Get_miou(path=self.path, weights=weights_path, classes=self.classes, net=self.net)
            self.miou = miou
            miou_folder_path = 'miou_out'
            if (os.path.exists(miou_folder_path)):
                shutil.rmtree(miou_folder_path)
            save_pre = "miou_out/pre_idx"
            save_label = 'miou_out/label_idx'
            # 分别创建两个文件夹
            pre_idx_folder_path = os.path.join(miou_folder_path, 'pre_idx')
            label_idx_folder_path = os.path.join(miou_folder_path, 'label_idx')
            os.makedirs(pre_idx_folder_path)
            os.makedirs(label_idx_folder_path)

            self.miou.save_idx(save_pre, save_label)
            miou_out_path_1 = "miou_out_1"
            hist, IoUs, PA_Recall, Precision, Dice, F1, Accuracy, self.classes = self.miou.compute_iou(save_pre,
                                                                                                       save_label,
                                                                                                       len(self.classes))
            self.accuracy = Accuracy
            self.miou.show_results_1(miou_out_path_1, hist, IoUs, PA_Recall, Precision, Dice, F1, self.classes)
    def precision_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if("Precision.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1,"Precision.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "Precision.png")
        if ("Precision.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "Precision.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self.scene_two.addItem(pixmap_item)
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B  "+"Precision.png")
    def recall_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if ("Recall.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "Recall.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "Recall.png")
        if ("Recall.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "Recall.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self.scene_two.addItem(pixmap_item)
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B " + "Recall.png")
    def F1_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if ("F1.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "F1.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "F1.png")
        if ("F1.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "F1.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self.scene_two.addItem(pixmap_item)
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B  " + "F1.png")
    def mpa_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if ("mPA.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "mPA.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "mPA.png")
        if ("mPA.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "mPA.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene_two.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B  " + "mPA.png")
    def miou_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if ("mIoU.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "mIoU.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "mIoU.png")
        if ("mIoU.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "mIoU.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene_two.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B  " + "mIoU.png")
    def dice_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if ("Dice.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "Dice.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "Dice.png")
        if ("Dice.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "Dice.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene_two.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B  " + "Dice.png")
    def roc_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if ("ROC_A.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "ROC_A.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "ROC_A.png")
        if ("ROC_B.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "ROC_B.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene_two.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B  " + "ROC_B.png")
    def pr_cb(self):
        out_path_1 = "miou_out_1"
        name_1 = os.listdir(out_path_1)
        out_path_2 = "miou_out_2"
        name_2 = os.listdir(out_path_2)
        if ("PR_A.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "PR_A.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "PR_A.png")
        if ("PR_B.png" in name_2):
            self.scene_two.clear()
            file_path = os.path.join(out_path_2, "PR_B.png")
            image = cv2.imread(file_path)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene_two.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B  " + "PR_B.png")
    def heat_map_cb(self):
        out_path_1 = "heat_testfiles"
        name_1 = os.listdir(out_path_1)
        if ("heat_map_A.png" in name_1):
            self.scene.clear()
            if ("heat_map_B.png" not in name_1):
                self.scene_two.clear()
            file_path = os.path.join(out_path_1, "heat_map_A.png")
            image = Image.open(file_path)
            image = np.array(image)
            image = cv2.resize(image,(400,400))
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setAlignment(Qt.AlignLeft)
            self.image_show.setAlignment(Qt.AlignVCenter)
            self.image_show.setScene(self.scene)
            self.label_A.setText("①  " + "heat_map_A.png")
        else:
            self.label_A.setText("①  ")
        if ("heat_map_B.png" in name_1):
            self.scene_two.clear()
            if ("heat_map_A.png" not in name_1):
                self.scene.clear()
            file_path = os.path.join(out_path_1, "heat_map_B.png")
            image = Image.open(file_path)
            image = np.array(image)
            image = cv2.resize(image,(400,400))
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene_two.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show_two.setAlignment(Qt.AlignLeft)
            self.image_show_two.setAlignment(Qt.AlignVCenter)
            self.image_show_two.setScene(self.scene_two)
            self.label_B.setText("B " + "heat_map_B.png")
        else:
            self.label_B.setText("B  ")
    def feature_map_cb(self):
        out_path_1 = "feature_testfiles"
        name_1 = os.listdir(out_path_1)
        if ("feature_map.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "feature_map.png")
            image = cv2.imread(file_path)
            image = cv2.resize(image, (700, 500))
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setAlignment(Qt.AlignLeft)
            self.image_show.setAlignment(Qt.AlignVCenter)
            self.image_show.setScene(self.scene)
            self.scene_two.clear()
            self.label_A.setText("A  " + "feature_map.png")
            self.label_B.setText("B  " )
        else:
            self.label_A.setText("A  ")
    def loss_map_cb(self):
        out_path_1 = "loss_testfiles"
        name_1 = os.listdir(out_path_1)
        if ("loss_map.png" in name_1):
            self.scene.clear()
            file_path = os.path.join(out_path_1, "loss_map.png")
            image = cv2.imread(file_path)
            image = cv2.resize(image, (400, 400))
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Set QPixmap to QGraphicsPixmapItem
            pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # Add QGraphicsPixmapItem to QGraphicsScene
            self.scene.addItem(pixmap_item)
            # Set QGraphicsScene to QGraphicsView
            self.image_show.setAlignment(Qt.AlignLeft)
            self.image_show.setAlignment(Qt.AlignVCenter)
            self.image_show.setScene(self.scene)
            self.label_A.setText("A  " + "loss_map.png")
            self.label_B.setText("B")
            self.scene_two.clear()
    def structura_map_cb(self):
        out_path = "structura_testfiles"
        name = os.listdir(out_path)
        if("structura_map.png" in name):
            file_path = os.path.join(out_path,"structura_map.png")
            os.startfile(file_path)
    def re_input_cb(self):
        self.scene.clear()
        self.scene_two.clear()
        self.acc_show.clear()
        self.acc_show_two.clear()
        self.classes = ["background", "LuanShu", "MuMian", "RongShu", "ShuiSha", "YangTiJia", "ZongLvShu"]
        self.roc_pr_color = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown']
        self.classes_idx = {}
        self.step_A = 0
        self.step_B = 0
        for pos in range(len(self.classes)):
            self.classes_idx[self.classes[pos]] = pos
        self.Accuracy_1 = 0
        self.Accuracy_2 = 0
        self.iterator_index_A = 0
        self.iterator_index_B = 0
        self.weights_path_A = None
        self.weights_path_B = None
        self.json_path_B = None
        self.json_path_A = None
        self.url_path = None
        self.path = None
        self.current_text = ""
        self.label_A.setText("A")
        self.label_B.setText("B")
        self.weights_select_floders = None
        if (os.path.exists("miou_out_1")):
            shutil.rmtree("miou_out_1")
        os.makedirs("miou_out_1")
        if (os.path.exists("miou_out_2")):
            shutil.rmtree("miou_out_2")
        os.makedirs("miou_out_2")
        if (os.path.exists("feature_testfiles")):
            shutil.rmtree("feature_testfiles")
        os.makedirs("feature_testfiles")
        if (os.path.exists("heat_testfiles")):
            shutil.rmtree("heat_testfiles")
        os.makedirs("heat_testfiles")
        if (os.path.exists("loss_testfiles")):
            shutil.rmtree("loss_testfiles")
        os.makedirs("loss_testfiles")
    def accuracy_cb(self):
        self.accuracy_str  = "   "+str(self.Accuracy_1) + '%'
        self.acc_show.setText(self.accuracy_str)

        self.accuracy_str = "   "+str(self.Accuracy_2) + '%'
        self.acc_show_two.setText(self.accuracy_str)
    def run(self):
        time.sleep(self.T)
        self.finished.emit()
    def showMessageBox(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("提示")
        msg.exec_()
    def process(self):
        self.thread = Ui_From()
        self.thread.start()
    def update_progress_bar_position(self, event):
        window_width = self.centralwidget.width()
        window_height = self.centralwidget.height()

        progress_width = 300
        progress_height = 30
        progress_x = (window_width - progress_width) // 2
        progress_y = (window_height - progress_height) // 2

        self.progressBar.setGeometry(QtCore.QRect(progress_x, progress_y, progress_width, progress_height))
    def openURL(self):
        self.url_path = window.Ui_From.select_image_ennew(self)
        weight_A = QFileDialog.getOpenFileName(None, "选择权重文件",  self.url_path,
                                                      "权重文件 (*.pth);;所有文件 (*)")[0]
        weight_B = QFileDialog.getOpenFileName(None, "选择权重文件", self.url_path,
                                                    "权重文件 (*.pth);;所有文件 (*)")[0]
        if(self.url_path == '' or weight_A == ''):
            weight_A = None
        if (self.url_path == '' or weight_B == ''):
            weight_B = None
        if(self.url_path is not None and weight_A is not None):
            save_name = "A"
            Heat_Map1(self.url_path,save_name = save_name,weights_path = weight_A)
            self.feature_map_cb()
        if (self.url_path is not None and weight_B is not None):
            save_name = "B"
            Heat_Map1(self.url_path, save_name=save_name, weights_path=weight_B)
            self.feature_map_cb()
    def retranslateUi(self, From):
        _translate = QtCore.QCoreApplication.translate
        From.setWindowTitle(_translate("From", "SegEv"))
        self.strat.setText(_translate("From", "Start"))
        self.ennew.setText(_translate("From", "Visualization"))
        self.re_input.setText(_translate("From", "Refresh"))
        self.get_dir.setText(_translate("From", "Select file"))
        self.get_A_com_dir.setText(_translate("From", "Select mask file A"))
        self.get_B_com_dir.setText(_translate("From", "Select mask file B"))
        self.get_comparison_A.setText(_translate("From", "Compare weight A"))
        self.get_comparison_B.setText(_translate("From", "Compare weight B"))
        self.heat_map.setText(_translate("From", "heat map"))
        self.feature_map.setText(_translate("From", "feature map"))
        self.loss_map.setText(_translate("From", "Loss map"))
        self.structura_map.setText(_translate("From", "Structural map"))
        self.Precision_show.setText(_translate("From", "Precision"))
        self.eval_cb.setText(_translate("From", "Evaluation"))
        self.Recall_show.setText(_translate("From", "Recall"))
        self.F1_show.setText(_translate("From", "F1"))
        self.mPA_show.setText(_translate("From", "mPA"))
        self.mIou_show.setText(_translate("From", "mIou"))
        self.Dice_show.setText(_translate("From", "Dice"))
        self.Roc_show.setText(_translate("From", "ROC"))
        self.PR_show.setText(_translate("From", "PR"))
        self.Accuracy.setText(_translate("From", "Accuracy"))
        self.jump_cb.setText(_translate("From", "Go"))
