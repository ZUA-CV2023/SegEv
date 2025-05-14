from PyQt5 import QtWidgets, uic
import sys

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("ui.ui")  # 直接加载 .ui 文件
window.show()
sys.exit(app.exec_())