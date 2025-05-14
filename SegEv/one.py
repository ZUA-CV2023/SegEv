from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

def show_message_box():
    msg_box = QMessageBox()
    msg_box.setText("This is a non-modal message box.")
    msg_box.setInformativeText("This box will close automatically in a few seconds.")
    msg_box.setIcon(QMessageBox.Information)

    # Create a timer to close the message box after a delay
    timer = QTimer(msg_box)
    timer.timeout.connect(msg_box.accept)
    timer.start(4000)  # 3000 milliseconds (3 seconds)

    # Show the message box
    msg_box.exec_()

if __name__ == "__main__":
    app = QApplication([])
    show_message_box()
    app.exec_()
