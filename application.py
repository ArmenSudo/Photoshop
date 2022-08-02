from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, Qt

import cv2

import numpy as np

import sys
import os
import shutil

import gray_filter
import max_rgb_filter
import edge_detection
import contrast
import gausian_blur

try:
    os.mkdir(".cache_File")
except FileExistsError:
    pass

global_photo = 0


# _________________________________________________         Camera code          ______________________________________
class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    frame = 0

    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            global global_photo
            ret, global_photo = Capture.read()
            if ret:
                Image = cv2.cvtColor(global_photo, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0],
                                           QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1280, 960, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()
        return


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle("Photo Shop")
        self.setGeometry(350, 100, 1200, 800)
        self.setStyleSheet("background-color:#C0C0C0;")

        self.createMenuBar()

        self.img_label = QtWidgets.QLabel(self)
        self.img_label.setGeometry(20, 40, 900, 740)
        self.img_label.setStyleSheet("background-color:#FFFFFF; border: 3px solid blue")

        # For Camera
        self.img_label1 = QtWidgets.QLabel(self)
        self.img_label1.setGeometry(20, 40, 900, 740)
        self.img_label1.setStyleSheet("background-color:#FFFFFF; border: 3px solid blue")
        self.img_label1.hide()

        # Max RGB filter button
        self.btn_max_rgb = QtWidgets.QPushButton("Max RGB", self)
        self.btn_max_rgb.setGeometry(980, 40, 150, 60)
        self.btn_max_rgb.setStyleSheet("border-radius: 5; background-color: #FFFF00")
        self.btn_max_rgb.setContentsMargins(5, 5, 5, 5)
        self.btn_max_rgb.clicked.connect(self.clicked_btn)

        # Blur filter button
        self.btn_blur = QtWidgets.QPushButton("Blur", self)
        self.btn_blur.setGeometry(980, 120, 150, 60)
        self.btn_blur.setStyleSheet("border-radius: 5; background-color: #FFFF00")
        self.btn_blur.setContentsMargins(5, 5, 5, 5)
        self.btn_blur.clicked.connect(self.clicked_btn)

        # Edge Detection filter button
        self.btn_edge_det = QtWidgets.QPushButton("Edge Detection", self)
        self.btn_edge_det.setGeometry(980, 200, 150, 60)
        self.btn_edge_det.setStyleSheet("border-radius: 5; background-color: #FFFF00")
        self.btn_edge_det.setContentsMargins(5, 5, 5, 5)
        self.btn_edge_det.clicked.connect(self.clicked_btn)

        # Contrast filter button
        self.btn_contrast = QtWidgets.QPushButton("Contrast", self)
        self.btn_contrast.setGeometry(980, 280, 150, 60)
        self.btn_contrast.setStyleSheet("border-radius: 5; background-color: #FFFF00")
        self.btn_contrast.setContentsMargins(5, 5, 5, 5)
        self.btn_contrast.clicked.connect(self.clicked_btn)

        # Gray filter button
        self.btn_gray = QtWidgets.QPushButton("Gray", self)
        self.btn_gray.setGeometry(980, 360, 150, 60)
        self.btn_gray.setStyleSheet("border-radius: 5; background-color: #FFFF00")
        self.btn_gray.setContentsMargins(5, 5, 5, 5)
        self.btn_gray.clicked.connect(self.clicked_btn)

        # undo button
        self.btn_undo = QtWidgets.QPushButton("Undo", self)
        self.btn_undo.setGeometry(980, 440, 150, 60)
        self.btn_undo.setStyleSheet("border-radius: 5; background-color: #FFFF00")
        self.btn_undo.setContentsMargins(5, 5, 5, 5)
        self.btn_undo.clicked.connect(self.clicked_btn)

        # camera button
        self.btn_camera = QtWidgets.QPushButton("ON Camera", self)
        self.btn_camera.setGeometry(980, 520, 150, 60)
        self.btn_camera.setContentsMargins(5, 5, 5, 5)
        self.btn_camera.setCheckable(True)
        self.btn_camera.clicked.connect(self.camera_status_button)

        # Status bar
        # self.status_bar = QtWidgets.QLabel

        # Action when you close Program
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.btn_camera = QtWidgets.QPushButton("ON Camera", self)
        self.btn_camera.setGeometry(980, 520, 150, 60)
        self.btn_camera.setContentsMargins(5, 5, 5, 5)
        self.btn_camera.setCheckable(True)
        self.btn_camera.clicked.connect(self.camera_status_button)
        # Camera
        self.btn_photo = QtWidgets.QPushButton("Photo", self)
        self.btn_photo.setGeometry(980, 600, 150, 60)
        self.btn_photo.setStyleSheet("border-radius: 5; background-color: #FFFF00")
        self.btn_photo.setContentsMargins(5, 5, 5, 5)
        self.btn_photo.hide()
        self.btn_photo.clicked.connect(self.CancelFeed)
        self.btn_photo.clicked.connect(self.clicked_btn)

    # ____________- Camera ________________
    def ImageUpdateSlot(self, Image):
        pixmap = QPixmap(QPixmap.fromImage(Image))
        pixmap_resized = pixmap.scaled(900, 740, QtCore.Qt.KeepAspectRatio)
        self.img_label1.resize(pixmap_resized.width(), pixmap_resized.height())
        self.img_label1.setAlignment(QtCore.Qt.AlignCenter)
        self.img_label1.setPixmap(pixmap_resized)

    def CancelFeed(self):
        try:
            self.Worker1.stop()
            self.img_label1.setPixmap(QtGui.QPixmap())
        except AttributeError:
            error = QMessageBox()
            error.setWindowTitle("Empty")
            error.setText("No Image!")
            error.setInformativeText("Add image and use filter")
            error.setIcon(QMessageBox.Information)
            error.exec_()

    def camera_status_button(self):
        if self.btn_camera.isChecked():
            self.Worker1 = Worker1()
            self.img_label.hide()
            self.img_label1.show()
            self.Worker1.start()
            self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
            self.btn_camera.setText("OFF Camera")
            self.btn_photo.show()
        else:
            self.CancelFeed()
            self.btn_camera.setText("ON Camera")
            self.btn_photo.hide()
            self.img_label1.hide()
            self.img_label.show()

    # Menu Bar
    def createMenuBar(self):
        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)

        self.menuBar.setStyleSheet("background-color:#FFFFFF")
        fileMenu = QMenu("&File", self)
        self.menuBar.addMenu(fileMenu)
        fileMenu.setStyleSheet("""QMenuBar {
                     background-color: blue;
                }""")

        fileMenu.addAction("Open", self.action_clicked)
        fileMenu.addAction("Save", self.action_clicked)


    # Menu button action
    @QtCore.pyqtSlot()
    def action_clicked(self):
        action = self.sender()
        if action.text() == "Open":
            try:
                self.fname = QFileDialog.getOpenFileName(self)[0]
                self.img = cv2.imread(self.fname)
                if self.fname != '':
                    self.open_img(self.fname)
                    cv2.imwrite('.cache_File/_________________orginal_______img___.jpg', self.img)

            except:
                print("No Such File")
        elif action.text() == "Save":

            try:

                save_path = QFileDialog.getSaveFileName(self)[0]
                cv2.imwrite(save_path, self.final_img)
            except cv2.error:
                print("Brnelem")
            except AttributeError:
                print("ba axper 2")

    # open image from file
    def open_img(self, path):
        self.final_img = cv2.imread(path)
        pixmap = QPixmap(path)
        pixmap_resized = pixmap.scaled(900, 740, QtCore.Qt.KeepAspectRatio)
        self.img_label.resize(pixmap_resized.width(), pixmap_resized.height())
        self.img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_label.setPixmap(pixmap_resized)

    # Button actions
    def clicked_btn(self):

        click = self.sender()
        try:
            match click.text():
                case "Max RGB":
                    img0 = max_rgb_filter.filtering(self.img)
                    cv2.imwrite('.cache_File/max_rgb.jpg', img0)
                    self.open_img('.cache_File/max_rgb.jpg')
                case "Blur":
                    img0 = gausian_blur.gaussian_blur(self.img)
                    cv2.imwrite('.cache_File/blur.jpg', img0)
                    self.open_img('.cache_File/blur.jpg')

                case "Edge Detection":
                    img0 = edge_detection.laplacian(self.img)
                    cv2.imwrite('.cache_File/edge_detect.jpg', img0)
                    self.open_img('.cache_File/edge_detect.jpg')
                case "Contrast":
                    img0 = contrast.contrast(self.img)
                    cv2.imwrite('.cache_File/contrast.jpg', img0)
                    self.open_img('.cache_File/contrast.jpg')
                case "Gray":
                    img = gray_filter.gray_imp(self.img)
                    cv2.imwrite('.cache_File/gray.jpg', img)
                    self.open_img('.cache_File/gray.jpg')
                case 'Photo':
                    self.img = global_photo
                    self.btn_photo.hide()
                    self.img = np.fliplr(self.img)
                    self.btn_camera.click()
                    cv2.imwrite('.cache_File/_________________orginal_______img___.jpg', self.img)
                    self.fname = '.cache_File/_________________orginal_______img___.jpg'
                    self.open_img('.cache_File/_________________orginal_______img___.jpg')
                case "Undo":
                    if self.fname != 0:
                        self.open_img('.cache_File/_________________orginal_______img___.jpg')
        except AttributeError:
            error = QMessageBox()
            error.setWindowTitle("Empty")
            error.setText("No Image!")
            error.setInformativeText("Add image and use filter")
            error.setIcon(QMessageBox.Information)
            error.exec_()
        except ValueError:
            error = QMessageBox()
            error.setWindowTitle("Empty")
            error.setText("No Image!")
            error.setInformativeText("Open Camera")
            error.setIcon(QMessageBox.Information)
            error.exec_()

    # When programme closed remove .cache_file directory
    def closeEvent(self, event):
        shutil.rmtree('.cache_File')


def application():
    app = QApplication(sys.argv)
    window = Window()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    application()

# pip install opencv-python-headless  ---->  camera problem
