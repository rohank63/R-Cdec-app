# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rcdec.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1200, 720)
        MainWindow.setMinimumSize(QtCore.QSize(1200, 720))
        MainWindow.setMaximumSize(QtCore.QSize(1200, 720))
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(970, 660, 191, 32))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.clear = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.clear.setFont(font)
        self.clear.setObjectName("clear")
        self.gridLayout_3.addWidget(self.clear, 0, 0, 1, 1)
        self.quit = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        self.quit.setFont(font)
        self.quit.setObjectName("quit")
        self.gridLayout_3.addWidget(self.quit, 0, 1, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(14, 10, 1171, 641))
        font = QtGui.QFont()
        font.setBold(True)
        self.tabWidget.setFont(font)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_main = QtWidgets.QWidget()
        self.tab_main.setObjectName("tab_main")
        self.image_name_label = QtWidgets.QLabel(self.tab_main)
        self.image_name_label.setGeometry(QtCore.QRect(440, 360, 271, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        self.image_name_label.setFont(font)
        self.image_name_label.setText("")
        self.image_name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_name_label.setObjectName("image_name_label")
        self.cmd_btn = QtWidgets.QCommandLinkButton(self.tab_main)
        self.cmd_btn.setGeometry(QtCore.QRect(681, 90, 30, 30))
        font = QtGui.QFont()
        font.setBold(False)
        self.cmd_btn.setFont(font)
        self.cmd_btn.setAutoFillBackground(False)
        self.cmd_btn.setText("")
        self.cmd_btn.setObjectName("cmd_btn")
        self.layoutWidget_2 = QtWidgets.QWidget(self.tab_main)
        self.layoutWidget_2.setGeometry(QtCore.QRect(30, 100, 19, 191))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ci = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.ci.setEnabled(False)
        self.ci.setText("")
        self.ci.setObjectName("ci")
        self.verticalLayout.addWidget(self.ci)
        self.cw = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.cw.setEnabled(False)
        self.cw.setText("")
        self.cw.setObjectName("cw")
        self.verticalLayout.addWidget(self.cw)
        self.ccf = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.ccf.setEnabled(False)
        self.ccf.setText("")
        self.ccf.setObjectName("ccf")
        self.verticalLayout.addWidget(self.ccf)
        self.ccl = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.ccl.setEnabled(False)
        self.ccl.setText("")
        self.ccl.setObjectName("ccl")
        self.verticalLayout.addWidget(self.ccl)
        self.layoutWidget_3 = QtWidgets.QWidget(self.tab_main)
        self.layoutWidget_3.setGeometry(QtCore.QRect(200, 410, 191, 32))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget_3)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.r_pre = QtWidgets.QPushButton(self.layoutWidget_3)
        font = QtGui.QFont()
        font.setBold(True)
        self.r_pre.setFont(font)
        self.r_pre.setObjectName("r_pre")
        self.gridLayout_2.addWidget(self.r_pre, 0, 0, 1, 1)
        self.r_res = QtWidgets.QPushButton(self.layoutWidget_3)
        self.r_res.setEnabled(False)
        font = QtGui.QFont()
        font.setBold(True)
        self.r_res.setFont(font)
        self.r_res.setObjectName("r_res")
        self.gridLayout_2.addWidget(self.r_res, 0, 1, 1, 1)
        self.l_img_no = QtWidgets.QLabel(self.tab_main)
        self.l_img_no.setGeometry(QtCore.QRect(545, 90, 58, 16))
        self.l_img_no.setText("")
        self.l_img_no.setAlignment(QtCore.Qt.AlignCenter)
        self.l_img_no.setObjectName("l_img_no")
        self.layoutWidget_4 = QtWidgets.QWidget(self.tab_main)
        self.layoutWidget_4.setGeometry(QtCore.QRect(60, 110, 331, 263))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.layoutWidget_4)
        self.gridLayout_10.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget_4)
        self.label.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.u_image = QtWidgets.QPushButton(self.layoutWidget_4)
        self.u_image.setAcceptDrops(False)
        self.u_image.setObjectName("u_image")
        self.gridLayout.addWidget(self.u_image, 0, 1, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget_4)
        self.label_4.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        self.u_weight = QtWidgets.QPushButton(self.layoutWidget_4)
        self.u_weight.setObjectName("u_weight")
        self.gridLayout_4.addWidget(self.u_weight, 0, 1, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_4, 1, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.u_cfg = QtWidgets.QPushButton(self.layoutWidget_4)
        self.u_cfg.setObjectName("u_cfg")
        self.gridLayout_5.addWidget(self.u_cfg, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget_4)
        self.label_5.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_5, 2, 0, 1, 1)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget_4)
        self.label_6.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_6.addWidget(self.label_6, 0, 0, 1, 1)
        self.u_class = QtWidgets.QPushButton(self.layoutWidget_4)
        self.u_class.setObjectName("u_class")
        self.gridLayout_6.addWidget(self.u_class, 0, 1, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_6, 3, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 0, 1, 1)
        self.gridLayout_10.addLayout(self.gridLayout_8, 0, 0, 1, 1)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.in_prob = QtWidgets.QPushButton(self.layoutWidget_4)
        font = QtGui.QFont()
        font.setBold(True)
        self.in_prob.setFont(font)
        self.in_prob.setObjectName("in_prob")
        self.gridLayout_9.addWidget(self.in_prob, 0, 1, 1, 1)
        self.in_thres = QtWidgets.QPushButton(self.layoutWidget_4)
        font = QtGui.QFont()
        font.setBold(True)
        self.in_thres.setFont(font)
        self.in_thres.setObjectName("in_thres")
        self.gridLayout_9.addWidget(self.in_thres, 0, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem1, 0, 3, 1, 1)
        self.gridLayout_10.addLayout(self.gridLayout_9, 1, 0, 1, 1)
        self.l_image_thumb = QtWidgets.QLabel(self.tab_main)
        self.l_image_thumb.setEnabled(True)
        self.l_image_thumb.setGeometry(QtCore.QRect(440, 120, 271, 221))
        self.l_image_thumb.setText("")
        self.l_image_thumb.setObjectName("l_image_thumb")
        self.layoutWidget_5 = QtWidgets.QWidget(self.tab_main)
        self.layoutWidget_5.setGeometry(QtCore.QRect(760, 20, 391, 571))
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.layoutWidget_5)
        self.gridLayout_11.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.bar = QtWidgets.QProgressBar(self.layoutWidget_5)
        font = QtGui.QFont()
        font.setBold(False)
        self.bar.setFont(font)
        self.bar.setProperty("value", 24)
        self.bar.setObjectName("bar")
        self.gridLayout_11.addWidget(self.bar, 2, 0, 1, 1)
        self.log = QtWidgets.QTextBrowser(self.layoutWidget_5)
        font = QtGui.QFont()
        font.setBold(False)
        self.log.setFont(font)
        self.log.setObjectName("log")
        self.gridLayout_11.addWidget(self.log, 0, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.gridLayout_11.addItem(spacerItem2, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_main, "")
        self.tab_filter = QtWidgets.QWidget()
        self.tab_filter.setObjectName("tab_filter")
        self.layoutWidget_6 = QtWidgets.QWidget(self.tab_filter)
        self.layoutWidget_6.setGeometry(QtCore.QRect(500, 430, 211, 32))
        self.layoutWidget_6.setObjectName("layoutWidget_6")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.layoutWidget_6)
        self.gridLayout_12.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget_6)
        self.label_2.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_12.addWidget(self.label_2, 0, 0, 1, 1)
        self.pb_filter_open = QtWidgets.QPushButton(self.layoutWidget_6)
        self.pb_filter_open.setAcceptDrops(False)
        self.pb_filter_open.setObjectName("pb_filter_open")
        self.gridLayout_12.addWidget(self.pb_filter_open, 0, 1, 1, 1)
        self.l_filter_img_tmb_in = QtWidgets.QLabel(self.tab_filter)
        self.l_filter_img_tmb_in.setEnabled(True)
        self.l_filter_img_tmb_in.setGeometry(QtCore.QRect(150, 110, 271, 221))
        self.l_filter_img_tmb_in.setText("")
        self.l_filter_img_tmb_in.setObjectName("l_filter_img_tmb_in")
        self.l_filter_name_in = QtWidgets.QLabel(self.tab_filter)
        self.l_filter_name_in.setGeometry(QtCore.QRect(150, 350, 271, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        self.l_filter_name_in.setFont(font)
        self.l_filter_name_in.setText("")
        self.l_filter_name_in.setAlignment(QtCore.Qt.AlignCenter)
        self.l_filter_name_in.setObjectName("l_filter_name_in")
        self.l_filter_img_tmb_out = QtWidgets.QLabel(self.tab_filter)
        self.l_filter_img_tmb_out.setEnabled(True)
        self.l_filter_img_tmb_out.setGeometry(QtCore.QRect(740, 110, 271, 221))
        self.l_filter_img_tmb_out.setText("")
        self.l_filter_img_tmb_out.setObjectName("l_filter_img_tmb_out")
        self.pb_filter_convert = QtWidgets.QPushButton(self.tab_filter)
        self.pb_filter_convert.setEnabled(False)
        self.pb_filter_convert.setGeometry(QtCore.QRect(560, 540, 81, 32))
        self.pb_filter_convert.setAcceptDrops(False)
        self.pb_filter_convert.setObjectName("pb_filter_convert")
        self.rb_grayscale = QtWidgets.QRadioButton(self.tab_filter)
        self.rb_grayscale.setGeometry(QtCore.QRect(550, 494, 19, 18))
        self.rb_grayscale.setText("")
        self.rb_grayscale.setObjectName("rb_grayscale")
        self.label_3 = QtWidgets.QLabel(self.tab_filter)
        self.label_3.setGeometry(QtCore.QRect(575, 494, 64, 16))
        self.label_3.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.tabWidget.addTab(self.tab_filter, "")
        self.tab_history = QtWidgets.QWidget()
        self.tab_history.setObjectName("tab_history")
        self.tbl_view = QtWidgets.QTableView(self.tab_history)
        self.tbl_view.setGeometry(QtCore.QRect(50, 60, 1070, 431))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tbl_view.sizePolicy().hasHeightForWidth())
        self.tbl_view.setSizePolicy(sizePolicy)
        self.tbl_view.setLineWidth(1)
        self.tbl_view.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tbl_view.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.tbl_view.setSortingEnabled(True)
        self.tbl_view.setObjectName("tbl_view")
        self.tbl_view.horizontalHeader().setStretchLastSection(True)
        self.tbl_view.verticalHeader().setStretchLastSection(False)
        self.pb_save = QtWidgets.QPushButton(self.tab_history)
        self.pb_save.setEnabled(False)
        self.pb_save.setGeometry(QtCore.QRect(925, 530, 100, 32))
        self.pb_save.setObjectName("pb_save")
        self.l_logo_exl = QtWidgets.QLabel(self.tab_history)
        self.l_logo_exl.setGeometry(QtCore.QRect(875, 525, 50, 45))
        self.l_logo_exl.setText("")
        self.l_logo_exl.setAlignment(QtCore.Qt.AlignCenter)
        self.l_logo_exl.setObjectName("l_logo_exl")
        self.tabWidget.addTab(self.tab_history, "")
        self.tab_update = QtWidgets.QWidget()
        self.tab_update.setObjectName("tab_update")
        self.pb_download = QtWidgets.QPushButton(self.tab_update)
        self.pb_download.setGeometry(QtCore.QRect(385, 540, 100, 32))
        self.pb_download.setObjectName("pb_download")
        self.log_download = QtWidgets.QTextBrowser(self.tab_update)
        self.log_download.setGeometry(QtCore.QRect(385, 320, 440, 200))
        self.log_download.setObjectName("log_download")
        self.l_gif = QtWidgets.QLabel(self.tab_update)
        self.l_gif.setGeometry(QtCore.QRect(370, 30, 471, 271))
        self.l_gif.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.l_gif.setFrameShadow(QtWidgets.QFrame.Plain)
        self.l_gif.setText("")
        self.l_gif.setAlignment(QtCore.Qt.AlignCenter)
        self.l_gif.setObjectName("l_gif")
        self.tabWidget.addTab(self.tab_update, "")
        self.tab_about = QtWidgets.QWidget()
        self.tab_about.setObjectName("tab_about")
        self.l_bgr = QtWidgets.QLabel(self.tab_about)
        self.l_bgr.setGeometry(QtCore.QRect(10, 0, 1200, 720))
        self.l_bgr.setText("")
        self.l_bgr.setObjectName("l_bgr")
        self.l_about = QtWidgets.QLabel(self.tab_about)
        self.l_about.setGeometry(QtCore.QRect(440, 150, 291, 261))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        self.l_about.setFont(font)
        self.l_about.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.l_about.setText("")
        self.l_about.setAlignment(QtCore.Qt.AlignCenter)
        self.l_about.setObjectName("l_about")
        self.tabWidget.addTab(self.tab_about, "")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "R-Cdec"))
        self.clear.setText(_translate("MainWindow", "Clear"))
        self.quit.setText(_translate("MainWindow", "Quit"))
        self.r_pre.setText(_translate("MainWindow", "Predict"))
        self.r_res.setText(_translate("MainWindow", "Result"))
        self.label.setText(_translate("MainWindow", "Upload Images"))
        self.u_image.setText(_translate("MainWindow", "Open"))
        self.label_4.setText(_translate("MainWindow", "Upload Weights"))
        self.u_weight.setText(_translate("MainWindow", "Open"))
        self.u_cfg.setText(_translate("MainWindow", "Open"))
        self.label_5.setText(_translate("MainWindow", "Upload Configuration"))
        self.label_6.setText(_translate("MainWindow", "Upload Classes"))
        self.u_class.setText(_translate("MainWindow", "Open"))
        self.in_prob.setText(_translate("MainWindow", "Probability"))
        self.in_thres.setText(_translate("MainWindow", "Threshold"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_main), _translate("MainWindow", "Main"))
        self.label_2.setText(_translate("MainWindow", "Upload Image"))
        self.pb_filter_open.setText(_translate("MainWindow", "Open"))
        self.pb_filter_convert.setText(_translate("MainWindow", "Convert"))
        self.label_3.setText(_translate("MainWindow", "Grayscale"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_filter), _translate("MainWindow", "Filter"))
        self.pb_save.setText(_translate("MainWindow", "Save"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_history), _translate("MainWindow", "History"))
        self.pb_download.setText(_translate("MainWindow", "Download"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_update), _translate("MainWindow", "Update"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_about), _translate("MainWindow", "About"))
