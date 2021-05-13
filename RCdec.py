from pathlib import Path
import sys
# QtWidgets to work with widgets
from PyQt5 import QtWidgets
# QPixmap to work with images
from PyQt5.QtGui import QPixmap,QIcon, QMovie, QFont
from PyQt5.QtCore import QAbstractTableModel, Qt
from PIL import Image
import pandas as pd
from datetime import date
import urllib.request
import requests


import ui
from yolo4 import infer,grayscale

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class MainApp(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    # Constructor of the class
    def __init__(self):
        # We use here super() that allows multiple inheritance of all variables,
        # methods, etc. from file design
        # And avoiding referring to the base class explicitly
        super().__init__()

        # Initializing created design that is inside file design
        self.setupUi(self)

        # Connecting event of clicking on the button with needed function
        self.set_params()
        self.quit.clicked.connect(self.show_msg_dialog)
        self.clear.clicked.connect(self.text_box_clear)

        # Tab Main
        self.u_image.clicked.connect(self.upload_image)
        self.u_weight.clicked.connect(self.upload_weight)
        self.u_cfg.clicked.connect(self.upload_cfg)
        self.u_class.clicked.connect(self.upload_class)
        self.in_prob.clicked.connect(self.set_prob)
        self.in_thres.clicked.connect(self.set_thres)
        self.r_pre.clicked.connect(self.predict)
        self.r_res.clicked.connect(self.result)
        self.cmd_btn.clicked.connect(self.next_image)

        # Tab filter
        self.pb_filter_open.clicked.connect(self.filter_upload_image)
        self.pb_filter_convert.clicked.connect(self.filter_convert_image)

        # Tab History
        self.pb_save.clicked.connect(self.save_to_excel)

        # Tab Update
        self.pb_download.clicked.connect(self.download_repo) 
    
    def set_params(self):

        self.image_path = (None,None)
        self.weight_path = (None,None)
        self.cfg_path = (None,None)
        self.class_path = (None,None)
        self.probability = 0.4
        self.threshold = 0.9
        self.bar.setValue(0)
        self.img_index = 0

        # Filter
        self.filter_image_path = (None,None)

        # Dataframe
        self.df = pd.DataFrame(columns=['Image', 'Total Number of Cells\nDetected', 'Number of Cells Detected\n(With Non-Maximum Supression)', 'Probability', 'Threshold'])
        
        # Execl logo
        exl_img_path = "./files/exl.png"
        pixmap_image = QPixmap(exl_img_path)
        self.l_logo_exl.setPixmap(pixmap_image.scaled(self.l_logo_exl.width(), self.l_logo_exl.height()))

        # Gif
        self.movie = QMovie("./files/giphy.gif")
        #self.movie.setScaledSize(self.l_gif.size())
        self.l_gif.setMovie(self.movie)
        self.movie.start()

        # About
        #self.l_about.setFont(QFont('Times', 16))
        self.l_about.setText("R-Cdec is an\n\n Image Visualization,\n Image Filtering,\n and Cell Detection\nsoftware built to do the cell counting\nwith over 96% accuracy for\nlarge batch of Images\n\n  v 1.0.1")
        
    def mssgbtn(self):
        #self.log.append("QUIT")
        pass
    
    def show_msg_dialog(self):

        
        self.msg = QtWidgets.QMessageBox(self)
        self.msg.setIcon(QtWidgets.QMessageBox.Information)

        self.msg.setText("Are you sure you want to Quit?")
        #msg.setInformativeText("This is additional information")
        self.msg.setWindowTitle("R-Cdec")
        #msg.setDetailedText("The details are as follows:")
        self.msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        self.msg.buttonClicked.connect(self.mssgbtn)

        returnValue = self.msg.exec()
        if returnValue == QtWidgets.QMessageBox.Ok:
            sys.exit()

    def next_image(self):

        if not self.image_path[0]:
            self.log.append('<p style="color: red">No Image</p>')
        else:
            self.img_index = ( self.img_index + 1 ) % len(self.image_path[0])

            # Display image no
            self.l_img_no.setText( f"{self.img_index + 1} / {len(self.image_path[0])}" )

            file_name = Path(self.image_path[0][self.img_index]).name
            self.image_name_label.setText(file_name)

            pixmap_image = QPixmap(self.image_path[0][self.img_index])
            self.l_image_thumb.setPixmap(pixmap_image.scaled(self.l_image_thumb.width(), self.l_image_thumb.height()))


    def upload_image(self):


        self.image_path = \
            QtWidgets.QFileDialog.getOpenFileNames(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.png *.jpg *.bmp')
                                                 

        #print(self.image_path[0])  # /home/my_name/Downloads/example.png
        #print(type(self.image_path)) 
        
        if not self.image_path[0]:
            self.log.append('<p style="color: red">Failed to Upload</p>')
            self.ci.setChecked(False)
            self.l_image_thumb.clear()
            self.image_name_label.clear()
            self.l_img_no.clear()
        else:
            self.ci.setChecked(True)
            self.bar.setValue(20)
            self.log.append('<p style="color: green"><b>Images Uploaded Sucessfully</b></p>')

            for img in self.image_path[0]:
                self.log.append(img)

            # Display image no
            self.l_img_no.setText( f"1 / {len(self.image_path[0])}" )

            file_name = Path(self.image_path[0][0]).name
            self.image_name_label.setText(file_name)
            #print(file_name)

            pixmap_image = QPixmap(self.image_path[0][0])
            self.l_image_thumb.setPixmap(pixmap_image.scaled(self.l_image_thumb.width(), self.l_image_thumb.height()))
            #self.log.append(f'<img src={self.image_path[0]} width="300" height="170">')
        

    def upload_weight(self):


        self.weight_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Weights to Open',
                                                  '.',
                                                  '*.weights')

        #print(self.weight_path[0])  # /home/my_name/Downloads/example.png
        

        if not self.weight_path[0]:
            self.log.append('<p style="color: red">Failed to Upload</p>')
            self.cw.setChecked(False)
        else:
            self.cw.setChecked(True)
            self.bar.setValue(40)
            self.log.append('<p style="color: green"><b>Weights Uploaded Sucessfully</b></p>')
            self.log.append(self.weight_path[0])


    def upload_cfg(self):

        self.cfg_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Configuration to Open',
                                                  '.',
                                                  '*.cfg')

        #print(self.cfg_path[0])  # /home/my_name/Downloads/example.png
        

        if not self.cfg_path[0]:
            self.log.append('<p style="color: red">Failed to Upload</p>')
            self.ccf.setChecked(False)
        else:
            self.ccf.setChecked(True)
            self.bar.setValue(60)
            self.log.append('<p style="color: green"><b>Configuration Uploaded Sucessfully</b></p>')
            self.log.append(self.cfg_path[0])

    def upload_class(self):

        self.class_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Classes to Open',
                                                  '.',
                                                  '*.names')

        #print(self.class_path[0])  # /home/my_name/Downloads/example.png
        

        if not self.class_path[0]:
            self.log.append('<p style="color: red">Failed to Upload</p>')
            self.ccl.setChecked(False)
        else:
            self.ccl.setChecked(True)
            self.bar.setValue(80)
            self.log.append('<p style="color: green"><b>Classes Uploaded Sucessfully</b></p>')
            self.log.append(self.class_path[0])

    def set_prob(self):
        self.probability, _ = QtWidgets.QInputDialog.getText(
            self, 'Probability', 'e.g 0.4') 
        
        self.bar.setValue(85)

        if not self.probability:
            self.probability = 0.4
            self.log.append(f'<p style="color: green"> Probability {self.probability}</p>')
        
        #print(self.probability)
        else:
            try:
                if 0 <= float(self.probability) <= 1:
                    self.log.append(f'<p style="color: green"> Probability {self.probability}</p>')
                else:
                    raise
            except Exception as exc:
                print(exc)
                self.log.append('<p style="color: red"> Enter number between 0 and 1</p>')
                self.probability = 0.4
    
    
    def set_thres(self):
        self.threshold, _ = QtWidgets.QInputDialog.getText(
            self, 'Threshold', 'e.g 0.9')

        self.bar.setValue(90)

        if not self.threshold:
            self.threshold = 0.9
            self.log.append(f'<p style="color: green"> Threshold {self.threshold}</p>')
        
        else:
            try:
                if 0<= float(self.threshold) <= 1:
                    self.log.append(f'<p style="color: green"> Threshold {self.threshold}</p>')
                else:
                    raise
            except Exception as exc:
                print(exc)
                self.log.append('<p style="color: red"> Enter number between 0 and 1</p>')
                self.threshold = 0.9
        #print(self.threshold)


    
    def predict(self):
        
        if self.image_path[0] and self.weight_path[0] \
            and self.cfg_path[0] and self.class_path[0]:

            try:
                self.bar.setValue(100)

                out_dir = Path.cwd() / "output"
                if out_dir.is_dir():
                    pass
                else:
                    out_dir.mkdir()
                
                

                for img in self.image_path[0]:

                    img_loc = out_dir / Path(img).name
                    
                    messages, t_box , box_nms = infer(
                        image_path = img,
                        weights_path = self.weight_path[0],
                        test_cfg_path = self.cfg_path[0],
                        class_path = self.class_path[0],
                        probability_minimum = float(self.probability),
                        threshold = float(self.threshold), 
                        img_name = str(img_loc) 
                    )
                    
                    for mssg in messages:
                        self.log.append(f'<p style="color: green"><b> {mssg} </b></p>')
                    self.log.append('')

                    new_row = pd.Series([img, t_box , box_nms, self.probability, self.threshold], index=self.df.columns)
                    self.df = self.df.append(new_row, ignore_index=True)
                
                model = pandasModel(self.df)
                self.tbl_view.setModel(model)
                #self.tbl_view.resizeRowsToContents()
                self.tbl_view.resizeColumnsToContents()
                self.tbl_view.show()
                    

                self.r_res.setEnabled(True) 
                self.pb_save.setEnabled(True)
                #print(self.r_res.isEnabled())
            except Exception as exc:
                print(exc)
                self.log.append('<p style="color: red">Failed to create "output" folder or folder not found</p>')
                self.log.append('<p style="color: red">Check folder permissions.</p>')
        
        else:
            self.log.append('<p style="color: red">Oops, Make sure you uploaded everything </p>') 


    def result(self):

        if self.r_res.setEnabled:
            try:
                # pixmap_image = QPixmap('result.jpg')
                # pixmap_image.load()
                final_img_result = \
                QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Images to Open',
                                                    './output/',
                                                    '*.png *.jpg *.bmp')

                if final_img_result[0]:
                    self.log.append('<p>Opening Image.....</p>')
                    image = Image.open(final_img_result[0])
                    image.show()
            except Exception as exc:
                print(exc)
                self.log.append('<p style="color: red">Oops, Something is wrong</p>')
    
    def save_to_excel(self):

        if self.pb_save.setEnabled:
            try:
                out_dir = Path.cwd() / "output"
                if out_dir.is_dir():
                    pass
                else:
                    out_dir.mkdir()

                _file = out_dir / f"Results-{date.today()}.csv"
                self.df.to_csv(str(_file))

                self.log.append(f'<p style="color: green"><b>{_file.name} Saved Sucessfully</b></p>')
            except Exception as exc:
                print(exc)
                self.log.append('<p style="color: red">Failed to create "output" folder or folder not found</p>')
                self.log.append('<p style="color: red">Check folder permissions.</p>')


    
    def text_box_clear(self):
        cur_tab_index = self.tabWidget.currentIndex()
        #print(cur_tab_index)
        if cur_tab_index == 0:
            self.log.clear()
        if cur_tab_index == 2:
            self.log_download.clear()

# Filter ################################################

    def filter_upload_image(self):

        self.filter_image_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.png *.jpg *.bmp')
        
        if not self.filter_image_path[0]:
            self.l_filter_img_tmb_in.clear()
            self.l_filter_name_in.clear()
            self.pb_filter_convert.setEnabled(False)
        else:
            
            file_name = Path(self.filter_image_path[0]).name
            self.l_filter_name_in.setText(file_name)

            pixmap_image = QPixmap(self.filter_image_path[0])
            self.l_filter_img_tmb_in.setPixmap(pixmap_image.scaled(self.l_filter_img_tmb_in.width(), self.l_filter_img_tmb_in.height()))

            self.pb_filter_convert.setEnabled(True)
    
    def filter_convert_image(self):

        if self.rb_grayscale.isChecked():

            if self.rb_grayscale.isChecked():

                try:
                    out_dir = Path.cwd() / "output"
                    if out_dir.is_dir():
                        pass
                    else:
                        out_dir.mkdir()

                
                    _file = out_dir / f"Grayscale-{Path(self.filter_image_path[0]).name}"
                    
                    gray_array = grayscale(
                        img_path=self.filter_image_path[0],
                        out_path=str(_file)
                        )
                    
                    pixmap_image = QPixmap(str(_file))
                    self.l_filter_img_tmb_out.setPixmap(pixmap_image.scaled(self.l_filter_img_tmb_out.width(), self.l_filter_img_tmb_out.height()))
                except Exception as exc:
                    print(exc)
                    self.log.append('<p style="color: red">Failed to create "output" folder or folder not found</p>')
                    self.log.append('<p style="color: red">Check folder permissions.</p>')


# Update #################################################
    def download_repo(self):

        def check_internet():
            try:
                urllib.request.urlopen('http://google.com') 
                return True
            except:
                return False

        self.log_download.append('Checking for Internet')
        self.log_download.append('')

       
        if check_internet():
            self.log_download.append('<p style="color: green"><b>Connection created Sucessfully</b></p>')

            try:
                url = 'https://github.com/rohank63/RCdec-latest/archive/refs/heads/main.zip'
                req = requests.get(url, stream=True)
                with open("RCdec-latest.zip",'wb') as mw: 
                    for chunk in req.iter_content(chunk_size=1024): 
                
                        # writing one chunk at a time to pdf file 
                        if chunk: 
                            mw.write(chunk) 
                
                self.log_download.append('<p style="color: green"><b>Updated Sucessfully !</b></p>')
            except Exception as exc:
                print(exc)
                self.log_download.append('<p style="color: red">Something went wrong</p>')
        else:
            self.log_download.append('<p style="color: red">No Internet! Connection Timeout</p>')
        
        self.log_download.append('')



def main():
    # Initializing instance of Qt Application
    app = QtWidgets.QApplication(sys.argv)

    # Initializing object of designed GUI
    window = MainApp()

    # Showing designed GUI
    window.show()

    # Running application
    app.exec_()


"""
End of: 
Main function
"""


# Checking if current namespace is main, that is file is not imported
if __name__ == '__main__':
    # Implementing main() function
    main()
