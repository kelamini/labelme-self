import re

from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.logger import logger
import labelme.utils


QT5 = QT_VERSION[0] == "5"


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class LabelQLineEdit(QtWidgets.QLineEdit):
    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)


class TrackDialog(QtWidgets.QDialog):
    def __init__(
        self,        
        parent=None,
    ):        

        super(TrackDialog, self).__init__(parent)
        self.edit_start = QtWidgets.QLineEdit()
        self.edit_start.setPlaceholderText('start_frame')
        self.edit_start.resize(100,20)
        self.edit_start.move(10,20)

        self.edit_end = QtWidgets.QLineEdit()
        self.edit_end.setPlaceholderText('end_frame')
        self.edit_end.resize(100,20)
        self.edit_end.move(150,20)

        layout = QtWidgets.QVBoxLayout()

        layout_edit = QtWidgets.QHBoxLayout()
        layout_edit.addWidget(self.edit_start, 6)
        layout_edit.addWidget(self.edit_end, 2)
        layout.addLayout(layout_edit)    
        self.edit_track_src = QtWidgets.QLineEdit()
        self.edit_track_src.setPlaceholderText('src_track')
        # self.edit_track_src.resize(50,20)
        # self.edit_track_src.move(10,80)

        self.edit_track_dst = QtWidgets.QLineEdit()
        self.edit_track_dst.setPlaceholderText('dst_track')
        # self.edit_track_dst.resize(50,20)
        # self.edit_track_dst.move(120,80)

        self.edit_track_label = QtWidgets.QLineEdit()
        self.edit_track_label.setPlaceholderText('label')
        # self.edit_track_label.resize(50,20)
        # self.edit_track_label.move(10,120)
        layout_edit1 = QtWidgets.QHBoxLayout()
        layout_edit1.addWidget(self.edit_track_src, 2)
        layout_edit1.addWidget(self.edit_track_dst, 2)
        layout_edit1.addWidget(self.edit_track_label, 2)
        layout.addLayout(layout_edit1)  

        # # buttons
        self.buttonBox = QtWidgets.QPushButton("process", self)
        self.buttonBox.clicked.connect(self.process_mot)

        layout_btn1 = QtWidgets.QHBoxLayout()
        self.setInvalidBox = QtWidgets.QPushButton("set_invalid", self)
        self.setInvalidBox.clicked.connect(self.set_invalid)
        self.clear_box = QtWidgets.QPushButton("clear", self)
        self.clear_box.clicked.connect(self.clear)
        layout_btn1.addWidget(self.setInvalidBox, 2)
        layout_btn1.addWidget(self.clear_box, 2)

        layout.addLayout(layout_btn1)  

        # self.btn_del = QtWidgets.QPushButton("del", self)
        # self.buttonBox.clicked.connect(self.process_mot)
        # self.buttonBox.resize(self.buttonBox.sizeHint())
        # self.buttonBox.move(150, 120)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
        self.parent = parent

    def get_text(self,edit_ctrl):
        text = edit_ctrl.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trim()
        return text
    def process_mot(self):
        frame_start = int(self.get_text(self.edit_start))
        frame_end = int(self.get_text(self.edit_end))
        src_track = self.get_text(self.edit_track_src)
        dst_track = self.get_text(self.edit_track_dst)
        label = int(self.get_text(self.edit_track_label))
        
        track_map = None
        if src_track != '':
            track_map = {}
            src_items = src_track.split(',')
            src_items = [int(v) for v in src_items]
            dst_items = dst_track.split(',')
            if len(dst_items) == 0 or dst_track == '':
                dst_items = src_items
            elif len(dst_items) == 1:
                dst_items = [int(dst_items[0])]*len(src_items)
            else:
                dst_items = [int(v) for v in dst_items]
                assert(len(src_items) == len(dst_items))

            
            for i in range(len(src_items)):
                track_map[src_items[i]] = dst_items[i]
            
        if self.parent != None:
            self.parent.process_mot(frame_start,frame_end,track_map,label)

    def set_invalid(self):
        src_track = self.get_text(self.edit_track_src)
        if self.parent != None:
            self.parent.set_invalid(src_track)

    def clear(self):
        if self.parent != None:
            self.parent.resetState()