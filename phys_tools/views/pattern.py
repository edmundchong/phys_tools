import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from phys_tools.models import PatternSession
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap, subplots
import numpy as np

COLORS = get_cmap('Vega10').colors


class PatternSessionWidget(QTabWidget):
    session_updated = pyqtSignal(set)
    unit_updated = pyqtSignal(list)

    def __init__(self, parent):

        super(PatternSessionWidget, self).__init__(parent)
        self._last_sessions = set()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(QSize(800, 300))
        self.current_session = None
        self.stimview = StimuliViewer(self)
        self.session_updated.connect(self.stimview.update_session)
        self.spatialmapview = SpatialMapViewer(self)
        self.unit_updated.connect(self.spatialmapview.update_units)
        self.addTab(self.spatialmapview, 'Spot responses')
        self.addTab(self.stimview, 'Stimulus filtering')



    @pyqtSlot(list)
    def update_units(self, selected_units):
        #todo: this should only be called if the session is actually updated. If just a unit, don't call.
        sessions = set([x.session for x in selected_units])
        if sessions != self._last_sessions:  # only update sessions if they really are different.
            self.session_updated.emit(sessions)
        self.unit_updated.emit(selected_units)
        self._last_sessions = sessions


class StimuliViewer(QWidget):
    def __init__(self, parent):
        super(StimuliViewer, self).__init__(parent)
        self._last_session = None
        layout = QHBoxLayout(self)
        self.stim_list = QListWidget(self)
        self.stim_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.stim_list)

    @pyqtSlot(list)
    def update_session(self, session_set):
        """

        :param sessions: list of selected units.
        :return:
        """
        if len(session_set) == 0:
            pass
        elif len(session_set) == 1:
            self.current_session = session_set.pop()
            assert type(self.current_session) == PatternSession
            if self.current_session != self._last_session:
                self.stim_list.clear()
                for i in self.current_session.sequence_dict.keys():
                    self.stim_list.addItem(str(i))
        else:
            pass



class SpatialMapViewer(QWidget):
    def __init__(self, parent):
        self._current_units = []
        super(SpatialMapViewer, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        self.plot = SpatialMapPlot(self)
        layout.addWidget(self.plot)

        controls_layout = QVBoxLayout()
        pre_pad_label = QLabel('Pre plot (ms)')
        pre_pad_box = QSpinBox(self)
        pre_pad_box.setSingleStep(20)
        pre_pad_box.setRange(0, 2000)
        pre_pad_box.setValue(100)
        pre_pad_box.valueChanged.connect(self.plot_param_changed)
        self.pre_pad_box = pre_pad_box

        post_pad_label = QLabel('Post plot (ms)')
        post_pad_box = QSpinBox(self)
        post_pad_box.setSingleStep(20)
        post_pad_box.setRange(0, 2000)
        post_pad_box.setValue(200)
        post_pad_box.valueChanged.connect(self.plot_param_changed)
        self.post_pad_box = post_pad_box

        binsize_label = QLabel('Binsize (ms)')
        binsize_box = QSpinBox(self)
        binsize_box.setSingleStep(1)
        binsize_box.setRange(1, 200)
        binsize_box.setValue(20)
        binsize_box.valueChanged.connect(self.plot_param_changed)
        self.binsize_box = binsize_box

        controls_layout.addWidget(pre_pad_label)
        controls_layout.addWidget(pre_pad_box)
        controls_layout.addWidget(post_pad_label)
        controls_layout.addWidget(post_pad_box)
        controls_layout.addWidget(binsize_label)
        controls_layout.addWidget(binsize_box)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)


    @pyqtSlot(int)
    def plot_param_changed(self, _):
        self.update_units(self._current_units)

    @pyqtSlot(list)
    def update_units(self, units):
        self.plot.clr()
        self._current_units = units
        pre, pst, bs = self.pre_pad_box.value(), self.post_pad_box.value(), self.binsize_box.value()
        for i, u in enumerate(units):
            u.plot_spots(pre, pst, bs, self.plot.axis,
                         color=COLORS[i % len(COLORS)])
        self.plot.draw()


class SpatialMapPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # fig, axes = subplots(6,6,figsize=(width, height), dpi=dpi, sharex=True, sharey=True)
        # self.axis = axes  ### DOING SUBPLOTS FOR THIS IS MASSIVELY SLOW!!
        fig = Figure(figsize=(width, height))
        super(SpatialMapPlot, self).__init__(fig)
        self.axis = fig.add_subplot(111)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clr(self):
        self.axis.cla()