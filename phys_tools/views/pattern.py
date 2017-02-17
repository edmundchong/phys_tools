from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from phys_tools.models import PatternSession
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap, subplots
# import seaborn as sns
# import pyqtgraph as pg
import time as time

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
        super(SpatialMapViewer, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        self.plot = SpatialMapPlot(self)
        layout.addWidget(self.plot)

    @pyqtSlot(list)
    def update_units(self, units):
        self.plot.clr()
        t = time.time()
        for i, u in enumerate(units):
            u.plot_spots(100, 200, 25, self.plot.axis, color=COLORS[i % len(COLORS)])
        t2 = time.time()
        print(t2 - t)
        self.plot.draw()
        print(time.time()-t2)

class SpatialMapPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # fig, axes = subplots(6,6,figsize=(width, height), dpi=dpi, sharex=True, sharey=True)
        fig = Figure(figsize=(width, height))
        super(SpatialMapPlot, self).__init__(fig)
        self.axis = fig.add_subplot(111)
        self.axis.plot([843,434525])
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clr(self):
        self.axis.cla()

# class SpatialMapPlot(pg.PlotWidget):
#
#     def __init__(self, parent):
#
#         super(SpatialMapPlot, self).__init__(parent, useOpenGL=False)
#         self.axis = self
#
#
#
#
#     def clr(self):
#         self.clear()
#         self.disableAutoRange()
#
#     def draw(self):
#         self.enableAutoRange()


