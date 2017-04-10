from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from phys_tools.models import PatternSession
from .main_views import PsthViewWidget, COLORS


class PatternSessionWidget(QTabWidget):
    session_updated = pyqtSignal(set)
    unit_updated = pyqtSignal(list)

    def __init__(self, parent):

        super(PatternSessionWidget, self).__init__(parent)
        self._last_sessions = set()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.current_session = None
        self.stimview = StimuliViewer(self)
        self.session_updated.connect(self.stimview.update_session)
        self.spotmapview = SpotMapViewWidget(self)
        self.unit_updated.connect(self.spotmapview.update_unit_plots)
        self.addTab(self.spotmapview, 'Spot responses')
        self.addTab(self.stimview, 'Stimulus filtering')

    def sizeHint(self):
        return QSize(800, 800)

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


class SpotMapViewWidget(PsthViewWidget):

    def __init__(self, *args, **kwargs):
        super(SpotMapViewWidget, self).__init__(*args, **kwargs)
        self.plot.axis.set_axis_off()

    @pyqtSlot(list)
    def update_unit_plots(self, units):

        self.plot.axis.cla()
        self.plot.axis.set_axis_off()

        self._current_units = units
        mthd = self.method_box.currentText()
        pre, pst, bs = self.pre_pad_box.value(), self.post_pad_box.value(), self.binsize_box.value()
        for i, u in enumerate(units):
            u.plot_spots(pre, pst, bs, axis=self.plot.axis,
                         color=COLORS[i % len(COLORS)], convolve=mthd)
        self.plot.draw()