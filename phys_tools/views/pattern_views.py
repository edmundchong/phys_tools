from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from phys_tools.models import polygon
from .main_views import PsthViewWidget, COLORS


class PatternSessionWidget(QWidget):
    session_updated = pyqtSignal(set)
    stimuli_updated = pyqtSignal(set)
    plot_params_updated = pyqtSignal(list, set)

    def __init__(self, parent):
        super(PatternSessionWidget, self).__init__(parent)
        self.filter = StimFilterWidget(self)
        self.filter.filter_updated.connect(self.update_plot_stims)
        self.plotted_units = None
        self._last_sessions = set()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.current_session = None
        self.spotmapview = SpotMapViewWidget(self)
        self.plot_params_updated.connect(self.spotmapview.update_unit_plots)
        layout = QHBoxLayout(self)
        layout.addWidget(self.spotmapview)
        self.stimuli_updated.connect(self.filter.update_stims)
        self.plotted_stims = set()
        self.filter.show()  # TODO: make ui callable.
        self.filter.raise_()


    def sizeHint(self):
        return QSize(800, 800)

    @pyqtSlot(list)
    def update_units(self, selected_units):
        self.plotted_units = selected_units
        sessions = set([x.session for x in selected_units])  #TODO: optimize.
        if sessions != self._last_sessions:  # only update sessions if they really are different.
            self.session_updated.emit(sessions)
            stims = set()
            for s in sessions:  # type: polygon.PatternSession
                for stim in s.sequence_dict.keys():  # type: polygon.FrameSequence
                    stims.add(stim)
            if stims != self.plotted_stims:
                self.plotted_stims = stims
                self.stimuli_updated.emit(self.plotted_stims)
        self._last_sessions = sessions
        self._update_plots()


    @pyqtSlot(set)
    def update_plot_stims(self, stims):
        self.plotted_stims = stims
        self._update_plots()

    def _update_plots(self):
        if len(self.plotted_units) and len(self.plotted_stims):
            self.plot_params_updated.emit(self.plotted_units, self.plotted_stims)


class SpotMapViewWidget(PsthViewWidget):

    def __init__(self, *args, **kwargs):
        super(SpotMapViewWidget, self).__init__(*args, **kwargs)
        self._stims = None
        self.plot.axis.set_axis_off()

    @pyqtSlot(list, set)
    def update_unit_plots(self, units, stims=None):
        if stims is None:
            stims = self._stims
        else:
            self._stims = stims
        self.plot.axis.cla()
        self.plot.axis.set_axis_off()
        self.current_units = units
        mthd = self.method_box.currentText()
        pre, pst, bs = self.pre_pad_box.value(), self.post_pad_box.value(), self.binsize_box.value()
        if stims:
            for i, u in enumerate(units):
                u.plot_spots(pre, pst, bs, sequences=stims, axis=self.plot.axis,
                             color=COLORS[i % len(COLORS)], convolve=mthd)
        self.plot.draw()


class StimFilterWidget(QWidget):

    filter_updated = pyqtSignal(set)
    update_lists = pyqtSignal(set)

    def __init__(self, parent):
        super(StimFilterWidget, self).__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle('Stimulus filter')
        self.filtered_stims = set()
        self.all_stims = set()
        layout = QHBoxLayout(self)
        self.intensity_list = IntensityFilterList(self)
        layout.addWidget(self.intensity_list)
        self.update_lists.connect(self.intensity_list.update_list)
        self.intensity_list.itemSelectionChanged.connect(self.filter)

    @pyqtSlot(set)
    def update_stims(self, stims: set):
        new_stims = stims.difference(self.all_stims)
        self.update_lists.emit(new_stims)
        self.all_stims |= new_stims
        self.filter()

    @pyqtSlot()
    def filter(self):
        filtered = set()  # todo: move to diff model instead of rebuilding the set every time.
        intensies = self.intensity_list.selected_values()
        for s in self.all_stims:  # type: polygon.FrameSequence
            if s.intensities[0][0] in intensies:
                filtered.add(s)
        if filtered != self.filtered_stims:
            self.filtered_stims = filtered
            self.filter_updated.emit(filtered)


class FilterList(QListWidget):

    def __init__(self, parent):
        super(FilterList, self).__init__(parent)
        self.values = set()
        self.setSortingEnabled(True)
        self.setSelectionMode(self.ExtendedSelection)
        # self.itemSelectionChanged.connect(self.selected_values)

    @pyqtSlot()
    def selected_values(self):
        vals = set()
        for i in self.selectedItems():  #type: QListWidgetItem
            vals.add(i.data(Qt.UserRole))
        return vals

    @pyqtSlot(set)
    def update_list(self, stims):
        """
        to be replaced by implementation
        :param stims: set of stimuli to extract values from.
        :return: 
        """
        raise NotImplementedError


class IntensityFilterList(FilterList):

    @pyqtSlot(set)
    def update_list(self, stims):
        for stim in stims:  # type: polygon.FrameSequence
            stim_i = stim.intensities[0][0]
            if stim_i not in self.values:
                self.values.add(stim_i)
                v_str = "{:0.1f}".format(stim_i)
                item = QListWidgetItem(v_str, self)
                item.setData(Qt.UserRole, stim_i)
                self.addItem(item)

        if self.count() == 1:  # select the only value present in the filter by default.
            item.setSelected(True)
        elif not self.selectedItems():
            item = self.item(self.count() - 1)  # if more than one value, set the max item only.
            item.setSelected(True)
