from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from phys_tools.models import OdorSession, OdorUnit
from .main_views import PsthViewWidget, COLORS

startpath = '/Users/chris/Data/ephys/stanford_collab/mouse_8521/sess_003'

class OdorSessionWidget(QWidget):
    session_updated = pyqtSignal(set)
    unit_updated = pyqtSignal(list)

    def __init__(self, parent):

        super(OdorSessionWidget, self).__init__(parent)
        self._last_sessions = set()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.current_session = None
        self.stimview = StimuliViewer(self)
        self.session_updated.connect(self.stimview.update_session)
        self.odorpsthview = OdorPsthViewWidget(self)
        self.unit_updated.connect(self.odorpsthview.update_unit_plots)
        self.stimview.stim_changed.connect(self.odorpsthview.update_stim_plots)
        layout = QHBoxLayout(self)
        layout.addWidget(self.stimview)
        layout.addWidget(self.odorpsthview)


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
    stim_changed = pyqtSignal(set)
    def __init__(self, parent):
        super(StimuliViewer, self).__init__(parent)
        self._last_session = None
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        self.stim_list = QTreeWidget(self)
        self.stim_list.setMaximumWidth(300)
        self.stim_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.stim_list)
        self.concs_by_odor_dict = {}
        self.stim_list.itemChanged.connect(self.odor_selection_changed)
        self.selected = set()
        self._updating_stim_list = False  # guard for itemChanged when adding items.
        self.setMaximumWidth(300)
        self.current_odor_set = set()

    @pyqtSlot(list)
    def update_session(self, session_set):
        """

        :param sessions: list of selected units.
        :return:
        """
        self._updating_stim_list = True
        new_odor_set = set()
        new_odor_concs = dict()

        for s in session_set:  # type: OdorSession
            for o, v in s.concentrations_by_odor.items():
                for c in v:
                    c_str = '{:0.1e}'.format(c)
                    new_odor_set.add((o, c_str))
                    new_odor_concs[(o, c_str)] = c
        # add odors to the odor list view that are not there yet.
        for o, c_str in new_odor_set.difference(self.current_odor_set):
            if o not in self.concs_by_odor_dict.keys():
                self.concs_by_odor_dict[o] = {}
                odor_item = QTreeWidgetItem(self.stim_list, [o])
                odor_item.setExpanded(True)
            else:
                odor_item = self.stim_list.findItems(o, Qt.MatchFixedString, 0)[0]
            c_item = QTreeWidgetItem(odor_item, [c_str])
            c_item.setFlags(c_item.flags() | Qt.ItemIsUserCheckable)
            c_item.setCheckState(0, Qt.Unchecked)
            c = new_odor_concs[(o, c_str)]
            self.concs_by_odor_dict[o][c_str] = c
            self.current_odor_set.add((o, c_str))
        # remove concentrations that are not represented in the current sessions.
        for o, c_str in self.current_odor_set.difference(new_odor_set):
            _ = self.concs_by_odor_dict[o].pop(c_str)
            conc_items = self.stim_list.findItems(c_str, Qt.MatchFixedString | Qt.MatchRecursive, 0)
            for c_item in conc_items:  # type: QTreeWidgetItem
                if c_item.parent().text(0) == o:
                    c_item.parent().removeChild(c_item)
                    del c_item
            self.current_odor_set.discard((o, c_str))
        # cleanup unused odor nodes.
        _to_pop = []
        for o, v in self.concs_by_odor_dict.items():
            if not len(v):
                _to_pop.append(o)
                odor_item = self.stim_list.findItems(o, Qt.MatchFixedString, 0)[0]  # type: QTreeWidgetItem
                self.stim_list.invisibleRootItem().removeChild(odor_item)
                del odor_item
        for o in _to_pop:  # so that we don't change the size of the dict during iteration.
            self.concs_by_odor_dict.pop(o)
        self._updating_stim_list = False

    @pyqtSlot(QTreeWidgetItem, int)
    def odor_selection_changed(self, item:QTreeWidgetItem, i):
        if not self._updating_stim_list:
            odr = item.parent().text(0)
            cstr = item.text(0)
            c = self.concs_by_odor_dict[odr][cstr]
            o_c = (odr, c)
            if item.checkState(0):
                self.selected.add(o_c)
            elif not item.checkState(0) and o_c in self.selected:
                self.selected.remove(o_c)
            self.stim_changed.emit(self.selected)


class OdorPsthViewWidget(PsthViewWidget):

    def __init__(self, parent):
        super(OdorPsthViewWidget, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stim_by_odor = dict()
        self._odor_colors = dict()
        self._i_color = 0
        self.post_pad_box.setValue(500)
        self.binsize_box.setValue(70)


    @pyqtSlot(set)
    def update_stim_plots(self, stimset):
        self.stim_by_odor.clear()
        for i, (o, c) in enumerate(stimset):
            if o not in self.stim_by_odor:
                self.stim_by_odor[o] = []
            self.stim_by_odor[o].append(c)
            if o not in self._odor_colors.keys():
                self._odor_colors[o] = COLORS[self._i_color % len(COLORS)]
                self._i_color += 1
        self.update_unit_plots()

    @pyqtSlot(list)
    def update_unit_plots(self, units=None):
        self.plot.clr()
        if units is not None:
            self._current_units = units
        else:
            units = self._current_units
        mthd = self.method_box.currentText()
        pre, pst, bs = self.pre_pad_box.value(), self.post_pad_box.value(), self.binsize_box.value()

        for i, u in enumerate(units):  # type: OdorUnit
            for o, cs in self.stim_by_odor.items():
                color = self._odor_colors[o]
                for c in cs:
                    u.plot_odor_psth(o, c, pre, pst, bs, self.plot.axis, color=color, convolve=mthd)
        self.plot.draw()
