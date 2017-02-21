from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from abc import abstractmethod
from glob import glob
import os

startpath = '/Users/chris/Data/ephys_patterns/mouse_9082/sess_001'  # todo: needs to be moved to config or something.
COLORS = get_cmap('Vega10').colors


class MainWindow(QMainWindow):
    sessions_opened = pyqtSignal(list)

    def __init__(self, session_model_type, SessionViewType, windowname=''):
        """
        Makes window to display unit and session widgets. Both view and models are customized to match
        different data found in different sessions using the two parameters.

        :param session_model_type: Session model class (ie OdorSession, PatternSession, etc.)
        :param session_view_type: Session view QWidget class (ie OdorSessionWidget, PatterSessionWidget)
        """
        super(MainWindow, self).__init__()
        menu = self.menuBar()
        menu.setNativeMenuBar(False)
        self.move(30, 30)
        self.make_menu(menu)
        self.mainwidget = MainWidgetEphys(self, session_model_type, SessionViewType)
        sbar = self.statusBar()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sessions_opened.connect(self.mainwidget.open_sessions)
        self.setCentralWidget(self.mainwidget)
        self.setWindowTitle(windowname)
        self.resize(1400, 900)

    def make_menu(self, menu):
        """makes menu items"""
        filemenu = menu.addMenu('&File')
        open_action = QAction('&Open...', self)
        open_action.setStatusTip('Open a single session.')
        open_action.triggered.connect(self.open_sessions)
        filemenu.addAction(open_action)
        open_folder_action = QAction('Open &folder...', self)
        open_folder_action.setStatusTip('Open all sessions within a folder recursively.')
        open_folder_action.triggered.connect(self.open_folder)
        filemenu.addAction(open_folder_action)

    @pyqtSlot()
    def open_sessions(self):
        filepaths, _ = QFileDialog.getOpenFileNames(self,
                                                    "Select one or more files to open",
                                                    startpath)
        self.sessions_opened.emit(filepaths)

    @pyqtSlot()
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(parent=self, caption='Select a folder or folders to open.',
                                                        directory=startpath)

        #TODO: enable selection of multiple
        filepaths = glob(os.path.join(folder, '**/*.dat'), recursive=True)
        self.sessions_opened.emit(filepaths)


class MainWidgetEphys(QWidget):
    units_selected = pyqtSignal(list)
    update_unit_list = pyqtSignal(dict)

    def __init__(self, parent, session_model_type, SessionViewType):
        super(MainWidgetEphys, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._SessionModelType = session_model_type
        self.session_models = {}
        self.selected_session_models = []
        self.selected_session_units = {}  # cache these for quick, hashed access.
        self.setBaseSize(self.sizeHint())

        main_layout = QHBoxLayout()
        selector_layout = QVBoxLayout()
        main_layout.addLayout(selector_layout)

        self.unit_list_widget = UnitListWidget(self)
        self.update_unit_list.connect(self.unit_list_widget.update_list)
        self.unit_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.unit_list_widget.itemSelectionChanged.connect(self.unit_selection_changed)

        self.session_selector = QListWidget(self,)
        self.session_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.session_selector.setMaximumHeight(150)  #todo: set size based on width of rows not pixels
        self.session_selector.itemSelectionChanged.connect(self.session_selection_changed)
        self.session_selector.setSelectionMode(QListWidget.ExtendedSelection)
        self.session_selector.setMaximumWidth(125)

        unit_filter_layout = QHBoxLayout()
        unit_filter_label = QLabel('Min. unit rating: ')
        self.unit_filter_spinbox = QSpinBox(self)
        self.unit_filter_spinbox.setMaximum(5)
        self.unit_filter_spinbox.setValue(2)
        self.unit_filter_spinbox.valueChanged.connect(self.update_units)
        unit_filter_layout.addWidget(unit_filter_label)
        unit_filter_layout.addWidget(self.unit_filter_spinbox,)

        selector_layout.addWidget(QLabel('Sessions:'))
        selector_layout.addWidget(self.session_selector)
        selector_layout.addWidget(QLabel('Units:'))
        selector_layout.addWidget(self.unit_list_widget)
        selector_layout.addLayout(unit_filter_layout)

        self.session_widget = SessionViewType(self)
        self.units_selected.connect(self.session_widget.update_units)

        data_layout = QVBoxLayout()
        data_layout.addWidget(self.session_widget)
        self.unit_widget = UnitWidget(self)
        self.units_selected.connect(self.unit_widget.update_unit)
        data_layout.addWidget(self.unit_widget)
        main_layout.addLayout(data_layout)
        self.setLayout(main_layout)

    @pyqtSlot(list)
    def open_sessions(self, filepaths):
        for f in filepaths:
            try:
                s = self._SessionModelType(f)
                s_str = str(s)
                self.session_models[s_str] = s
                s_item = QListWidgetItem(s_str, self.session_selector)
                s_item.setSelected(True)
            except Exception as e:
                print("File cannot be opened: {}.".format(f, e))
                print(e)

    @pyqtSlot()
    def session_selection_changed(self):
        selected_items = self.session_selector.selectedItems()
        self.selected_session_models = [self.session_models[x.text()] for x in selected_items]
        self.update_units(self.unit_filter_spinbox.value())

    @pyqtSlot(int)
    def update_units(self, filt):
        self.selected_session_units.clear()
        for s in self.selected_session_models:
            for u in s.units_gte(filt):
                self.selected_session_units[str(u)] = u
        self.update_unit_list.emit(self.selected_session_units)

    @pyqtSlot()
    def unit_selection_changed(self):
        selected_items = self.unit_list_widget.selectedItems()
        selected_units = [self.selected_session_units[x.text()] for x in selected_items]
        self.units_selected.emit(selected_units)


class UnitListWidget(QListWidget):
    def __init__(self, parent):
        self.unit_models = []
        super(UnitListWidget, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setSelectionMode(self.ExtendedSelection)
        self.setMaximumWidth(125)

    @pyqtSlot(dict)
    def update_list(self, selected_units):
        unit_strings = list(selected_units.keys())
        unit_strings.sort()
        self.clear()
        self.addItems(unit_strings)


class UnitWidget(QTabWidget):
    units_updated = pyqtSignal(list)

    def __init__(self, parent):
        super(UnitWidget, self).__init__(parent)
        self.units = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(QSize(500,400))

        stats_widget = UnitCharacteristicsWidget(self)
        self.units_updated.connect(stats_widget.update_units)
        self.addTab(stats_widget, 'Statistics')

        response_widget = ResponseWidget(self)
        self.addTab(response_widget, 'Response')

    @pyqtSlot(list)
    def update_unit(self, units):
        self.units = units
        self.units_updated.emit(units)


class UnitCharacteristicsWidget(QWidget):
    def __init__(self, parent):
        super(UnitCharacteristicsWidget, self).__init__(parent)
        main_layout = QHBoxLayout(self)
        self.autocor_widget = UnitCharacteristicPlots(self)
        main_layout.addWidget(self.autocor_widget)

    @pyqtSlot(list)
    def update_units(self, units):
        self.autocor_widget.update_plots(units)


class UnitCharacteristicPlots(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""


    def __init__(self, parent=None, width=5, height=4, dpi=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super(UnitCharacteristicPlots, self).__init__(fig)
        self.acor_axes = fig.add_subplot(121)
        self.temp_axes = fig.add_subplot(122)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def update_plots(self, units):
        self.acor_axes.cla()
        self.temp_axes.cla()
        for i, u in enumerate(units):
            c = COLORS[i % len(COLORS)]
            u.plot_autocorrelation(axis=self.acor_axes, color=c)
            u.plot_template(axis=self.temp_axes, color=c)
        self.draw()

class ResponseWidget(QWidget):
    """
    plotting of unit responses.
    """

    def __init__(self, parent):
        super(ResponseWidget, self).__init__(parent)
        main_layout = QHBoxLayout()

        controls_layout = QVBoxLayout()
        pre_pad_label = QLabel('Pre padding (ms)')
        pre_pad_box = QSpinBox(self)

        post_pad_label = QLabel('Post padding (ms)')
        post_pad_box = QSpinBox(self)


        controls_layout.addWidget(pre_pad_label)
        controls_layout.addWidget(pre_pad_box)
        controls_layout.addWidget(post_pad_label)
        controls_layout.addWidget(post_pad_box)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        controls_layout.setSizeConstraint(controls_layout.SetFixedSize)
        main_layout.addStretch()
        self.setLayout(main_layout)

    @pyqtSlot()
    def update_response(self):
        pass


class PsthViewWidget(QWidget):
    """
    builds gui items for a typical psth display.
    The update_unit_plots method needs to be writen for the display you're interested in!
    """

    def __init__(self, parent):
        self._current_units = []
        super(PsthViewWidget, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        self.plot = _PsthPlotCanvas(self)
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

        method_label = QLabel('Binning method')
        method_box = QComboBox(self)
        method_box.addItems(('gaussian', 'boxcar', 'histogram'))
        method_box.setInsertPolicy(QComboBox.NoInsert)
        method_box.activated.connect(self.plot_param_changed)
        self.method_box = method_box
        # todo: make controls hideable.
        controls_layout.addWidget(pre_pad_label)
        controls_layout.addWidget(pre_pad_box)
        controls_layout.addWidget(post_pad_label)
        controls_layout.addWidget(post_pad_box)
        controls_layout.addWidget(binsize_label)
        controls_layout.addWidget(binsize_box)
        controls_layout.addWidget(method_label)
        controls_layout.addWidget(method_box)
        controls_layout.addStretch()
        self.controls_layout = controls_layout

        layout.addLayout(controls_layout)

    @pyqtSlot(int)
    def plot_param_changed(self, _):
        self.update_unit_plots(self._current_units)

    @pyqtSlot(list)
    @abstractmethod
    def update_unit_plots(self, units):

        raise NotImplementedError('This method needs to be overwritten with logic to plot psths.')
        # self.plot.clr()
        # self._current_units = units
        # mthd = self.method_box.currentText()
        # pre, pst, bs = self.pre_pad_box.value(), self.post_pad_box.value(), self.binsize_box.value()
        # for i, u in enumerate(units):
        #     u.plot_spots(pre, pst, bs, self.plot.axis,
        #                  color=COLORS[i % len(COLORS)], convolve=mthd)
        # self.plot.draw()


class _PsthPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # fig, axes = subplots(6,6,figsize=(width, height), dpi=dpi, sharex=True, sharey=True)
        # self.axis = axes  ### DOING SUBPLOTS FOR THIS IS MASSIVELY SLOW!!
        fig = Figure(figsize=(width, height))
        super(_PsthPlotCanvas, self).__init__(fig)
        self.axis = fig.add_subplot(111)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clr(self):
        self.axis.cla()

def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()