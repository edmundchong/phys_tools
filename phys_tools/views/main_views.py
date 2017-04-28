from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from abc import abstractmethod
from glob import glob
import os
import sip
from phys_tools.models.base import Session, Unit
from phys_tools.utils.json import save_units_json, load_units_json

startpath = '/Users/chris/Data/'  # todo: needs to be moved to config or something.
COLORS = plt.get_cmap('Vega10').colors
LISTWIDTH = 125


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
        # menu.setNativeMenuBar(False)
        self.move(30, 30)
        self.mainwidget = MainWidgetEphys(self, session_model_type, SessionViewType)
        self.unit_sub_dock = QDockWidget('Subset Editor', self)
        self.unit_subset_widget = UnitSubsetWidget()
        self.unit_sub_dock.setWidget(self.unit_subset_widget)
        self.unit_sub_dock.hide()
        self.unit_sub_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.mainwidget.unit_list_widget.addtosubset.connect(self.unit_subset_widget.list.addKeyEvent)
        self.unit_subset_widget.list.selection_changed_list.connect(
            self.mainwidget.unit_list_widget.select_external
        )

        self.addDockWidget(Qt.RightDockWidgetArea, self.unit_sub_dock)
        self.make_menu(menu)

        sbar = self.statusBar()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sessions_opened.connect(self.mainwidget.load_files)
        self.setCentralWidget(self.mainwidget)
        self.setWindowTitle(windowname)
        self.resize(1400, 900)

    def make_menu(self, menu):
        """makes menu items"""
        filemenu = menu.addMenu('&File')
        open_action = QAction('&Open...', self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.setStatusTip('Open a single session.')
        open_action.triggered.connect(self.open_sessions)
        filemenu.addAction(open_action)
        open_folder_action = QAction('Open &folder...', self)
        open_folder_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_folder_action.setStatusTip('Open all sessions within a folder recursively.')
        open_folder_action.triggered.connect(self.open_folder)
        filemenu.addAction(open_folder_action)

        #full screen
        fullscreen_Action = QAction("Toggle full screen...", self)
        fullscreen_Action.triggered.connect(self.toggle_screen)
        fullscreen_Action.setStatusTip("Toggle full screen.")
        fullscreen_Action.setShortcut("Ctrl+E")
        filemenu.addAction(fullscreen_Action)

        toolmenu = menu.addMenu('Subset')
        act = self.unit_sub_dock.toggleViewAction()  # type: QAction
        act.setText('Open subset list')
        act.setShortcut(QKeySequence('Ctrl+Shift+S'))
        toolmenu.addAction(act)
        subset_save = QAction('Save subset...', self)
        subset_save.setShortcut("Ctrl+S")
        subset_save.triggered.connect(self.unit_subset_widget.save_list)
        toolmenu.addAction(subset_save)

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


    def toggle_screen(self):
        if (self.windowState() == Qt.WindowMaximized):
            self.showNormal()
        else:
            self.showMaximized()





class MainWidgetEphys(QWidget):
    units_selected = pyqtSignal(list)
    update_unit_list = pyqtSignal(dict)

    def __init__(self, parent, session_model_type: Session, SessionViewType):
        super(MainWidgetEphys, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._SessionModelType = session_model_type
        self.session_models = {}
        self.unit_models = {}  # container for all units in opened sessions. cache these for quick, hashed access
        self.selected_session_models = []
        self.selected_session_units = {}  # subset of only units in selected sessions
        self.setBaseSize(self.sizeHint())

        main_layout = QHBoxLayout()
        selector_layout = QVBoxLayout()
        main_layout.addLayout(selector_layout)

        self.unit_list_widget = UnitListWidget(self)
        self.update_unit_list.connect(self.unit_list_widget.update_list)
        self.unit_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.unit_list_widget.itemSelectionChanged.connect(self.unit_selection_changed)
        self.unit_list_widget.stability.connect(self.show_stability)

        self.session_selector = QListWidget(self,)
        self.session_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.session_selector.setMaximumHeight(150)  #todo: set size based on width of rows not pixels
        self.session_selector.itemSelectionChanged.connect(self.session_selection_changed)
        self.session_selector.setSelectionMode(QListWidget.ExtendedSelection)
        self.session_selector.setMaximumWidth(LISTWIDTH)

        unit_filter_layout = QHBoxLayout()
        unit_filter_layout.setSpacing(0)
        unit_filter_label = QLabel('Min. unit rating: ')
        self.unit_filter_spinbox = QSpinBox(self)
        self.unit_filter_spinbox.setMaximum(5)
        self.unit_filter_spinbox.setValue(3)
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
        self.unit_widget = UnitTabWidget(self)
        self.units_selected.connect(self.unit_widget.update_unit)
        data_layout.addWidget(self.unit_widget)
        main_layout.addLayout(data_layout)
        self.setLayout(main_layout)

    @pyqtSlot(list)
    def load_files(self, filepaths):
        errors = 0
        self.parent().setStatusTip('Opening {} files...'.format(len(filepaths)))
        for f in filepaths:
            if os.path.splitext(f)[-1].lower() == '.json':
                sessions = load_units_json(f)
                for s in sessions:
                    s_str = str(s)
                    self.session_models[s_str] = s
                    s_item = QListWidgetItem(s_str, self.session_selector)
                    s_item.setSelected(True)
                    for u in s.units():
                        self.unit_models[str(u)] = u
            else:
                try:
                    s = self._SessionModelType(f)  #type: Session
                    s_str = str(s)
                    self.session_models[s_str] = s
                    s_item = QListWidgetItem(s_str, self.session_selector)
                    s_item.setSelected(True)
                    for u in s.units():
                        self.unit_models[str(u)] = u
                except Exception as e:
                    print("File cannot be opened: {}.\n {}".format(f, e))
                    errors += 1
        if errors:
            self.parent().setStatusTip(
                '{} of {} files could not be opened due to errors. Check log.'.format(errors, len(filepaths))
            )
        else:
            self.parent().setStatusTip('Complete.')

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

    @pyqtSlot(QListWidgetItem)
    def show_stability(self, item):
        unit = self.selected_session_units[item.text()]
        stability_dialog = StabilityWidget(self, unit)
        stability_dialog.show()

class UnitListWidget(QListWidget):
    addtosubset = pyqtSignal(list)
    stability = pyqtSignal(QListWidgetItem)

    def __init__(self, parent):
        self.unit_models = []
        super(UnitListWidget, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setSelectionMode(self.ExtendedSelection)
        self.setMaximumWidth(LISTWIDTH)
        self.setDragDropMode(QListWidget.DragOnly)
        self.setStatusTip("Press 's' to add unit to subset")
        self.itemActivated.connect(self._itemActivated)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context)

    @pyqtSlot(dict)
    def update_list(self, selected_units: dict):
        unit_strings = list(selected_units.keys())
        unit_strings.sort()
        self.clear()
        self.addItems(unit_strings)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_S:
            self.addtosubset.emit(self.selectedItems())
        else:
            super(UnitListWidget, self).keyPressEvent(event)

    @pyqtSlot(list)
    def select_external(self, item_list: list):
        self.clearSelection()
        for it in item_list:
            myitem = self.findItems(it.text(), Qt.MatchFixedString)[0]  # type: QListWidgetItem # this should always be true.
            if not myitem.isSelected():
                myitem.setSelected(True)

    @pyqtSlot(QListWidgetItem)
    def _itemActivated(self, item):
        self.addtosubset.emit([item])

    @pyqtSlot(QPoint)
    def _context(self,pos):
        item = self.currentItem()
        if item:
            self._currentitem = item
            pos = self.mapToGlobal(pos)

            menu = QMenu()
            menu.addAction("Stability", self._stability)
            menu.exec(pos)

    @pyqtSlot()
    def _stability(self):
        item = self.currentItem()
        print(item)
        self.stability.emit(item)


class UnitSubsetWidget(QWidget):
    def __init__(self, parent=None):
        super(UnitSubsetWidget, self).__init__(parent)
        self.list = UnitSubsetList(self)
        self.savebutton = QPushButton('Save', self)
        self.savebutton.clicked.connect(self.save_list)
        self.clearbutton = QPushButton('Clear', self)
        self.clearbutton.clicked.connect(self.list.clear)
        mainlayout = QVBoxLayout(self)
        buttonlayout = QHBoxLayout()
        buttonlayout.setSpacing(0)
        buttonlayout.addWidget(self.savebutton)
        buttonlayout.addWidget(self.clearbutton)
        mainlayout.addWidget(self.list)
        mainlayout.addLayout(buttonlayout)

    @pyqtSlot()
    def save_list(self):
        if not self.list.count():
            self.parent().parent().setStatusTip('No units in subset, please add before saving list.')
        else:
            filename, _ = QFileDialog.getSaveFileName(
                self, "select a save path", startpath, filter='JSON (*.json)'
            )
            if not filename:
                return
            p = self.parent().parent() #type: MainWindow
            all_units = p.mainwidget.unit_models
            units_to_save = []
            n_units = self.list.count()

            for i in range(n_units):
                item = self.list.item(i)  #type: QListWidgetItem
                name = item.text()
                units_to_save.append(all_units[name])

            save_units_json(filename, units_to_save)
            self.parent().parent().setStatusTip('Save complete.')

    def closeEvent(self, event):
        self.hide()


class UnitSubsetList(QListWidget):
    selection_changed_list = pyqtSignal(list)

    def __init__(self, parent):
        super(UnitSubsetList, self).__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setToolTip('Drag units here from main list.')
        self.itemSelectionChanged.connect(self._selection_changed)
        self.setMaximumWidth(LISTWIDTH)

    def dropEvent(self, event: QDropEvent):
        """
        Reimplementation to guard against adding duplicate members to the list and prevent dropping into the
        list from itself.
        """
        source = event.source()  # type: QListWidget
        if source == self:
            return
        else:
            items = source.selectedItems()
            for item in items:
                if not self.findItems(item.text(), Qt.MatchFixedString):
                    event.accept()
                    self.addItem(item.text())

    @pyqtSlot(list)
    def addKeyEvent(self, items):
        for item in items:
            if not self.findItems(item.text(), Qt.MatchFixedString):
                self.addItem(item.text())

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for it in self.selectedItems():  #type: QListWidgetItem
                sip.delete(it)

    @pyqtSlot()
    def _selection_changed(self):
        self.selection_changed_list.emit(self.selectedItems())


class UnitTabWidget(QTabWidget):
    """Tabbed widget for display of unit information."""
    update_graphs = pyqtSignal(list)

    def __init__(self, parent):
        super(UnitTabWidget, self).__init__(parent)
        self.units = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(QSize(500,400))

        stats_widget = UnitCharacteristicsWidget(self)
        self.update_graphs.connect(stats_widget.update_units)
        self.addTab(stats_widget, 'Statistics')

        response_widget = UnitFrOvertimeWidget(self)
        self.update_graphs.connect(response_widget.update_units)
        self.addTab(response_widget, 'Response')

    @pyqtSlot(list)
    def update_unit(self, units):
        self.units = units
        self.update_graphs.emit(units)


class UnitCharacteristicsWidget(QWidget):
    def __init__(self, parent):
        super(UnitCharacteristicsWidget, self).__init__(parent)
        main_layout = QHBoxLayout(self)
        self.autocor_widget = UnitCharacteristicPlots(self)
        main_layout.addWidget(self.autocor_widget)
        self.units = []

    def showEvent(self, e):
        """update the plots when the tab becomes visible."""
        self.update_units(self.units)

    @pyqtSlot(list)
    def update_units(self, units):
        self.units = units
        if self.isVisible():
            self.autocor_widget.update_plots(units)


class UnitCharacteristicPlots(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""


    def __init__(self, parent=None, width=5, height=4, dpi=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super(UnitCharacteristicPlots, self).__init__(fig)
        self.acor_axes = fig.add_subplot(121)  # type: Axes
        self.acor_axes.set_title('autocorr')
        self.temp_axes = fig.add_subplot(122)  # type: Axes
        self.temp_axes.set_title('waveform')
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_plots(self, units):
        # TODO: make heatmap if len(units) > ~3
        # TODO: rescale axis when adding units.
        self.acor_axes.cla()
        self.temp_axes.cla()
        self.temp_axes.set_title('Waveforms')
        self.acor_axes.set_title('Autocorr')
        for i, u in enumerate(units):  # type: Unit
            c = COLORS[i % len(COLORS)]
            u.plot_autocorrelation(axis=self.acor_axes, color=c)
            u.plot_template(axis=self.temp_axes, color=c)
        if len(units) == 1:
            xpos = self.acor_axes.get_xlim()[1] * .8
            self.acor_axes.text(xpos, 0., "{:0.2f} Hz".format(u.fr))
        self.draw()

class UnitFrOvertimeWidget(QWidget):
    """
    plotting of unit responses.
    """

    def __init__(self, parent):
        super(UnitFrOvertimeWidget, self).__init__(parent)
        layout = QHBoxLayout(self)
        self.figure = _PlotCanvas()
        layout.addWidget(self.figure)
        self.units = []

    def showEvent(self, e):
        """update the plots when the tab becomes visible."""
        self.update_units(self.units)

    @pyqtSlot(list)
    def update_units(self, units):
        self.units = units
        if self.isVisible():
            ax = self.figure.axis
            ax.cla()
            for u in units:  # type: Unit
                ax.hist(u.spiketimes, histtype='step', bins='auto')
                ax.set_title('Firing rate over time')
                ax.set_ylabel('N spikes')
                ax.set_xlabel('Recording time.')
            self.figure.draw()


class PsthViewWidget(QWidget):
    """Abstract gui item for a typical psth display."""

    def __init__(self, parent):
        self.current_units = []
        super(PsthViewWidget, self).__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        self.plot = _PlotCanvas(self)
        layout.addWidget(self.plot)

        controls_layout = QVBoxLayout()
        pre_pad_label = QLabel('Pre plot (ms)')
        pre_pad_box = QSpinBox(self)
        pre_pad_box.setSingleStep(20)
        pre_pad_box.setRange(0, 5000)
        pre_pad_box.setValue(100)
        pre_pad_box.valueChanged.connect(self.plot_param_changed)
        self.pre_pad_box = pre_pad_box

        post_pad_label = QLabel('Post plot (ms)')
        post_pad_box = QSpinBox(self)
        post_pad_box.setSingleStep(20)
        post_pad_box.setRange(0, 5000)
        post_pad_box.setValue(200)
        post_pad_box.valueChanged.connect(self.plot_param_changed)
        self.post_pad_box = post_pad_box

        binsize_label = QLabel('Binsize (ms)')
        binsize_box = QSpinBox(self)
        binsize_box.setSingleStep(5)
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
        self.update_unit_plots(self.current_units)

    def showEvent(self, event: QShowEvent):
        self.update_unit_plots(self.current_units)

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


class _PlotCanvas(FigureCanvas):
    """Plot canvas"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # fig, axes = subplots(6,6,figsize=(width, height), dpi=dpi, sharex=True, sharey=True)
        # self.axis = axes  ### DOING SUBPLOTS FOR THIS IS MASSIVELY SLOW!!
        fig = Figure(figsize=(width, height))
        super(_PlotCanvas, self).__init__(fig)
        self.axis = fig.add_subplot(111)  # type: Axes
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clr(self):
        self.axis.cla()


class RasterViewWidget(QWidget):
    """Abstract widget to display raster plots."""

    def __init__(self, parent):
        super(RasterViewWidget, self).__init__(parent)
        self.current_units = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        self.plot = _PlotCanvas(self)
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

        pointsize_label = QLabel('Point size')
        pointsize = QSpinBox(self)
        pointsize.setSingleStep(1)
        pointsize.setRange(1, 20)
        pointsize.setValue(5)
        pointsize.valueChanged.connect(self.plot_param_changed)
        self.pointsize_box = pointsize


        # todo: make controls hideable.
        controls_layout.addWidget(pre_pad_label)
        controls_layout.addWidget(pre_pad_box)
        controls_layout.addWidget(post_pad_label)
        controls_layout.addWidget(post_pad_box)
        controls_layout.addWidget(pointsize_label)
        controls_layout.addWidget(pointsize)
        controls_layout.addStretch()
        self.controls_layout = controls_layout
        layout.addLayout(controls_layout)

    def showEvent(self, event: QShowEvent):
        self.update_unit_plots(self.current_units)

    @pyqtSlot(int)
    def plot_param_changed(self, _):
        self.update_unit_plots(self.current_units)

    @pyqtSlot(list)
    @abstractmethod
    def update_unit_plots(self, units):
        raise NotImplementedError('This method needs to be overwritten with logic to plot psths.')


class StabilityWidget(QWidget):
    def __init__(self, parent, unit=None):
        super(StabilityWidget, self).__init__(parent, )
        self.setWindowFlags(Qt.Window)
        self.setFixedSize(300,400)
