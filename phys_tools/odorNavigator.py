from phys_tools.views import main_views, odor_views
from phys_tools.models import OdorSession
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os


class SystemTrayIcon(QSystemTrayIcon):
    def __init__(self, icon, parent=None):
        super(SystemTrayIcon, self).__init__(icon, parent)
        menu = QMenu(parent)
        exitAction = menu.addAction("Exit")
        menu.addAction(exitAction)
        self.setContextMenu(menu)


def main():
    import sys
    package_directory = os.path.dirname(os.path.abspath(__file__))
    iconpath = os.path.join(package_directory, "support/odorNavigator.icns")
    argv = sys.argv
    app = QApplication(argv)
    qApp.setWindowIcon(QIcon(iconpath))

    w = main_views.MainWindow(OdorSession, odor_views.OdorSessionWidget, 'odorNavigator')
    w.show()


    sys.exit(app.exec_())


if __name__ == "__main__":
    main()