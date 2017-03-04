from phys_tools.views import main_views, pattern_views
from phys_tools.models import PatternSession
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os


def main():
    import sys
    package_directory = os.path.dirname(os.path.abspath(__file__))
    iconpath = os.path.join(package_directory, "support/patterNavigator.icns")
    app = QApplication(sys.argv)
    qApp.setWindowIcon(QIcon(iconpath))
    w = main_views.MainWindow(PatternSession, pattern_views.PatternSessionWidget, 'patterNavigator')
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()