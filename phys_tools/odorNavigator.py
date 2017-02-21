from phys_tools.views import main_views, odor_views
from phys_tools.models import OdorSession
from PyQt5.QtWidgets import QApplication


def main():
    import sys
    app = QApplication(sys.argv)
    w = main_views.MainWindow(OdorSession, odor_views.OdorSessionWidget, 'odorNavigator')
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()