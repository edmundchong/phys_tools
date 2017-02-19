from phys_tools.views import main_views, pattern_views
from phys_tools.models import PatternSession
from PyQt5.QtWidgets import QApplication


def main():
    import sys
    app = QApplication(sys.argv)
    w = main_views.MainWindow(PatternSession, pattern_views.PatternSessionWidget)
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()