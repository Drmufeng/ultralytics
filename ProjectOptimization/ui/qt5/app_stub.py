import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.qt5.main_window import MainWindow  # noqa: E402


def create_app(argv: list[str] | None = None) -> QApplication:
    return QApplication(argv if argv is not None else sys.argv)


def run(argv: list[str] | None = None) -> int:
    app = create_app(argv)
    window = MainWindow()
    window.show()
    return app.exec_()


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
