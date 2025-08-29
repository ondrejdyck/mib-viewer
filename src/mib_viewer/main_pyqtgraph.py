#!/usr/bin/env python3
"""
Entry point for PyQtGraph-based MIB EELS Viewer
"""

def main():
    """Launch the PyQtGraph MIB viewer"""
    try:
        # Try relative import (when run as module)
        from .gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
        pyqtgraph_main()
    except ImportError:
        # Fall back for direct execution
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
        pyqtgraph_main()

if __name__ == "__main__":
    main()