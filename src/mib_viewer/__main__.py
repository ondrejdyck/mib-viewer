#!/usr/bin/env python3
"""
Entry point when running the package as a module: python -m mib_viewer
Also serves as PyInstaller entry point
"""

import sys
import os

def main():
    """Main entry point function with robust import handling."""
    try:
        # Try relative import first (when run as module)
        from .gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
        pyqtgraph_main()
    except ImportError:
        try:
            # Try absolute import (for PyInstaller)
            from mib_viewer.gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
            pyqtgraph_main()
        except ImportError:
            # Final fallback with path manipulation
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from mib_viewer.gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
            pyqtgraph_main()

if __name__ == "__main__":
    main()