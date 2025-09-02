#!/usr/bin/env python3
"""
Main entry point for MIB EELS Viewer application
"""

def main():
    """Main entry point function."""
    try:
        # Try relative import first (when run as module)
        from .gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
        pyqtgraph_main()
    except ImportError:
        # Fall back to absolute import (when run directly)
        import sys
        import os
        # Add src directory to path for direct execution
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from mib_viewer.gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
        pyqtgraph_main()

if __name__ == "__main__":
    main()
