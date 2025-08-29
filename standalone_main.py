#!/usr/bin/env python3
"""
Standalone entry point for PyInstaller executable
No relative imports - everything is absolute for packaging
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for absolute imports
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

def main():
    """Launch the PyQtGraph MIB viewer with error handling"""
    
    try:
        # Import and run the PyQtGraph version
        from mib_viewer.gui.mib_viewer_pyqtgraph import main as pyqtgraph_main
        pyqtgraph_main()
        
    except ImportError as e:
        # Try fallback to original Qt version
        try:
            from mib_viewer.gui.mib_viewer_qt import main as qt_main
            qt_main()
        except ImportError:
            print(f"ERROR: Could not import MIB viewer modules: {e}")
            print("This may be a packaging issue with the standalone executable.")
            input("Press Enter to exit...")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Application crashed: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()