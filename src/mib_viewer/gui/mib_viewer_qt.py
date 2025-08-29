#!/usr/bin/env python3
"""
MIB Viewer Application - PyQt5 Version
A standalone GUI application for viewing MIB EELS data and ndata images.
"""

import sys
import os
import json
import zipfile
import functools
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QMenuBar, QAction, QStatusBar, QCheckBox, QSizePolicy,
                             QDesktopWidget)
from PyQt5.QtCore import Qt

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, SpanSelector

class SpectrumNavigationToolbar(NavigationToolbar):
    """Custom navigation toolbar with only the tools we want"""
    # Only include: Home, Back, Forward, Pan, Zoom
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom')]

class MibProperties:
    """Class covering Merlin MIB file properties."""
    def __init__(self):
        self.path = ''
        self.merlin_size = (256, 256)
        self.single = True
        self.quad = False
        self.raw = False
        self.dyn_range = '12-bit'
        self.pixeltype = np.uint16
        self.headsize = 384
        self.offset = 0
        self.scan_size = (1, 1)
        self.xy = 1
        self.numberOfFramesInFile = 1
        self.detectorgeometry = '1x1'

def get_mib_properties(head):
    """Parse header of a MIB data and return object containing frame parameters"""
    fp = MibProperties()
    
    # Read detector size
    fp.merlin_size = (int(head[4]), int(head[5]))
    
    # Test if RAW
    if head[6] == 'R64':
        fp.raw = True
    
    if head[7].endswith('2x2'):
        fp.detectorgeometry = '2x2'
    if head[7].endswith('Nx1G'):
        fp.detectorgeometry = 'Nx1'
    
    # Test if single
    if head[2] == '00384':
        fp.single = True
    # Test if quad and read full quad header
    if head[2] == '00768':
        fp.headsize = 768
        fp.quad = True
        fp.single = False
    
    # Set bit-depths for processed data
    if not fp.raw:
        if head[6] == 'U08':
            fp.pixeltype = np.dtype('uint8')
            fp.dyn_range = '1 or 6-bit'
        if head[6] == 'U16':
            fp.pixeltype = np.dtype('>u2')
            fp.dyn_range = '12-bit'
        if head[6] == 'U32':
            fp.pixeltype = np.dtype('>u4')
            fp.dyn_range = '24-bit'
    
    return fp

def auto_detect_scan_size(num_frames):
    """Automatically detect scan size from number of frames"""
    # Try to find the best square or rectangular arrangement
    # Priority: square > rectangular with reasonable aspect ratio
    
    # First try perfect square
    sqrt_frames = int(np.sqrt(num_frames))
    if sqrt_frames * sqrt_frames == num_frames:
        return (sqrt_frames, sqrt_frames)
    
    # Try common rectangular arrangements
    factors = []
    for i in range(1, int(np.sqrt(num_frames)) + 1):
        if num_frames % i == 0:
            factors.append((i, num_frames // i))
    
    # Find the most square-like arrangement (closest to 1:1 aspect ratio)
    if factors:
        best_ratio = float('inf')
        best_size = factors[-1]
        for w, h in factors:
            ratio = max(w, h) / min(w, h)  # Aspect ratio
            if ratio < best_ratio:
                best_ratio = ratio
                best_size = (w, h)
        return best_size
    
    # Fallback: assume 1D scan
    return (num_frames, 1)

def load_mib(path_buffer, scan_size=None):
    """Load Quantum Detectors MIB file from a path."""
    
    # Read header from the start of the file
    try:
        with open(path_buffer, 'rb') as f:
            head = f.read(384).decode().split(',')
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
    except:
        raise ValueError('File does not contain MIB header')
    
    # Parse header info
    mib_prop = get_mib_properties(head)
    mib_prop.path = path_buffer
    
    # Find the size of the data
    merlin_frame_dtype = np.dtype([
        ('header', np.bytes_, mib_prop.headsize),
        ('data', mib_prop.pixeltype, mib_prop.merlin_size)
    ])
    mib_prop.numberOfFramesInFile = filesize // merlin_frame_dtype.itemsize
    
    # Auto-detect scan size if not provided
    if scan_size is None:
        scan_size = auto_detect_scan_size(mib_prop.numberOfFramesInFile)
        print(f"Auto-detected scan size: {scan_size[0]}x{scan_size[1]} from {mib_prop.numberOfFramesInFile} frames")
    
    mib_prop.scan_size = scan_size
    if type(scan_size) == int:
        mib_prop.xy = scan_size
    if type(scan_size) == tuple:
        mib_prop.xy = scan_size[0] * scan_size[1]
    
    if mib_prop.xy > mib_prop.numberOfFramesInFile:
        raise ValueError(f"Requested number of frames: {mib_prop.xy} exceeds available frames: {mib_prop.numberOfFramesInFile}")
    
    if mib_prop.raw:
        raise ValueError('RAW MIB data not supported.')
    
    # Load processed MIB file
    data = np.memmap(
        mib_prop.path,
        dtype=merlin_frame_dtype,
        offset=mib_prop.offset,
        shape=mib_prop.scan_size
    )
    
    return data['data']


class MibViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIB EELS Viewer")
        self.resize(1400, 800)
        self.center_on_screen()
        
        # Data storage
        self.eels_data = None
        self.ndata_data = None
        self.eels_filename = ""
        self.ndata_filename = ""
        
        # Interactive elements
        self.roi_selector_eels = None
        self.roi_selector_ndata = None
        self.energy_selector = None
        
        # Current ROI and energy range
        self.current_roi = None  # (x1, y1, x2, y2)
        self.current_energy_range = (10, 200)  # Default energy range in eV
        
        # Flags to prevent recursive updates
        self.updating_roi = False
        self.updating_energy = False
        
        # Log scale toggle
        self.log_scale = False
        
        # ROI mode (True = rectangle, False = point)
        self.roi_mode = True
        
        # Current point location (x, y) for point mode
        self.current_point = None
        
        self.setup_ui()
    
    def center_on_screen(self):
        """Center the main window on the screen"""
        # Get the available screen geometry (excludes taskbars, panels, etc.)
        screen = QApplication.desktop().availableGeometry()
        
        # Get the window size
        window_size = self.size()
        
        # Calculate center position
        x = (screen.width() - window_size.width()) // 2 + screen.x()
        y = (screen.height() - window_size.height()) // 2 + screen.y()
        
        # Move the window to center
        self.move(x, y)
    
    def setup_ui(self):
        """Set up the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menus()
        
        # Create file info layout (fixed height, no expansion)
        file_info_widget = QWidget()
        file_info_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_info_layout = QHBoxLayout(file_info_widget)
        self.eels_label = QLabel("EELS File: None loaded")
        self.ndata_label = QLabel("ndata File: None loaded")
        file_info_layout.addWidget(self.eels_label)
        file_info_layout.addStretch()
        file_info_layout.addWidget(self.ndata_label)
        main_layout.addWidget(file_info_widget)
        
        # Create matplotlib figure with custom subplot layout
        # Layout: EELS (left), ndata (middle), spectrum (right, 2x width)
        self.figure = Figure(figsize=(16, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar for spectrum pan/zoom
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.nav_toolbar)
        
        # Create plots layout with spectrum control
        plots_widget = QWidget()
        plots_layout = QHBoxLayout(plots_widget)
        plots_layout.addWidget(self.canvas)
        
        # Create spectrum controls (log scale toggle)
        spectrum_controls_widget = QWidget()
        spectrum_controls_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        spectrum_controls_layout = QVBoxLayout(spectrum_controls_widget)
        
        # Add log scale checkbox
        self.log_checkbox = QCheckBox("Log y")
        self.log_checkbox.toggled.connect(self.on_log_toggle)
        spectrum_controls_layout.addWidget(self.log_checkbox)
        
        # Add ROI mode checkbox
        self.roi_checkbox = QCheckBox("ROI Mode")
        self.roi_checkbox.setChecked(True)  # Start in ROI mode
        self.roi_checkbox.toggled.connect(self.on_roi_mode_toggle)
        spectrum_controls_layout.addWidget(self.roi_checkbox)
        
        spectrum_controls_layout.addStretch()  # Push controls to top
        
        plots_layout.addWidget(spectrum_controls_widget)
        main_layout.addWidget(plots_widget)
        
        # Create subplots with custom spacing
        # GridSpec: 1 row, 4 columns, spectrum gets 2 columns
        gs = self.figure.add_gridspec(1, 4, width_ratios=[1, 1, 2, 0])
        self.ax_eels = self.figure.add_subplot(gs[0, 0])
        self.ax_ndata = self.figure.add_subplot(gs[0, 1]) 
        self.ax_spectrum = self.figure.add_subplot(gs[0, 2])
        
        # Configure subplots
        self.ax_eels.set_title("EELS Image (Integrated)")
        self.ax_eels.set_aspect('equal')
        
        self.ax_ndata.set_title("ndata Image")
        self.ax_ndata.set_aspect('equal')
        
        self.ax_spectrum.set_title("Average Spectrum")
        self.ax_spectrum.set_xlabel("Energy (eV)")
        self.ax_spectrum.set_ylabel("Intensity")
        
        # Enable pan/zoom on spectrum plot only (not on image plots)
        self.ax_eels.set_navigate(False)  # Disable navigation for EELS image
        self.ax_ndata.set_navigate(False)  # Disable navigation for ndata image
        self.ax_spectrum.set_navigate(True)  # Enable navigation for spectrum
        
        # Adjust layout to minimize whitespace
        self.figure.tight_layout(pad=0.5)  # Reduce padding to ~0.25 inch
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize empty plots
        self.update_displays()
    
    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_mib_action = QAction('Load MIB File...', self)
        load_mib_action.triggered.connect(self.load_mib_file)
        file_menu.addAction(load_mib_action)
        
        load_ndata_action = QAction('Load ndata File...', self)
        load_ndata_action.triggered.connect(self.load_ndata_file)
        file_menu.addAction(load_ndata_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def load_mib_file(self):
        """Load a MIB file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select MIB file",
            "",
            "MIB files (*.mib);;All files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            self.status_bar.showMessage("Loading MIB file...")
            QApplication.processEvents()
            
            # Load the MIB data with automatic scan size detection
            raw_data = load_mib(filename)
            
            # Flip the energy axis and reshape
            self.eels_data = raw_data[:, :, ::-1, :]
            
            self.eels_filename = os.path.basename(filename)
            self.eels_label.setText(f"EELS File: {self.eels_filename}")
            
            # Set default ROI and point (centered)
            h, w = self.eels_data.shape[:2]
            if self.roi_mode:
                # Default ROI (centered, 50% size)
                roi_w, roi_h = w // 2, h // 2
                roi_x, roi_y = w // 4, h // 4
                self.current_roi = (roi_x, roi_y, roi_x + roi_w, roi_y + roi_h)
            else:
                # Default point (center of image)
                self.current_point = (w // 2, h // 2)
            
            # Check compatibility with loaded ndata
            if self.ndata_data is not None:
                self.check_compatibility()
            
            self.update_displays()
            self.status_bar.showMessage(f"Loaded MIB file: {self.eels_filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load MIB file:\n{str(e)}")
            self.status_bar.showMessage("Ready")
    
    def load_ndata_file(self):
        """Load an ndata file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select ndata file",
            "",
            "ndata files (*.ndata*);;All files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            self.status_bar.showMessage("Loading ndata file...")
            QApplication.processEvents()
            
            # Extract and load ndata file
            with zipfile.ZipFile(filename, 'r') as zip_file:
                # Load numpy data
                with zip_file.open('data.npy') as npy_file:
                    self.ndata_data = np.load(npy_file)
                
                # Load metadata (optional, for future use)
                with zip_file.open('metadata.json') as json_file:
                    metadata = json.load(json_file)
            
            self.ndata_filename = os.path.basename(filename)
            self.ndata_label.setText(f"ndata File: {self.ndata_filename}")
            
            # Check compatibility with loaded EELS data
            if self.eels_data is not None:
                self.check_compatibility()
            
            self.update_displays()
            self.status_bar.showMessage(f"Loaded ndata file: {self.ndata_filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load ndata file:\n{str(e)}")
            self.status_bar.showMessage("Ready")
    
    def check_compatibility(self):
        """Check if EELS and ndata dimensions are compatible"""
        if self.eels_data is not None and self.ndata_data is not None:
            eels_shape = self.eels_data.shape[:2]  # (height, width)
            ndata_shape = self.ndata_data.shape
            
            if eels_shape != ndata_shape:
                QMessageBox.warning(
                    self,
                    "Dimension Mismatch",
                    f"EELS data shape: {eels_shape}\n"
                    f"ndata shape: {ndata_shape}\n\n"
                    f"The spatial dimensions must match for proper correlation."
                )
    
    def update_displays(self):
        """Update all display panels"""
        # Clear all axes
        self.ax_eels.clear()
        self.ax_ndata.clear()
        self.ax_spectrum.clear()
        
        # Update EELS image
        if self.eels_data is not None:
            self.update_eels_display()
        else:
            self.ax_eels.text(0.5, 0.5, 'No EELS data loaded', 
                             ha='center', va='center', transform=self.ax_eels.transAxes)
        
        # Update ndata image
        if self.ndata_data is not None:
            self.update_ndata_display()
        else:
            self.ax_ndata.text(0.5, 0.5, 'No ndata loaded', 
                              ha='center', va='center', transform=self.ax_ndata.transAxes)
        
        # Update spectrum
        if self.eels_data is not None and (self.current_roi is not None or self.current_point is not None):
            self.update_spectrum_display()
        else:
            self.ax_spectrum.text(0.5, 0.5, 'No EELS data loaded', 
                                 ha='center', va='center', transform=self.ax_spectrum.transAxes)
        
        # Set titles
        self.ax_eels.set_title("EELS Image (Integrated)")
        self.ax_ndata.set_title("ndata Image")
        self.ax_spectrum.set_title("Average Spectrum")
        self.ax_spectrum.set_xlabel("Energy (eV)")
        if self.log_scale:
            self.ax_spectrum.set_ylabel("log₁₀(Intensity)")
        else:
            self.ax_spectrum.set_ylabel("Intensity")
        
        # Apply tight layout before drawing to minimize whitespace
        self.figure.tight_layout(pad=0.5)
        self.canvas.draw()
    
    def update_eels_display(self):
        """Update the EELS integrated image display"""
        if self.eels_data is None:
            return
        
        # Integrate over energy range
        energy_pixels = self.eels_data.shape[2]
        
        try:
            # Try to use the specified energy range (assuming 1 eV per pixel for now)
            start_idx = max(0, int(self.current_energy_range[0]))
            end_idx = min(energy_pixels, int(self.current_energy_range[1]))
        except:
            # Fallback: use first half of spectrum
            start_idx = 0
            end_idx = energy_pixels // 2
        
        # Integrate EELS data
        integrated = np.sum(self.eels_data[:, :, start_idx:end_idx, :], axis=(2, 3))
        
        # Display image
        self.ax_eels.imshow(integrated, cmap='viridis')
        
        # Set up interactive selection based on mode
        if self.roi_selector_eels is not None:
            self.roi_selector_eels.disconnect_events()
            self.roi_selector_eels = None
        
        if self.roi_mode:
            # Rectangle ROI mode
            self.roi_selector_eels = RectangleSelector(
                self.ax_eels, 
                self.on_roi_select_eels,
                useblit=True,
                button=[1],  # Only left mouse button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True
            )
            
            # Hook into motion events for real-time updates (modern approach)
            eels_onmove_callback = functools.partial(self.on_motion_event, selector=self.roi_selector_eels, callback=self.on_roi_drag_eels)
            self.roi_selector_eels.connect_event('motion_notify_event', eels_onmove_callback)
            
            # Set initial ROI if defined
            if self.current_roi is not None:
                x1, y1, x2, y2 = self.current_roi
                self.roi_selector_eels.extents = (x1, x2, y1, y2)
        else:
            # Point marker mode
            # Connect click events for point selection
            self.ax_eels.figure.canvas.mpl_connect('button_press_event', self.on_point_click_eels)
            self.ax_eels.figure.canvas.mpl_connect('button_release_event', self.on_point_release_eels)
            self.ax_eels.figure.canvas.mpl_connect('motion_notify_event', self.on_point_drag_eels)
            
            # Draw point marker if defined
            if self.current_point is not None:
                px, py = self.current_point
                self.ax_eels.plot(px, py, 'r+', markersize=15, markeredgewidth=3)
    
    def update_ndata_display(self):
        """Update the ndata image display"""
        if self.ndata_data is None:
            return
        
        # Display image
        self.ax_ndata.imshow(self.ndata_data, cmap='gray')
        
        # Set up interactive selection based on mode
        if self.roi_selector_ndata is not None:
            self.roi_selector_ndata.disconnect_events()
            self.roi_selector_ndata = None
        
        if self.roi_mode:
            # Rectangle ROI mode
            self.roi_selector_ndata = RectangleSelector(
                self.ax_ndata, 
                self.on_roi_select_ndata,
                useblit=True,
                button=[1],  # Only left mouse button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True
            )
            
            # Hook into motion events for real-time updates (modern approach)
            ndata_onmove_callback = functools.partial(self.on_motion_event, selector=self.roi_selector_ndata, callback=self.on_roi_drag_ndata)
            self.roi_selector_ndata.connect_event('motion_notify_event', ndata_onmove_callback)
            
            # Set initial ROI if defined
            if self.current_roi is not None:
                x1, y1, x2, y2 = self.current_roi
                self.roi_selector_ndata.extents = (x1, x2, y1, y2)
        else:
            # Point marker mode
            # Connect click events for point selection
            self.ax_ndata.figure.canvas.mpl_connect('button_press_event', self.on_point_click_ndata)
            self.ax_ndata.figure.canvas.mpl_connect('button_release_event', self.on_point_release_ndata)
            self.ax_ndata.figure.canvas.mpl_connect('motion_notify_event', self.on_point_drag_ndata)
            
            # Draw point marker if defined
            if self.current_point is not None:
                px, py = self.current_point
                self.ax_ndata.plot(px, py, 'r+', markersize=15, markeredgewidth=3)
    
    def update_spectrum_display(self):
        """Update the spectrum plot"""
        if self.eels_data is None:
            return
        
        if self.roi_mode and self.current_roi is not None:
            # Extract spectrum from ROI
            x1, y1, x2, y2 = self.current_roi
            x1, x2 = int(max(0, x1)), int(min(self.eels_data.shape[1], x2))
            y1, y2 = int(max(0, y1)), int(min(self.eels_data.shape[0], y2))
            
            # Average spectrum over ROI
            roi_spectra = self.eels_data[y1:y2, x1:x2, :, :]
            avg_spectrum = np.mean(roi_spectra, axis=(0, 1, 3))
        elif not self.roi_mode and self.current_point is not None:
            # Extract spectrum from single point
            px, py = self.current_point
            px, py = int(max(0, min(self.eels_data.shape[1]-1, px))), int(max(0, min(self.eels_data.shape[0]-1, py)))
            
            # Single point spectrum (average over detector pixels)
            point_spectra = self.eels_data[py, px, :, :]
            avg_spectrum = np.mean(point_spectra, axis=1)
        else:
            return
        
        # Create energy axis (assuming 1 eV per pixel for now)
        energy_axis = np.arange(len(avg_spectrum))
        
        # Plot spectrum (with optional log scale)
        if self.log_scale:
            # Use log scale, avoiding zero/negative values
            log_spectrum = np.log10(np.maximum(avg_spectrum, 1e-10))
            self.ax_spectrum.plot(energy_axis, log_spectrum)
            self.ax_spectrum.set_ylabel("log₁₀(Intensity)")
        else:
            self.ax_spectrum.plot(energy_axis, avg_spectrum)
            self.ax_spectrum.set_ylabel("Intensity")
        
        # Set up interactive energy range selector
        if self.energy_selector is not None:
            self.energy_selector.disconnect_events()
        
        self.energy_selector = SpanSelector(
            self.ax_spectrum,
            self.on_energy_select,
            'horizontal',
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor='red')
        )
        
        # Hook into motion events for real-time updates (modern approach)
        energy_onmove_callback = functools.partial(self.on_motion_event, selector=self.energy_selector, callback=self.on_energy_drag)
        self.energy_selector.connect_event('motion_notify_event', energy_onmove_callback)
        
        # Set initial energy range
        if self.current_energy_range is not None:
            self.energy_selector.extents = self.current_energy_range
    
    def on_roi_select_eels(self, eclick, erelease):
        """Handle ROI selection on EELS image"""
        if self.updating_roi:
            return
        
        # Get new ROI coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Update current ROI
        self.current_roi = (x1, y1, x2, y2)
        
        # Update the other ROI selector
        self.sync_roi_selectors()
        
        # Update spectrum
        self.ax_spectrum.clear()
        self.update_spectrum_display()
        self.canvas.draw()
    
    def on_roi_select_ndata(self, eclick, erelease):
        """Handle ROI selection on ndata image"""
        if self.updating_roi:
            return
        
        # Get new ROI coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Update current ROI
        self.current_roi = (x1, y1, x2, y2)
        
        # Update the other ROI selector
        self.sync_roi_selectors()
        
        # Update spectrum
        self.ax_spectrum.clear()
        self.update_spectrum_display()
        self.canvas.draw()
    
    def on_energy_select(self, vmin, vmax):
        """Handle energy range selection on spectrum"""
        if self.updating_energy:
            return
        
        # Update current energy range
        self.current_energy_range = (vmin, vmax)
        
        # Update EELS image (re-integrate with new energy range)
        self.updating_energy = True
        self.ax_eels.clear()
        self.update_eels_display()
        self.updating_energy = False
        
        self.canvas.draw()
    
    def on_motion_event(self, event, selector, callback):
        """Filter motion events to only trigger during active dragging"""
        # Only call callback if selector is actively being used
        if not selector.ignore(event) and hasattr(selector, '_eventpress') and selector._eventpress:
            callback()
    
    def on_roi_drag_eels(self, *args):
        """Handle real-time ROI dragging on EELS image"""
        if self.updating_roi:
            return
        
        # Get current extents from the selector
        try:
            x1, x2, y1, y2 = self.roi_selector_eels.extents
        except:
            return  # Skip if extents not available
        
        # Update current ROI
        self.current_roi = (x1, y1, x2, y2)
        
        # Update spectrum in real-time (the magic!)
        self.ax_spectrum.clear()
        self.update_spectrum_display()
        self.canvas.draw_idle()  # Use draw_idle for better performance
    
    def on_roi_drag_ndata(self, *args):
        """Handle real-time ROI dragging on ndata image"""
        if self.updating_roi:
            return
        
        # Get current extents from the selector
        try:
            x1, x2, y1, y2 = self.roi_selector_ndata.extents
        except:
            return  # Skip if extents not available
        
        # Update current ROI
        self.current_roi = (x1, y1, x2, y2)
        
        # Update spectrum in real-time (the magic!)
        self.ax_spectrum.clear()
        self.update_spectrum_display()
        self.canvas.draw_idle()  # Use draw_idle for better performance
    
    def on_energy_drag(self, *args):
        """Handle real-time energy range dragging"""
        if self.updating_energy:
            return
        
        # Get current extents from the selector
        try:
            vmin, vmax = self.energy_selector.extents
        except:
            return  # Skip if extents not available
        
        # Update current energy range
        self.current_energy_range = (vmin, vmax)
        
        # Update EELS image in real-time (double magic!)
        self.updating_energy = True
        self.ax_eels.clear()
        self.update_eels_display()
        self.updating_energy = False
        
        self.canvas.draw_idle()  # Use draw_idle for better performance
    
    # Point mode event handlers
    def on_point_click_eels(self, event):
        """Handle point click on EELS image"""
        if not self.roi_mode and event.inaxes == self.ax_eels and event.button == 1:
            self.current_point = (event.xdata, event.ydata)
            self.sync_point_markers()
            self.ax_spectrum.clear()
            self.update_spectrum_display()
            self.canvas.draw()
    
    def on_point_release_eels(self, event):
        """Handle point release on EELS image"""
        pass  # Nothing needed for release in point mode
    
    def on_point_drag_eels(self, event):
        """Handle point dragging on EELS image"""
        if (not self.roi_mode and event.inaxes == self.ax_eels and 
            event.button == 1 and self.current_point is not None):
            self.current_point = (event.xdata, event.ydata)
            self.update_displays()
    
    def on_point_click_ndata(self, event):
        """Handle point click on ndata image"""
        if not self.roi_mode and event.inaxes == self.ax_ndata and event.button == 1:
            self.current_point = (event.xdata, event.ydata)
            self.sync_point_markers()
            self.ax_spectrum.clear()
            self.update_spectrum_display()
            self.canvas.draw()
    
    def on_point_release_ndata(self, event):
        """Handle point release on ndata image"""
        pass  # Nothing needed for release in point mode
    
    def on_point_drag_ndata(self, event):
        """Handle point dragging on ndata image"""
        if (not self.roi_mode and event.inaxes == self.ax_ndata and 
            event.button == 1 and self.current_point is not None):
            self.current_point = (event.xdata, event.ydata)
            self.update_displays()
    
    def sync_point_markers(self):
        """Synchronize point markers between EELS and ndata images"""
        # This is handled automatically by update_displays()
        pass
    
    def sync_roi_selectors(self):
        """Synchronize ROI selectors between EELS and ndata images"""
        if self.current_roi is None:
            return
        
        self.updating_roi = True
        
        x1, y1, x2, y2 = self.current_roi
        
        # Update EELS ROI selector
        if (self.roi_selector_eels is not None and 
            self.eels_data is not None):
            self.roi_selector_eels.extents = (x1, x2, y1, y2)
        
        # Update ndata ROI selector  
        if (self.roi_selector_ndata is not None and 
            self.ndata_data is not None):
            self.roi_selector_ndata.extents = (x1, x2, y1, y2)
        
        self.updating_roi = False
    
    def on_log_toggle(self, checked):
        """Handle log scale toggle"""
        self.log_scale = checked
        # Update spectrum display
        if self.eels_data is not None and (self.current_roi is not None or self.current_point is not None):
            self.ax_spectrum.clear()
            self.update_spectrum_display()
            self.canvas.draw()
    
    def on_roi_mode_toggle(self, checked):
        """Handle ROI mode toggle"""
        self.roi_mode = checked
        
        if self.roi_mode:
            # Switching to ROI mode
            if self.current_point is not None and self.current_roi is None:
                # Convert point to default ROI
                px, py = self.current_point
                # Create a small ROI around the point
                roi_size = 10  # 10x10 pixel ROI
                self.current_roi = (px - roi_size//2, py - roi_size//2, 
                                   px + roi_size//2, py + roi_size//2)
        else:
            # Switching to point mode
            if self.current_roi is not None:
                # Convert ROI to point (center of ROI)
                x1, y1, x2, y2 = self.current_roi
                self.current_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            elif self.current_point is None and self.eels_data is not None:
                # Set default point to center of image
                h, w = self.eels_data.shape[:2]
                self.current_point = (w // 2, h // 2)
        
        # Update displays
        self.update_displays()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "MIB EELS Viewer\n\n"
                         "A tool for viewing MIB EELS data and ndata images.\n"
                         "Load MIB files and optionally corresponding ndata files\n"
                         "to interactively explore spectroscopic data.")

def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    viewer = MibViewer()
    viewer.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()