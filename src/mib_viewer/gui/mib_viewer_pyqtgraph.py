#!/usr/bin/env python3
"""
MIB Viewer Application - PyQtGraph Version
A high-performance GUI application for viewing MIB EELS data with real-time interactions.
"""

import sys
import os
import json
import zipfile
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QMenuBar, QAction, QStatusBar, QCheckBox, QSizePolicy,
                             QSplitter, QGroupBox, QDesktopWidget, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

# Import MIB loading functions
try:
    # Try relative import (when run as module)
    from .mib_viewer_qt import load_mib, get_mib_properties, auto_detect_scan_size
except ImportError:
    # Fall back for direct execution
    from mib_viewer_qt import load_mib, get_mib_properties, auto_detect_scan_size

# Configure PyQtGraph
pg.setConfigOptions(antialias=True, useOpenGL=True)

class MibViewerPyQtGraph(QMainWindow):
    """
    High-performance MIB EELS viewer using PyQtGraph for real-time interactions
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIB EELS Viewer - PyQtGraph")
        self.resize(1600, 1600)  # Make window square (tall as it is wide)
        self.center_on_screen()
        
        # Data storage
        self.eels_data = None
        self.ndata_data = None
        self.eels_filename = ""
        self.ndata_filename = ""
        
        # Current selections
        self.current_roi = None  # Will store ROI bounds
        self.current_energy_range = (10, 200)  # Default energy range
        
        # PyQtGraph items
        self.eels_image_item = None
        self.ndata_image_item = None
        self.spectrum_curve = None
        self.eels_roi = None
        self.ndata_roi = None
        self.energy_region = None
        
        # Settings
        self.log_scale = False
        self.roi_mode = True  # True = rectangle ROI, False = crosshair
        
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
        """Set up the user interface with PyQtGraph widgets"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menus()
        
        # Create file info layout
        file_info_widget = QWidget()
        file_info_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_info_layout = QHBoxLayout(file_info_widget)
        self.eels_label = QLabel("EELS File: None loaded")
        self.ndata_label = QLabel("ndata File: None loaded")
        file_info_layout.addWidget(self.eels_label)
        file_info_layout.addStretch()
        file_info_layout.addWidget(self.ndata_label)
        main_layout.addWidget(file_info_widget)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create EELS tab
        self.eels_tab = self.create_eels_tab()
        self.tab_widget.addTab(self.eels_tab, "EELS Analysis")
        
        # Create 4D STEM tab
        self.stem4d_tab = self.create_4d_stem_tab()
        self.tab_widget.addTab(self.stem4d_tab, "4D STEM")
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load MIB file to begin")
        
        # Initialize plots
        self.setup_plots()
    
    def create_eels_tab(self):
        """Create the EELS analysis tab"""
        # Create main horizontal splitter for plots and controls
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel with vertical splitter for plots  
        plots_panel = QWidget()
        plots_panel_layout = QVBoxLayout(plots_panel)
        plots_panel_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for clean splitter
        
        # Create vertical splitter to divide images (top) from spectrum (bottom)
        plots_splitter = QSplitter(Qt.Vertical)
        plots_panel_layout.addWidget(plots_splitter)
        
        # Create top section with horizontal splitter for images
        images_splitter = QSplitter(Qt.Horizontal)
        
        # EELS plot
        eels_group = QGroupBox("EELS Image (Integrated)")
        eels_layout = QVBoxLayout(eels_group)
        self.eels_plot = pg.PlotWidget()
        self.eels_plot.setAspectLocked(True)
        self.eels_plot.hideAxis('left')
        self.eels_plot.hideAxis('bottom')
        eels_layout.addWidget(self.eels_plot)
        images_splitter.addWidget(eels_group)
        
        # ndata plot
        ndata_group = QGroupBox("ndata Image")
        ndata_layout = QVBoxLayout(ndata_group)
        self.ndata_plot = pg.PlotWidget()
        self.ndata_plot.setAspectLocked(True)
        self.ndata_plot.hideAxis('left')
        self.ndata_plot.hideAxis('bottom')
        ndata_layout.addWidget(self.ndata_plot)
        images_splitter.addWidget(ndata_group)
        
        # Set equal sizes for the two images
        images_splitter.setSizes([400, 400])
        plots_splitter.addWidget(images_splitter)
        
        # Spectrum plot spanning full width below images
        spectrum_group = QGroupBox("Average Spectrum")
        spectrum_layout = QVBoxLayout(spectrum_group)
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Intensity')
        self.spectrum_plot.setLabel('bottom', 'Energy (eV)')
        spectrum_layout.addWidget(self.spectrum_plot)
        plots_splitter.addWidget(spectrum_group)
        
        # Set proportions - images take 50%, spectrum takes 50%
        plots_splitter.setSizes([800, 800])  # 50% / 50% split of available height
        
        main_splitter.addWidget(plots_panel)
        
        # Create narrow right panel for controls
        controls_panel = QWidget()
        controls_panel.setMinimumWidth(120)  # Minimum width
        controls_panel.setMaximumWidth(180)  # Maximum width for flexibility
        controls_layout = QVBoxLayout(controls_panel)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_group_layout = QVBoxLayout(controls_group)
        
        # Log scale checkbox
        self.log_checkbox = QCheckBox("Log Scale")
        self.log_checkbox.toggled.connect(self.on_log_toggle)
        controls_group_layout.addWidget(self.log_checkbox)
        
        # ROI mode checkbox
        self.roi_checkbox = QCheckBox("ROI Mode")
        self.roi_checkbox.setChecked(True)
        self.roi_checkbox.setToolTip("Toggle between rectangular ROI (resizable, rotatable) and crosshair cursor")
        self.roi_checkbox.toggled.connect(self.on_roi_mode_toggle)
        controls_group_layout.addWidget(self.roi_checkbox)
        
        controls_group_layout.addStretch()
        controls_layout.addWidget(controls_group)
        controls_layout.addStretch()
        
        main_splitter.addWidget(controls_panel)
        
        # Set main splitter proportions - plots get most space, controls get minimal
        main_splitter.setSizes([1450, 150])
        
        return main_splitter
    
    def create_4d_stem_tab(self):
        """Create the 4D STEM analysis tab"""
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel for scan and diffraction images
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Vertical splitter for scan image (top) and diffraction pattern (bottom)
        left_splitter = QSplitter(Qt.Vertical)
        left_layout.addWidget(left_splitter)
        
        # Real space scan image (top left)
        scan_group = QGroupBox("Real Space - Scan Position")
        scan_layout = QVBoxLayout(scan_group)
        self.scan_plot = pg.PlotWidget()
        self.scan_plot.setAspectLocked(True)
        self.scan_plot.setLabel('left', 'Y (pixels)')
        self.scan_plot.setLabel('bottom', 'X (pixels)')
        scan_layout.addWidget(self.scan_plot)
        left_splitter.addWidget(scan_group)
        
        # Reciprocal space diffraction pattern (bottom left)
        diffraction_group = QGroupBox("Reciprocal Space - Diffraction Pattern")
        diffraction_layout = QVBoxLayout(diffraction_group)
        self.diffraction_plot = pg.PlotWidget()
        self.diffraction_plot.setAspectLocked(True)
        self.diffraction_plot.setLabel('left', 'Q_y (pixels)')
        self.diffraction_plot.setLabel('bottom', 'Q_x (pixels)')
        diffraction_layout.addWidget(self.diffraction_plot)
        left_splitter.addWidget(diffraction_group)
        
        # Set equal split for scan and diffraction
        left_splitter.setSizes([400, 400])
        main_splitter.addWidget(left_panel)
        
        # Create middle panel for virtual imaging (2 images stacked)
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create vertical splitter for BF and DF images
        virtual_splitter = QSplitter(Qt.Vertical)
        middle_layout.addWidget(virtual_splitter)
        
        # Bright Field virtual image
        bf_group = QGroupBox("Bright Field (BF) - Disk Detector")
        bf_layout = QVBoxLayout(bf_group)
        self.bf_plot = pg.PlotWidget()
        self.bf_plot.setAspectLocked(True)
        self.bf_plot.setLabel('left', 'Y (pixels)')
        self.bf_plot.setLabel('bottom', 'X (pixels)')
        bf_layout.addWidget(self.bf_plot)
        virtual_splitter.addWidget(bf_group)
        
        # Dark Field virtual image
        df_group = QGroupBox("Dark Field (DF) - Annular Detector")
        df_layout = QVBoxLayout(df_group)
        self.df_plot = pg.PlotWidget()
        self.df_plot.setAspectLocked(True)
        self.df_plot.setLabel('left', 'Y (pixels)')
        self.df_plot.setLabel('bottom', 'X (pixels)')
        df_layout.addWidget(self.df_plot)
        virtual_splitter.addWidget(df_group)
        
        # Set equal split for BF and DF
        virtual_splitter.setSizes([400, 400])
        main_splitter.addWidget(middle_panel)
        
        # Create narrow right panel for 4D STEM controls
        stem_controls_panel = QWidget()
        stem_controls_panel.setMinimumWidth(120)
        stem_controls_panel.setMaximumWidth(180)
        stem_controls_layout = QVBoxLayout(stem_controls_panel)
        
        # 4D STEM controls group
        stem_controls_group = QGroupBox("4D STEM Controls")
        controls_group_layout = QVBoxLayout(stem_controls_group)
        
        # Detector overlay visibility toggle
        self.detector_overlay_checkbox = QCheckBox("Show Detectors")
        self.detector_overlay_checkbox.setChecked(True)
        self.detector_overlay_checkbox.setToolTip("Toggle visibility of virtual detector overlays")
        self.detector_overlay_checkbox.toggled.connect(self.on_detector_overlay_toggle)
        controls_group_layout.addWidget(self.detector_overlay_checkbox)
        
        # Info label
        info_label = QLabel("Virtual Detectors:\n\n"
                           "• BF: Central disk\n"
                           "• DF: Outer annulus\n"
                           "• Resize handles adjust detector size\n"
                           "• Click scan image to select position")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 10px;")
        controls_group_layout.addWidget(info_label)
        
        controls_group_layout.addStretch()
        stem_controls_layout.addWidget(stem_controls_group)
        stem_controls_layout.addStretch()
        
        main_splitter.addWidget(stem_controls_panel)
        
        # Set proportions - scan/diffraction get most space, virtual imaging medium, controls minimal
        main_splitter.setSizes([800, 600, 200])
        
        return main_splitter
    
    def setup_plots(self):
        """Initialize PyQtGraph plot items"""
        # EELS tab items
        self.eels_image_item = pg.ImageItem()
        self.eels_plot.addItem(self.eels_image_item)
        
        self.ndata_image_item = pg.ImageItem()
        self.ndata_plot.addItem(self.ndata_image_item)
        
        self.spectrum_curve = self.spectrum_plot.plot(pen='w')
        
        # ROI widgets (will be configured when data is loaded)
        self.setup_roi_widgets()
        
        # Energy selection region
        self.setup_energy_region()
        
        # 4D STEM tab items
        self.scan_image_item = pg.ImageItem()
        self.scan_plot.addItem(self.scan_image_item)
        
        self.diffraction_image_item = pg.ImageItem()
        self.diffraction_plot.addItem(self.diffraction_image_item)
        
        # BF and DF virtual images
        self.bf_image_item = pg.ImageItem()
        self.bf_plot.addItem(self.bf_image_item)
        
        self.df_image_item = pg.ImageItem()
        self.df_plot.addItem(self.df_image_item)
        
        # Cursor for scan position selection
        self.scan_cursor = pg.CircleROI([0, 0], [5, 5], pen='r', removable=False)
        self.scan_cursor.sigRegionChanged.connect(self.on_scan_position_changed)
        self.scan_plot.addItem(self.scan_cursor)
        
        # Virtual detector overlays on diffraction pattern
        # BF detector - resizable centered disk
        self.bf_detector = pg.CircleROI([0, 0], [50, 50], pen='g', movable=False, removable=False)
        self.bf_detector.sigRegionChanged.connect(self.on_detector_changed)
        self.diffraction_plot.addItem(self.bf_detector)
        
        # DF detector - resizable centered annular ring  
        # Using EllipseROI for outer boundary
        self.df_detector_outer = pg.CircleROI([0, 0], [100, 100], pen='b', movable=False, removable=False)
        self.df_detector_outer.sigRegionChanged.connect(self.on_detector_changed)
        self.diffraction_plot.addItem(self.df_detector_outer)
        
        # Inner boundary of annular detector
        self.df_detector_inner = pg.CircleROI([0, 0], [60, 60], pen='b', movable=False, removable=False)
        self.df_detector_inner.sigRegionChanged.connect(self.on_detector_changed)
        self.diffraction_plot.addItem(self.df_detector_inner)
        
        # Initialize detector visibility
        self.detector_overlays_visible = True
        
        # Initialize detector update timer (for delayed recalculation)
        self.detector_update_timer = QTimer()
        self.detector_update_timer.setSingleShot(True)
        self.detector_update_timer.timeout.connect(self.delayed_calculate_virtual_images)
    
    def setup_roi_widgets(self):
        """Setup ROI selection widgets"""
        if self.roi_mode:
            # Create fully adjustable rectangular ROI widgets
            self.eels_roi = pg.ROI([0, 0], [100, 100], pen='r', movable=True, resizable=True, 
                                   rotatable=True, removable=False)
            self.ndata_roi = pg.ROI([0, 0], [100, 100], pen='r', movable=True, resizable=True,
                                    rotatable=True, removable=False)
            
            # Add corner handles for resizing (this makes corners grabbable)
            self.eels_roi.addScaleHandle([1, 1], [0, 0])    # Bottom-right handle
            self.eels_roi.addScaleHandle([0, 0], [1, 1])    # Top-left handle  
            self.eels_roi.addScaleHandle([1, 0], [0, 1])    # Top-right handle
            self.eels_roi.addScaleHandle([0, 1], [1, 0])    # Bottom-left handle
            
            self.ndata_roi.addScaleHandle([1, 1], [0, 0])   # Bottom-right handle
            self.ndata_roi.addScaleHandle([0, 0], [1, 1])   # Top-left handle
            self.ndata_roi.addScaleHandle([1, 0], [0, 1])   # Top-right handle
            self.ndata_roi.addScaleHandle([0, 1], [1, 0])   # Bottom-left handle
            
            # Add rotation handle (small circle for rotation)
            self.eels_roi.addRotateHandle([1, 0], [0.5, 0.5])   # Rotation handle at top-right
            self.ndata_roi.addRotateHandle([1, 0], [0.5, 0.5]) # Rotation handle at top-right
            
            # Connect ROI change signals
            self.eels_roi.sigRegionChanged.connect(lambda: self.on_roi_changed(source='eels'))
            self.ndata_roi.sigRegionChanged.connect(lambda: self.on_roi_changed(source='ndata'))
        else:
            # Crosshair cursors
            self.eels_roi = pg.CrosshairROI([0, 0], size=1, pen='r')
            self.ndata_roi = pg.CrosshairROI([0, 0], size=1, pen='r')
            
            self.eels_roi.sigRegionChanged.connect(lambda: self.on_crosshair_changed(source='eels'))
            self.ndata_roi.sigRegionChanged.connect(lambda: self.on_crosshair_changed(source='ndata'))
    
    def setup_energy_region(self):
        """Setup energy range selection widget"""
        self.energy_region = pg.LinearRegionItem(
            values=self.current_energy_range,
            brush=pg.mkBrush(255, 0, 0, 50),
            movable=True
        )
        self.spectrum_plot.addItem(self.energy_region)
        self.energy_region.sigRegionChanged.connect(self.on_energy_range_changed)
    
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
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        reset_view_action = QAction('Reset View', self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
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
            
            # Flip the energy axis and store
            self.eels_data = raw_data[:, :, ::-1, :]
            
            self.eels_filename = os.path.basename(filename)
            self.eels_label.setText(f"EELS File: {self.eels_filename}")
            
            # Initialize ROI to center of image
            h, w = self.eels_data.shape[:2]
            if self.roi_mode:
                # Default ROI (centered, reasonable size, no rotation)
                roi_size = min(w, h) // 3
                roi_x = (w - roi_size) // 2
                roi_y = (h - roi_size) // 2
                self.current_roi = (roi_x, roi_y, roi_size, roi_size, 0.0)  # Added rotation angle
            else:
                # Default crosshair position (center)
                self.current_roi = (w // 2, h // 2)
            
            # Check compatibility with ndata
            if self.ndata_data is not None:
                self.check_compatibility()
            
            self.update_displays()
            self.setup_roi_positions()
            self.update_4d_displays()  # Update 4D STEM views
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
                
                # Load metadata (optional)
                with zip_file.open('metadata.json') as json_file:
                    metadata = json.load(json_file)
            
            self.ndata_filename = os.path.basename(filename)
            self.ndata_label.setText(f"ndata File: {self.ndata_filename}")
            
            # Check compatibility with EELS data
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
            eels_shape = self.eels_data.shape[:2]
            ndata_shape = self.ndata_data.shape
            
            if eels_shape != ndata_shape:
                QMessageBox.warning(
                    self,
                    "Dimension Mismatch",
                    f"EELS data shape: {eels_shape}\n"
                    f"ndata shape: {ndata_shape}\n\n"
                    f"The spatial dimensions must match for proper correlation."
                )
    
    def setup_roi_positions(self):
        """Set initial ROI positions"""
        if self.eels_data is None:
            return
        
        # Remove existing ROIs
        if self.eels_roi is not None:
            self.eels_plot.removeItem(self.eels_roi)
        if self.ndata_roi is not None:
            self.ndata_plot.removeItem(self.ndata_roi)
        
        # Create new ROIs based on mode
        self.setup_roi_widgets()
        
        # Position ROIs
        if self.roi_mode and self.current_roi is not None:
            if len(self.current_roi) == 5:
                x, y, w, h, angle = self.current_roi
            else:
                # Handle old format for backward compatibility
                x, y, w, h = self.current_roi
                angle = 0.0
            
            self.eels_roi.setPos([x, y])
            self.eels_roi.setSize([w, h])
            self.eels_roi.setAngle(angle)
            self.ndata_roi.setPos([x, y])
            self.ndata_roi.setSize([w, h])
            self.ndata_roi.setAngle(angle)
        elif not self.roi_mode and self.current_roi is not None:
            x, y = self.current_roi[:2]  # Handle both formats
            self.eels_roi.setPos([x, y])
            self.ndata_roi.setPos([x, y])
        
        # Add ROIs to plots
        self.eels_plot.addItem(self.eels_roi)
        if self.ndata_data is not None:
            self.ndata_plot.addItem(self.ndata_roi)
    
    def update_displays(self):
        """Update all display elements"""
        # Update EELS image
        if self.eels_data is not None:
            self.update_eels_image()
        
        # Update ndata image
        if self.ndata_data is not None:
            self.update_ndata_image()
        
        # Update spectrum
        if self.eels_data is not None:
            self.update_spectrum()
    
    def update_4d_displays(self):
        """Update 4D STEM displays"""
        if self.eels_data is None:
            return
        
        # Create scan overview image (integrated over detector)
        scan_overview = np.sum(self.eels_data, axis=(2, 3))  # Sum over energy and detector
        self.scan_image_item.setImage(scan_overview.T)
        
        # Initialize cursor at center
        h, w = self.eels_data.shape[:2]
        center_x, center_y = w // 2, h // 2
        self.scan_cursor.setPos([center_x, center_y])
        
        # Center virtual detectors on diffraction pattern
        self.center_virtual_detectors()
        
        # Update diffraction pattern for center position
        self.on_scan_position_changed()
        
        # Calculate virtual images
        self.calculate_virtual_images()
    
    def update_eels_image(self):
        """Update EELS integrated image"""
        if self.eels_data is None:
            return
        
        # Integrate over current energy range
        energy_pixels = self.eels_data.shape[2]
        start_idx = max(0, int(self.current_energy_range[0]))
        end_idx = min(energy_pixels, int(self.current_energy_range[1]))
        
        # Integrate EELS data over energy and detector dimensions
        integrated = np.sum(self.eels_data[:, :, start_idx:end_idx, :], axis=(2, 3))
        
        # Set image data (PyQtGraph automatically handles scaling and display)
        self.eels_image_item.setImage(integrated.T)  # Transpose for correct orientation
    
    def update_ndata_image(self):
        """Update ndata image display"""
        if self.ndata_data is None:
            return
        
        # Set image data
        self.ndata_image_item.setImage(self.ndata_data.T)  # Transpose for correct orientation
    
    def update_spectrum(self):
        """Update spectrum plot"""
        if self.eels_data is None:
            return
        
        # Calculate spectrum based on current ROI
        if self.roi_mode and self.current_roi is not None:
            # ROI mode - average over potentially rotated rectangular region
            if len(self.current_roi) == 5:
                x, y, w, h, angle = self.current_roi
            else:
                # Handle old format
                x, y, w, h = self.current_roi
                angle = 0.0
            
            # For now, simplify by using bounding box for rotated ROIs
            # (Full rotated extraction would be more complex)
            if abs(angle) < 1e-6:  # No rotation or negligible rotation
                # Standard rectangular extraction
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Ensure bounds are valid
                x = max(0, min(x, self.eels_data.shape[1] - 1))
                y = max(0, min(y, self.eels_data.shape[0] - 1))
                x2 = max(x + 1, min(x + w, self.eels_data.shape[1]))
                y2 = max(y + 1, min(y + h, self.eels_data.shape[0]))
                
                # Extract ROI and average
                roi_data = self.eels_data[y:y2, x:x2, :, :]
                avg_spectrum = np.mean(roi_data, axis=(0, 1, 3))
            else:
                # For rotated ROI, use center point for now (could be enhanced later)
                center_x, center_y = int(x + w/2), int(y + h/2)
                center_x = max(0, min(center_x, self.eels_data.shape[1] - 1))
                center_y = max(0, min(center_y, self.eels_data.shape[0] - 1))
                
                # Use a small region around center for rotated case
                region_size = min(int(w//4), int(h//4), 5)  # Small region
                x1 = max(0, center_x - region_size)
                x2 = min(self.eels_data.shape[1], center_x + region_size)
                y1 = max(0, center_y - region_size)
                y2 = min(self.eels_data.shape[0], center_y + region_size)
                
                roi_data = self.eels_data[y1:y2, x1:x2, :, :]
                avg_spectrum = np.mean(roi_data, axis=(0, 1, 3))
            
        elif not self.roi_mode and self.current_roi is not None:
            # Crosshair mode - single point spectrum
            x, y = self.current_roi
            x, y = int(max(0, min(x, self.eels_data.shape[1] - 1))), int(max(0, min(y, self.eels_data.shape[0] - 1)))
            
            # Extract single point spectrum and average over detector
            point_data = self.eels_data[y, x, :, :]
            avg_spectrum = np.mean(point_data, axis=1)
        else:
            return
        
        # Create energy axis
        energy_axis = np.arange(len(avg_spectrum))
        
        # Apply log scale if enabled
        if self.log_scale:
            spectrum_data = np.log10(np.maximum(avg_spectrum, 1e-10))
        else:
            spectrum_data = avg_spectrum
        
        # Update the spectrum curve (this is much faster than matplotlib!)
        self.spectrum_curve.setData(energy_axis, spectrum_data)
    
    def on_roi_changed(self, source='eels'):
        """Handle ROI change events (including position, size, and rotation)"""
        roi = self.eels_roi if source == 'eels' else self.ndata_roi
        other_roi = self.ndata_roi if source == 'eels' else self.eels_roi
        
        # Get ROI state (position, size, and angle)
        pos = roi.pos()
        size = roi.size()
        angle = roi.angle()
        
        # Store ROI state (extended to include angle)
        self.current_roi = (pos[0], pos[1], size[0], size[1], angle)
        
        # Sync the other ROI (prevent infinite recursion)
        if other_roi is not None:
            other_roi.blockSignals(True)
            other_roi.setPos(pos)
            other_roi.setSize(size)
            other_roi.setAngle(angle)  # Sync rotation too
            other_roi.blockSignals(False)
        
        # Update spectrum in real-time
        self.update_spectrum()
    
    def on_crosshair_changed(self, source='eels'):
        """Handle crosshair position change events"""
        roi = self.eels_roi if source == 'eels' else self.ndata_roi
        other_roi = self.ndata_roi if source == 'eels' else self.eels_roi
        
        # Get crosshair position
        pos = roi.pos()
        self.current_roi = (pos[0], pos[1])
        
        # Sync the other crosshair
        if other_roi is not None:
            other_roi.blockSignals(True)
            other_roi.setPos(pos)
            other_roi.blockSignals(False)
        
        # Update spectrum in real-time
        self.update_spectrum()
    
    def on_energy_range_changed(self):
        """Handle energy range selection change"""
        # Get new energy range
        self.current_energy_range = self.energy_region.getRegion()
        
        # Update EELS image with new energy integration
        self.update_eels_image()
    
    def on_log_toggle(self, checked):
        """Handle log scale toggle"""
        self.log_scale = checked
        
        # Update spectrum display
        self.update_spectrum()
        
        # Update y-axis label
        if self.log_scale:
            self.spectrum_plot.setLabel('left', 'log₁₀(Intensity)')
        else:
            self.spectrum_plot.setLabel('left', 'Intensity')
    
    def on_roi_mode_toggle(self, checked):
        """Handle ROI mode toggle"""
        old_mode = self.roi_mode
        self.roi_mode = checked
        
        # Convert between ROI types if data is loaded
        if self.eels_data is not None:
            if old_mode and not self.roi_mode:
                # Convert from ROI to crosshair (use ROI center)
                if self.current_roi is not None:
                    if len(self.current_roi) == 5:
                        x, y, w, h, angle = self.current_roi
                    else:
                        x, y, w, h = self.current_roi
                    self.current_roi = (x + w/2, y + h/2)
            elif not old_mode and self.roi_mode:
                # Convert from crosshair to ROI (create small ROI around point)
                if self.current_roi is not None:
                    x, y = self.current_roi[:2]
                    roi_size = 20  # Default ROI size
                    self.current_roi = (x - roi_size/2, y - roi_size/2, roi_size, roi_size, 0.0)
            
            # Update ROI widgets
            self.setup_roi_positions()
            self.update_spectrum()
    
    def center_virtual_detectors(self):
        """Center virtual detectors on the diffraction pattern"""
        if self.eels_data is None:
            return
        
        # Get diffraction pattern center
        det_h, det_w = self.eels_data.shape[2], self.eels_data.shape[3]
        center_y, center_x = det_h // 2, det_w // 2
        
        # Center BF detector (smaller disk)
        bf_radius = min(det_h, det_w) // 8  # 1/8 of detector size
        self.bf_detector.setPos([center_x - bf_radius, center_y - bf_radius])
        self.bf_detector.setSize([2 * bf_radius, 2 * bf_radius])
        
        # Center DF detectors (annular ring)
        df_outer_radius = min(det_h, det_w) // 4  # 1/4 of detector size
        df_inner_radius = min(det_h, det_w) // 6  # 1/6 of detector size
        
        self.df_detector_outer.setPos([center_x - df_outer_radius, center_y - df_outer_radius])
        self.df_detector_outer.setSize([2 * df_outer_radius, 2 * df_outer_radius])
        
        self.df_detector_inner.setPos([center_x - df_inner_radius, center_y - df_inner_radius])
        self.df_detector_inner.setSize([2 * df_inner_radius, 2 * df_inner_radius])
    
    def on_detector_changed(self):
        """Handle virtual detector geometry changes - use timer to delay recalculation"""
        # For performance, only update when we're actively viewing 4D tab
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 1:  # 4D STEM tab
            # Stop any existing timer and restart it with a 300ms delay
            # This allows smooth resizing before triggering the expensive recalculation
            self.detector_update_timer.stop()
            self.detector_update_timer.start(300)  # 300ms delay
    
    def on_detector_overlay_toggle(self, checked):
        """Toggle visibility of virtual detector overlays"""
        self.detector_overlays_visible = checked
        self.bf_detector.setVisible(checked)
        self.df_detector_outer.setVisible(checked)
        self.df_detector_inner.setVisible(checked)
    
    def delayed_calculate_virtual_images(self):
        """Called by timer after detector geometry changes to recalculate virtual images"""
        self.calculate_virtual_images()
    
    def calculate_virtual_images(self):
        """Calculate BF and DF virtual images for all scan positions"""
        if self.eels_data is None:
            return
        
        # Get detector geometries
        bf_pos = self.bf_detector.pos()
        bf_size = self.bf_detector.size()
        bf_center = [bf_pos[0] + bf_size[0]/2, bf_pos[1] + bf_size[1]/2]
        bf_radius = bf_size[0] / 2
        
        df_outer_pos = self.df_detector_outer.pos()
        df_outer_size = self.df_detector_outer.size()
        df_outer_center = [df_outer_pos[0] + df_outer_size[0]/2, df_outer_pos[1] + df_outer_size[1]/2]
        df_outer_radius = df_outer_size[0] / 2
        
        df_inner_pos = self.df_detector_inner.pos()
        df_inner_size = self.df_detector_inner.size()
        df_inner_center = [df_inner_pos[0] + df_inner_size[0]/2, df_inner_pos[1] + df_inner_size[1]/2]
        df_inner_radius = df_inner_size[0] / 2
        
        # Create coordinate grids for detector pixels
        det_h, det_w = self.eels_data.shape[2], self.eels_data.shape[3]
        y_coords, x_coords = np.ogrid[:det_h, :det_w]
        
        # Create BF detector mask (disk)
        bf_mask = ((x_coords - bf_center[0])**2 + (y_coords - bf_center[1])**2) <= bf_radius**2
        
        # Create DF detector mask (annulus)
        dist_from_df_center = (x_coords - df_outer_center[0])**2 + (y_coords - df_outer_center[1])**2
        df_mask = (dist_from_df_center <= df_outer_radius**2) & (dist_from_df_center >= df_inner_radius**2)
        
        # Calculate virtual images
        scan_h, scan_w = self.eels_data.shape[:2]
        bf_image = np.zeros((scan_h, scan_w))
        df_image = np.zeros((scan_h, scan_w))
        
        # Apply masks and sum for each scan position
        for y in range(scan_h):
            for x in range(scan_w):
                diffraction_pattern = self.eels_data[y, x, :, :]
                bf_image[y, x] = np.sum(diffraction_pattern[bf_mask])
                df_image[y, x] = np.sum(diffraction_pattern[df_mask])
        
        # Display virtual images
        self.bf_image_item.setImage(bf_image.T)
        self.df_image_item.setImage(df_image.T)
    
    def on_scan_position_changed(self):
        """Handle scan position change in 4D STEM mode"""
        if self.eels_data is None:
            return
        
        # Get cursor position
        pos = self.scan_cursor.pos()
        scan_x, scan_y = int(pos[0]), int(pos[1])
        
        # Ensure position is within bounds
        scan_x = max(0, min(scan_x, self.eels_data.shape[1] - 1))
        scan_y = max(0, min(scan_y, self.eels_data.shape[0] - 1))
        
        # Extract and display diffraction pattern at this position
        diffraction_pattern = self.eels_data[scan_y, scan_x, :, :]
        self.diffraction_image_item.setImage(diffraction_pattern.T)
    
    def reset_view(self):
        """Reset all plot views to show full data"""
        self.eels_plot.autoRange()
        self.ndata_plot.autoRange()
        self.spectrum_plot.autoRange()
        # Reset 4D STEM views
        self.scan_plot.autoRange()
        self.diffraction_plot.autoRange()
        self.bf_plot.autoRange()
        self.df_plot.autoRange()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "MIB Data Analysis Suite - PyQtGraph Edition\n\n"
                         "High-performance tool for viewing MIB data with\n"
                         "real-time interactive analysis.\n\n"
                         "EELS Analysis Features:\n"
                         "• Real-time ROI selection (resizable, rotatable)\n"
                         "• Energy range integration\n"
                         "• Hardware-accelerated graphics\n\n"
                         "4D STEM Features:\n"
                         "• Real-time diffraction pattern viewing\n"
                         "• Interactive scan position selection\n"
                         "• Virtual imaging capabilities\n"
                         "• Multi-panel synchronized analysis")

def main():
    """Main entry point for the PyQtGraph application"""
    # Enable high DPI support before creating QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    app = QApplication(sys.argv)
    
    viewer = MibViewerPyQtGraph()
    viewer.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()