#!/usr/bin/env python3
"""
MIB Viewer Application - PyQtGraph Version
A high-performance GUI application for viewing MIB EELS data with real-time interactions.
"""

import sys
import os
import json
import zipfile
import datetime
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QMenuBar, QAction, QStatusBar, QCheckBox, QSizePolicy,
                             QSplitter, QGroupBox, QDesktopWidget, QTabWidget, QLineEdit,
                             QComboBox, QProgressBar, QTextEdit, QGridLayout, QFrame, 
                             QRadioButton, QButtonGroup, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject

# Import MIB loading functions
try:
    # Try relative import (when run as module)
    from ..io.mib_loader import (load_mib, load_emd, load_data_file, get_data_file_info, 
                                     get_mib_properties, auto_detect_scan_size, MibProperties,
                                     detect_experiment_type, apply_data_processing, calculate_processed_size,
                                     get_valid_bin_factors)
    from ..io.mib_to_emd_converter import MibToEmdConverter
except ImportError:
    # Fall back for direct execution
    from mib_viewer.io.mib_loader import (load_mib, load_emd, load_data_file, get_data_file_info,
                                          get_mib_properties, auto_detect_scan_size, MibProperties,
                                          detect_experiment_type, apply_data_processing, calculate_processed_size,
                                          get_valid_bin_factors)
    from mib_viewer.io.mib_to_emd_converter import MibToEmdConverter

# Configure PyQtGraph
pg.setConfigOptions(antialias=True, useOpenGL=True)

class ConversionWorker(QObject):
    """Worker class for running MIB to EMD conversion in a separate thread"""
    
    # Define signals
    progress_updated = pyqtSignal(int, str)  # progress percentage, status message
    conversion_finished = pyqtSignal(dict)   # conversion statistics
    conversion_failed = pyqtSignal(str)      # error message
    log_message_signal = pyqtSignal(str, str)  # message, level - Qt-safe logging signal
    
    def __init__(self, input_path, output_path, compression, compression_level, processing_options=None, log_callback=None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.compression = compression
        self.compression_level = compression_level
        self.log_callback = log_callback
        self.processing_options = processing_options or {}
        self.cancelled = False
    
    def qt_safe_log(self, message, level="INFO"):
        """Qt-safe logging that emits signal instead of direct callback"""
        self.log_message_signal.emit(message, level)
    
    def run_conversion(self):
        """Run the conversion process with progress updates"""
        try:
            # Create converter
            converter = MibToEmdConverter(
                compression=self.compression,
                compression_level=self.compression_level,
                log_callback=self.qt_safe_log  # Use Qt-safe logging
            )
            
            # Phase 1: Analyze file (5% of progress)
            self.progress_updated.emit(5, "Analyzing MIB file...")
            if self.cancelled:
                return
            
            # Get file info for progress estimation
            file_info = get_data_file_info(self.input_path)
            file_size_gb = file_info['size_gb']
            
            # Phase 2: Load MIB data (20% of progress)
            self.progress_updated.emit(20, "Loading MIB data...")
            if self.cancelled:
                return
            
            # Simulate some progress during loading for large files
            import time
            if file_size_gb > 1.0:  # For files > 1GB, show intermediate progress
                for i in range(20, 40, 5):
                    if self.cancelled:
                        return
                    self.progress_updated.emit(i, "Loading MIB data...")
                    time.sleep(0.5)  # Small delay to show progress
            
            # Phase 3: Convert to EMD (60% of progress)
            self.progress_updated.emit(40, "Converting to EMD format...")
            if self.cancelled:
                return
            
            # Start actual conversion
            start_time = time.time()
            
            # We'll update progress during conversion by estimating based on time
            conversion_start = time.time()
            estimated_time = max(5, int(file_size_gb * 10))  # 10s per GB estimate
            
            # Start conversion in a way that allows progress updates
            stats = self.convert_with_progress(converter, estimated_time)
            
            if self.cancelled:
                return
            
            # Phase 4: Finalizing (100%)
            self.progress_updated.emit(100, "Conversion completed successfully!")
            
            # Add timing information to stats
            end_time = time.time()
            stats['actual_time'] = end_time - start_time
            
            # Emit success signal
            self.conversion_finished.emit(stats)
            
        except Exception as e:
            self.conversion_failed.emit(str(e))
    
    def convert_with_progress(self, converter, estimated_time):
        """Run conversion with simulated progress updates"""
        import time
        from threading import Thread
        
        # Store the result
        conversion_result = {}
        conversion_error = None
        
        def do_conversion():
            nonlocal conversion_result, conversion_error
            try:
                conversion_result['stats'] = converter.convert_to_emd(self.input_path, self.output_path, processing_options=self.processing_options)
            except Exception as e:
                conversion_error = e
        
        # Start conversion in background thread
        conversion_thread = Thread(target=do_conversion)
        conversion_thread.start()
        
        # Update progress while conversion runs
        start_time = time.time()
        progress_start = 40  # Starting progress
        progress_end = 95     # Ending progress (leave 5% for finalization)
        
        while conversion_thread.is_alive():
            if self.cancelled:
                return None
            
            elapsed = time.time() - start_time
            estimated_progress = min(elapsed / estimated_time, 1.0)
            current_progress = int(progress_start + (progress_end - progress_start) * estimated_progress)
            
            self.progress_updated.emit(current_progress, f"Converting... ({elapsed:.0f}s elapsed)")
            
            time.sleep(1)  # Update every second
        
        # Wait for thread to complete
        conversion_thread.join()
        
        if conversion_error:
            raise conversion_error
        
        return conversion_result['stats']
    
    def cancel(self):
        """Cancel the conversion"""
        self.cancelled = True


class MibViewerPyQtGraph(QMainWindow):
    """
    High-performance MIB EELS viewer using PyQtGraph for real-time interactions
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIB EELS Viewer - PyQtGraph")
        self.resize(800, 800)  # Make window square and appropriate for smaller monitors
        self.center_on_screen()
        
        # Data storage
        self.eels_data = None
        self.stem4d_data = None  # Add 4D STEM data storage
        self.ndata_data = None
        self.eels_filename = ""
        self.ndata_filename = ""
        
        # Current selections
        self.current_roi = None  # Will store ROI bounds
        self.current_energy_range = (10, 200)  # Default energy range
        
        # Plot focus tracking for FFT
        self.active_plot = None
        self.plot_widgets = {}  # Will be populated after UI setup
        
        # FFT window tracking
        self.fft_windows = {}  # Dict to track open FFT windows by plot name
        
        # FFT ROI tracking
        self.fft_rois = {}  # Track FFT ROIs: (plot_name, window_id) -> roi_instance
        self.fft_roi_counter = 0  # Unique window IDs for FFT ROIs
        
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
        
        # Create main splitter for tabs and log panel
        main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        main_splitter.addWidget(self.tab_widget)
        
        # Create log panel
        self.create_log_panel(main_splitter)
        
        # Create EELS tab
        self.eels_tab = self.create_eels_tab()
        self.tab_widget.addTab(self.eels_tab, "EELS Analysis")
        
        # Create 4D STEM tab
        self.stem4d_tab = self.create_4d_stem_tab()
        self.tab_widget.addTab(self.stem4d_tab, "4D STEM")
        
        # Create MIB to EMD conversion tab
        self.conversion_tab = self.create_conversion_tab()
        self.tab_widget.addTab(self.conversion_tab, "MIB → EMD")
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load MIB or EMD file to begin")
        
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
        
        # Colormap selection
        colormap_label = QLabel("Colormap:")
        controls_group_layout.addWidget(colormap_label)
        
        self.colormap_combo = QComboBox()
        # Include grayscale (default PyQtGraph behavior) plus color options
        self.colormap_combo.addItems([
            "gray", "viridis", "plasma", "inferno", "magma", 
            "turbo", "cividis"
        ])
        self.colormap_combo.setCurrentText("gray")  # Default (original behavior)
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        self.colormap_combo.setToolTip("Select colormap for image displays")
        controls_group_layout.addWidget(self.colormap_combo)
        
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
    
    def create_conversion_tab(self):
        """Create the MIB to EMD conversion tab"""
        # Create main widget and layout
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        
        # Input file section
        input_group = QGroupBox("Input File")
        input_layout = QGridLayout(input_group)
        
        # Input file selection
        self.input_file_path = QLineEdit()
        self.input_file_path.setPlaceholderText("No file selected...")
        self.input_file_path.setReadOnly(True)
        self.input_browse_btn = QPushButton("Browse...")
        self.input_browse_btn.clicked.connect(self.browse_input_file)
        
        input_layout.addWidget(QLabel("MIB File:"), 0, 0)
        input_layout.addWidget(self.input_file_path, 0, 1)
        input_layout.addWidget(self.input_browse_btn, 0, 2)
        
        # File info display
        self.file_info_label = QLabel("File info will appear here after selection")
        self.file_info_label.setStyleSheet("color: #666; font-style: italic;")
        input_layout.addWidget(self.file_info_label, 1, 0, 1, 3)
        
        # Output settings section
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout(output_group)
        
        # Output folder
        self.output_folder_path = QLineEdit()
        self.output_folder_path.setPlaceholderText("Same as input file")
        self.output_folder_browse_btn = QPushButton("Browse...")
        self.output_folder_browse_btn.clicked.connect(self.browse_output_folder)
        
        output_layout.addWidget(QLabel("Output Folder:"), 0, 0)
        output_layout.addWidget(self.output_folder_path, 0, 1)
        output_layout.addWidget(self.output_folder_browse_btn, 0, 2)
        
        # Output filename
        self.output_filename = QLineEdit()
        self.output_filename.setPlaceholderText("Will be auto-generated")
        output_layout.addWidget(QLabel("Output File:"), 1, 0)
        output_layout.addWidget(self.output_filename, 1, 1, 1, 2)
        
        # Compression settings
        compression_layout = QHBoxLayout()
        self.compression_combo = QComboBox()
        self.compression_combo.addItems(['gzip', 'szip', 'lzf', 'none'])
        self.compression_combo.setCurrentText('gzip')
        self.compression_combo.currentTextChanged.connect(self.update_compression_level)
        
        self.compression_level_combo = QComboBox()
        self.compression_level_combo.addItems(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.compression_level_combo.setCurrentText('6')
        
        compression_layout.addWidget(QLabel("Compression:"))
        compression_layout.addWidget(self.compression_combo)
        compression_layout.addWidget(QLabel("Level:"))
        compression_layout.addWidget(self.compression_level_combo)
        compression_layout.addStretch()
        
        output_layout.addLayout(compression_layout, 2, 0, 1, 3)
        
        # Data Processing section
        processing_group = QGroupBox("Data Processing")
        processing_layout = QVBoxLayout(processing_group)
        
        # Experiment type display
        self.experiment_type_label = QLabel("Select input file to see data type")
        self.experiment_type_label.setStyleSheet("color: #666; font-style: italic;")
        processing_layout.addWidget(self.experiment_type_label)
        
        # Processing options (initially hidden)
        self.processing_options_widget = QWidget()
        self.processing_options_widget.setVisible(False)
        processing_options_layout = QVBoxLayout(self.processing_options_widget)
        
        # Radio button group for processing mode
        self.processing_button_group = QButtonGroup()
        
        # EELS Y-summing option
        self.sum_y_radio = QRadioButton("Sum in Y direction (EELS)")
        self.processing_button_group.addButton(self.sum_y_radio, 0)
        processing_options_layout.addWidget(self.sum_y_radio)
        
        # 4D binning option
        binning_widget = QWidget()
        binning_layout = QHBoxLayout(binning_widget)
        binning_layout.setContentsMargins(0, 0, 0, 0)
        
        self.binning_radio = QRadioButton("Binning:")
        self.processing_button_group.addButton(self.binning_radio, 1)
        binning_layout.addWidget(self.binning_radio)
        
        self.bin_factor_combo = QComboBox()
        self.bin_factor_combo.addItems(['1x1', '2x2', '4x4', '8x8', '16x16'])
        self.bin_factor_combo.setCurrentText('2x2')
        binning_layout.addWidget(self.bin_factor_combo)
        
        self.bin_method_combo = QComboBox()
        self.bin_method_combo.addItems(['Mean', 'Sum'])
        self.bin_method_combo.setCurrentText('Mean')
        binning_layout.addWidget(self.bin_method_combo)
        
        binning_layout.addStretch()
        processing_options_layout.addWidget(binning_widget)
        
        # Advanced mode (both operations)
        self.advanced_radio = QRadioButton("Advanced: Bin then Sum Y")
        self.processing_button_group.addButton(self.advanced_radio, 2)
        processing_options_layout.addWidget(self.advanced_radio)
        
        advanced_warning = QLabel("⚠️ Warning: Unusual processing combination")
        advanced_warning.setStyleSheet("color: orange; font-size: 9pt; margin-left: 20px;")
        processing_options_layout.addWidget(advanced_warning)
        
        # No processing option (default)
        self.no_processing_radio = QRadioButton("No processing")
        self.no_processing_radio.setChecked(True)  # Default selection
        self.processing_button_group.addButton(self.no_processing_radio, 3)
        processing_options_layout.addWidget(self.no_processing_radio)
        
        # Connect signals for UI updates
        self.processing_button_group.buttonToggled.connect(self.update_processing_ui)
        self.bin_factor_combo.currentTextChanged.connect(self.update_conversion_preview)
        self.bin_method_combo.currentTextChanged.connect(self.update_conversion_preview)
        
        processing_layout.addWidget(self.processing_options_widget)
        
        # Preview section
        preview_group = QGroupBox("Conversion Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Select input file to see conversion preview")
        self.preview_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        preview_layout.addWidget(self.preview_label)
        
        # Conversion section
        conversion_group = QGroupBox("Conversion")
        conversion_layout = QVBoxLayout(conversion_group)
        
        # Progress bar
        self.conversion_progress = QProgressBar()
        self.conversion_progress.setVisible(False)
        conversion_layout.addWidget(self.conversion_progress)
        
        # Status label
        self.conversion_status = QLabel("")
        conversion_layout.addWidget(self.conversion_status)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.convert_btn = QPushButton("Convert to EMD")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.start_conversion)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        
        self.open_converted_btn = QPushButton("Open Converted File in Viewer")
        self.open_converted_btn.setEnabled(False)
        self.open_converted_btn.clicked.connect(self.open_converted_file)
        
        button_layout.addWidget(self.convert_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.open_converted_btn)
        
        conversion_layout.addLayout(button_layout)
        
        # Add all groups to main layout
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(processing_group)
        main_layout.addWidget(preview_group)
        main_layout.addWidget(conversion_group)
        main_layout.addStretch()
        
        # Initialize conversion state
        self.converter = None
        self.conversion_thread = None
        self.last_converted_file = None
        
        return tab_widget
    
    def create_log_panel(self, parent_splitter):
        """Create the resizable log panel at the bottom"""
        # Create log group box
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        # Create log text widget
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Remove height constraints to allow expansion with frame
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Style the log text widget
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                padding: 5px;
            }
        """)
        
        log_layout.addWidget(self.log_text)
        
        # Create button layout for log controls
        log_button_layout = QHBoxLayout()
        
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.clear_log)
        
        self.save_log_btn = QPushButton("Save Log...")
        self.save_log_btn.clicked.connect(self.save_log)
        
        log_button_layout.addWidget(self.clear_log_btn)
        log_button_layout.addWidget(self.save_log_btn)
        log_button_layout.addStretch()
        
        log_layout.addLayout(log_button_layout)
        
        # Add to splitter
        parent_splitter.addWidget(log_group)
        
        # Set splitter proportions - tabs get most space, log gets smaller portion
        parent_splitter.setSizes([800, 200])  # Tabs: 800px, Log: 200px
        
        # Add initial welcome message
        self.log_message("MIB Viewer started - Ready for data analysis and conversion")
    
    def log_message(self, message, level="INFO"):
        """Add a message to the log panel with timestamp"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Format message with level and timestamp
        if level == "ERROR":
            formatted_message = f"<span style='color: red;'>[{timestamp}] ERROR: {message}</span>"
        elif level == "WARNING":
            formatted_message = f"<span style='color: orange;'>[{timestamp}] WARNING: {message}</span>"
        elif level == "SUCCESS":
            formatted_message = f"<span style='color: green;'>[{timestamp}] SUCCESS: {message}</span>"
        else:  # INFO
            formatted_message = f"[{timestamp}] {message}"
        
        # Add to log
        self.log_text.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """Clear the log panel"""
        self.log_text.clear()
        self.log_message("Log cleared")
    
    def save_log(self):
        """Save log contents to a file"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log File",
            f"mib_viewer_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    # Get plain text version of log (no HTML formatting)
                    plain_text = self.log_text.toPlainText()
                    f.write(plain_text)
                self.log_message(f"Log saved to: {filename}", "SUCCESS")
            except Exception as e:
                self.log_message(f"Failed to save log: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Error", f"Failed to save log file:\n{str(e)}")
    
    def setup_plots(self):
        """Initialize PyQtGraph plot items"""
        # Start with default grayscale (no colormap) to preserve original behavior
        
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
        # Add DF detectors first (bottom layers) so BF detector can be clicked when overlapping
        
        # DF detector - movable annular ring  
        # Using separate outer and inner circles that move together
        self.df_detector_outer = pg.CircleROI([0, 0], [100, 100], pen='b', movable=True, removable=False)
        self.df_detector_outer.sigRegionChanged.connect(self.on_df_outer_changed)
        self.diffraction_plot.addItem(self.df_detector_outer)
        
        # Inner boundary of annular detector - also movable but coupled to outer
        self.df_detector_inner = pg.CircleROI([0, 0], [60, 60], pen='b', movable=True, removable=False)
        self.df_detector_inner.sigRegionChanged.connect(self.on_df_inner_changed)
        self.diffraction_plot.addItem(self.df_detector_inner)
        
        # BF detector - movable and resizable disk (added last so it's on top)
        self.bf_detector = pg.CircleROI([0, 0], [50, 50], pen='g', movable=True, removable=False)
        self.bf_detector.sigRegionChanged.connect(self.on_detector_changed)
        self.diffraction_plot.addItem(self.bf_detector)
        
        # Flag to prevent infinite recursion during coupled movement
        self._updating_df_detectors = False
        
        # Snap-to-center functionality
        self._shift_pressed = False
        self._snap_threshold = 20  # pixels - distance from center to trigger snap
        
        # Initialize detector visibility
        self.detector_overlays_visible = True
        
        # Initialize detector update timer (for delayed recalculation)
        self.detector_update_timer = QTimer()
        self.detector_update_timer.setSingleShot(True)
        self.detector_update_timer.timeout.connect(self.delayed_calculate_virtual_images)
        
        # Setup plot focus tracking after UI is created
        self.setup_plot_focus_tracking()
    
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
        
        load_mib_action = QAction('Load Data File (MIB/EMD)...', self)
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
        """Load a MIB or EMD file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select data file",
            "",
            "4D STEM files (*.mib *.emd);;MIB files (*.mib);;EMD files (*.emd);;All files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            # Determine file type and show appropriate loading message
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext == '.emd':
                self.status_bar.showMessage("Loading EMD file...")
                file_type = "EMD"
                self.log_message(f"Loading EMD file: {os.path.basename(filename)}")
            else:
                self.status_bar.showMessage("Loading MIB file...")
                file_type = "MIB"
                self.log_message(f"Loading MIB file: {os.path.basename(filename)}")
            QApplication.processEvents()
            
            # Load the data using universal loader
            raw_data = load_data_file(filename)
            
            # Log the loaded data dimensions
            self.log_message(f"Loaded data shape: {raw_data.shape} (scan_y, scan_x, detector_y, detector_x)")
            
            # Detect experiment type based on data shape
            experiment_type, exp_info = detect_experiment_type(raw_data.shape)
            self.log_message(f"Detected experiment type: {experiment_type} - {exp_info['detector_type']}")
            
            if experiment_type == "EELS":
                # Check if we need to sum along the shorter detector dimension
                sy, sx, dy, dx = raw_data.shape
                
                # For EELS data (dy != dx), check if we need to sum
                if min(dy, dx) > 1:  # 4D EELS - both detector dimensions > 1
                    if dy < dx:
                        # dy is shorter, sum along Y detector dimension (axis=2)
                        self.log_message(f"Auto-summing Y detector dimension: {dy}×{dx} → 1×{dx}")
                        summed_data = np.sum(raw_data, axis=2, keepdims=True)
                    else:
                        # dx is shorter, sum along X detector dimension (axis=3)
                        self.log_message(f"Auto-summing X detector dimension: {dy}×{dx} → {dy}×1")
                        summed_data = np.sum(raw_data, axis=3, keepdims=True)
                    # Flip energy axis (the longer dimension becomes the energy axis)
                    self.eels_data = summed_data[:, :, :, ::-1]
                else:
                    # 3D EELS - already summed, just flip energy axis
                    self.eels_data = raw_data[:, :, :, ::-1]
                
                self.eels_filename = os.path.basename(filename)
                self.eels_label.setText(f"EELS File: {self.eels_filename} ({file_type})")
                
                # Auto-switch to EELS tab
                self.tab_widget.setCurrentIndex(0)  # EELS is first tab
                
            elif experiment_type == "4D_STEM":
                # Store as 4D STEM data (no energy axis flip needed)
                self.stem4d_data = raw_data
                
                # TODO: Add 4D STEM file label to the 4D STEM tab
                self.log_message(f"Loaded 4D STEM data: {os.path.basename(filename)} ({file_type})")
                
                # Auto-switch to 4D STEM tab
                self.tab_widget.setCurrentIndex(1)  # 4D STEM is second tab
                
            else:
                # Unknown data type - default to EELS behavior for backward compatibility
                self.eels_data = raw_data[:, :, :, ::-1]
                self.eels_filename = os.path.basename(filename)
                self.eels_label.setText(f"Unknown Data: {self.eels_filename} ({file_type})")
                self.tab_widget.setCurrentIndex(0)
            
            # Initialize ROI to center of image  
            if experiment_type == "EELS" and self.eels_data is not None:
                h, w = self.eels_data.shape[:2]
            elif experiment_type == "4D_STEM" and self.stem4d_data is not None:
                h, w = self.stem4d_data.shape[:2]
            else:
                # Fallback for unknown data type
                h, w = self.eels_data.shape[:2] if self.eels_data is not None else (100, 100)
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
            self.status_bar.showMessage(f"Loaded {file_type} file: {self.eels_filename}")
            self.log_message(f"Successfully loaded {file_type} file: {self.eels_filename}", "SUCCESS")
            
        except Exception as e:
            error_msg = f"Failed to load data file: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.status_bar.showMessage("Ready")
            self.log_message(error_msg, "ERROR")
    
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
        """Update display elements based on data type"""
        # Only update EELS displays if EELS data is loaded
        if self.eels_data is not None:
            self.update_eels_image()
            self.update_spectrum()
        
        # Only update 4D STEM displays if 4D STEM data is loaded
        if self.stem4d_data is not None:
            self.update_4d_displays()
        
        # Update ndata image (this is separate data)
        if self.ndata_data is not None:
            self.update_ndata_image()
    
    def update_4d_displays(self):
        """Update 4D STEM displays"""
        if self.stem4d_data is None:
            return
        
        # Create scan overview image (integrated over detector)
        scan_overview = np.sum(self.stem4d_data, axis=(2, 3))  # Sum over detector dimensions
        self.scan_image_item.setImage(scan_overview.T)
        
        # Initialize cursor at center
        h, w = self.stem4d_data.shape[:2]
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
        # After transpose, EELS energy axis is in dimension 3 (last dimension)
        energy_pixels = self.eels_data.shape[3]
        start_idx = max(0, int(self.current_energy_range[0]))
        end_idx = min(energy_pixels, int(self.current_energy_range[1]))
        
        # Integrate EELS data over energy and detector dimensions
        # For EELS: integrate over all detector dimensions (2, 3)
        integrated = np.sum(self.eels_data[:, :, :, start_idx:end_idx], axis=(2, 3))
        
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
                # Average over spatial dimensions first
                spatial_avg = np.mean(roi_data, axis=(0, 1))  # Shape: (dy, dx)
                # Now average over the shorter detector dimension (non-energy axis)
                if spatial_avg.shape[0] > spatial_avg.shape[1]:
                    # Energy is in axis 0, average over axis 1
                    avg_spectrum = np.mean(spatial_avg, axis=1)
                else:
                    # Energy is in axis 1, average over axis 0
                    avg_spectrum = np.mean(spatial_avg, axis=0)
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
                # Average over spatial dimensions first
                spatial_avg = np.mean(roi_data, axis=(0, 1))  # Shape: (dy, dx)
                # Now average over the shorter detector dimension (non-energy axis)
                if spatial_avg.shape[0] > spatial_avg.shape[1]:
                    # Energy is in axis 0, average over axis 1
                    avg_spectrum = np.mean(spatial_avg, axis=1)
                else:
                    # Energy is in axis 1, average over axis 0
                    avg_spectrum = np.mean(spatial_avg, axis=0)
            
        elif not self.roi_mode and self.current_roi is not None:
            # Crosshair mode - single point spectrum
            x, y = self.current_roi
            x, y = int(max(0, min(x, self.eels_data.shape[1] - 1))), int(max(0, min(y, self.eels_data.shape[0] - 1)))
            
            # Extract single point spectrum and average over detector
            point_data = self.eels_data[y, x, :, :]  # Shape: (dy, dx) where one is 1, other is energy
            # Average over the shorter dimension (non-energy axis) to get energy spectrum
            if point_data.shape[0] > point_data.shape[1]:
                # Energy is in axis 0, average over axis 1
                avg_spectrum = np.mean(point_data, axis=1)
            else:
                # Energy is in axis 1, average over axis 0  
                avg_spectrum = np.mean(point_data, axis=0)
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
        
        # Update EELS image FFT if window is open
        self.update_fft_windows('eels_image')
    
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
    
    def on_colormap_changed(self, colormap_name):
        """Handle colormap selection change"""
        # Handle grayscale (default PyQtGraph behavior)
        if colormap_name == "gray":
            colormap = None  # No colormap = default grayscale
        else:
            # Get the colormap
            try:
                colormap = pg.colormap.get(colormap_name)
            except:
                # Fallback to no colormap (grayscale) if colormap not found
                colormap = None
        
        # Apply colormap to all image items
        image_items = [
            self.eels_image_item,
            self.ndata_image_item,
            self.scan_image_item,
            self.diffraction_image_item,
            self.bf_image_item,
            self.df_image_item
        ]
        
        for item in image_items:
            if item is not None:
                if colormap is None:
                    # Reset to default grayscale by removing the colormap property
                    # PyQtGraph uses a gray colormap as default when no colormap is set
                    try:
                        # Try to reset to default by setting a grayscale colormap
                        gray_colormap = pg.ColorMap([0, 1], [[0, 0, 0], [255, 255, 255]])
                        item.setColorMap(gray_colormap)
                    except:
                        # If that fails, try using a built-in grayscale approach
                        # Create a simple linear grayscale colormap
                        pos = [0.0, 1.0]
                        color = [[0, 0, 0, 255], [255, 255, 255, 255]]  # Black to white
                        gray_colormap = pg.ColorMap(pos, color)
                        item.setColorMap(gray_colormap)
                else:
                    item.setColorMap(colormap)
        
        # Update displays to show new colormap
        self.update_displays()
    
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
        if self.stem4d_data is None:
            return
        
        # Get diffraction pattern center
        det_h, det_w = self.stem4d_data.shape[2], self.stem4d_data.shape[3]
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
    
    def check_snap_to_center(self, detector):
        """Check if detector should snap to center when Shift is pressed"""
        if not self._shift_pressed or self.stem4d_data is None:
            return False
        
        # Get diffraction pattern center
        qy, qx = self.stem4d_data.shape[2], self.stem4d_data.shape[3]
        center_x, center_y = qx / 2, qy / 2
        
        # Get detector center
        detector_pos = detector.pos()
        detector_size = detector.size()
        detector_center_x = detector_pos[0] + detector_size[0] / 2
        detector_center_y = detector_pos[1] + detector_size[1] / 2
        
        # Calculate distance from center
        distance = np.sqrt((detector_center_x - center_x)**2 + (detector_center_y - center_y)**2)
        
        # If within snap threshold, snap to center
        if distance <= self._snap_threshold:
            new_x = center_x - detector_size[0] / 2
            new_y = center_y - detector_size[1] / 2
            detector.setPos([new_x, new_y])
            return True
        
        return False

    def on_detector_changed(self):
        """Handle BF detector geometry changes - use timer to delay recalculation"""
        # Check for snap-to-center first
        self.check_snap_to_center(self.bf_detector)
        
        # For performance, only update when we're actively viewing 4D tab
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 1:  # 4D STEM tab
            # Stop any existing timer and restart it with a 500ms delay
            # This allows smooth resizing before triggering the expensive recalculation
            self.detector_update_timer.stop()
            self.detector_update_timer.start(500)  # 500ms delay
    
    def on_df_outer_changed(self):
        """Handle DF outer detector changes - synchronize inner detector position"""
        if self._updating_df_detectors:
            return  # Prevent recursion
        
        self._updating_df_detectors = True
        try:
            # Check for snap-to-center first
            snapped = self.check_snap_to_center(self.df_detector_outer)
            
            # Get outer detector center (potentially updated by snap)
            outer_pos = self.df_detector_outer.pos()
            outer_size = self.df_detector_outer.size()
            outer_center_x = outer_pos[0] + outer_size[0] / 2
            outer_center_y = outer_pos[1] + outer_size[1] / 2
            
            # Keep inner detector centered within outer detector
            inner_size = self.df_detector_inner.size()
            new_inner_x = outer_center_x - inner_size[0] / 2
            new_inner_y = outer_center_y - inner_size[1] / 2
            
            self.df_detector_inner.setPos([new_inner_x, new_inner_y])
            
            # Trigger virtual image recalculation
            self.on_detector_changed()
            
        finally:
            self._updating_df_detectors = False
    
    def on_df_inner_changed(self):
        """Handle DF inner detector changes - synchronize outer detector position"""
        if self._updating_df_detectors:
            return  # Prevent recursion
            
        self._updating_df_detectors = True
        try:
            # Check for snap-to-center first
            snapped = self.check_snap_to_center(self.df_detector_inner)
            
            # Get inner detector center (potentially updated by snap)
            inner_pos = self.df_detector_inner.pos()
            inner_size = self.df_detector_inner.size()
            inner_center_x = inner_pos[0] + inner_size[0] / 2
            inner_center_y = inner_pos[1] + inner_size[1] / 2
            
            # Keep outer detector centered on inner detector
            outer_size = self.df_detector_outer.size()
            new_outer_x = inner_center_x - outer_size[0] / 2
            new_outer_y = inner_center_y - outer_size[1] / 2
            
            self.df_detector_outer.setPos([new_outer_x, new_outer_y])
            
            # Trigger virtual image recalculation
            self.on_detector_changed()
            
        finally:
            self._updating_df_detectors = False
    
    def on_detector_overlay_toggle(self, checked):
        """Toggle visibility of virtual detector overlays"""
        self.detector_overlays_visible = checked
        self.bf_detector.setVisible(checked)
        self.df_detector_outer.setVisible(checked)
        self.df_detector_inner.setVisible(checked)
    
    def delayed_calculate_virtual_images(self):
        """Called by timer after detector geometry changes to recalculate virtual images"""
        self.calculate_virtual_images()
        
        # Update virtual detector FFTs if windows are open
        self.update_fft_windows(['virtual_bf', 'virtual_df'])
    
    def calculate_virtual_images(self):
        """Calculate BF and DF virtual images for all scan positions"""
        if self.stem4d_data is None:
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
        det_h, det_w = self.stem4d_data.shape[2], self.stem4d_data.shape[3]
        y_coords, x_coords = np.ogrid[:det_h, :det_w]
        
        # Create BF detector mask (disk)
        bf_mask = ((x_coords - bf_center[0])**2 + (y_coords - bf_center[1])**2) <= bf_radius**2
        
        # Create DF detector mask (annulus)
        dist_from_df_center = (x_coords - df_outer_center[0])**2 + (y_coords - df_outer_center[1])**2
        df_mask = (dist_from_df_center <= df_outer_radius**2) & (dist_from_df_center >= df_inner_radius**2)
        
        # Calculate virtual images
        scan_h, scan_w = self.stem4d_data.shape[:2]
        bf_image = np.zeros((scan_h, scan_w))
        df_image = np.zeros((scan_h, scan_w))
        
        # Apply masks and sum for each scan position
        for y in range(scan_h):
            for x in range(scan_w):
                diffraction_pattern = self.stem4d_data[y, x, :, :]
                bf_image[y, x] = np.sum(diffraction_pattern[bf_mask])
                df_image[y, x] = np.sum(diffraction_pattern[df_mask])
        
        # Store virtual image data for FFT access
        self.bf_virtual_image = bf_image
        self.df_virtual_image = df_image
        
        # Display virtual images
        self.bf_image_item.setImage(bf_image.T)
        self.df_image_item.setImage(df_image.T)
    
    def on_scan_position_changed(self):
        """Handle scan position change in 4D STEM mode"""
        if self.stem4d_data is None:
            return
        
        # Get cursor position
        pos = self.scan_cursor.pos()
        scan_x, scan_y = int(pos[0]), int(pos[1])
        
        # Ensure position is within bounds
        scan_x = max(0, min(scan_x, self.stem4d_data.shape[1] - 1))
        scan_y = max(0, min(scan_y, self.stem4d_data.shape[0] - 1))
        
        # Extract and display diffraction pattern at this position
        diffraction_pattern = self.stem4d_data[scan_y, scan_x, :, :]
        self.diffraction_image_item.setImage(diffraction_pattern.T)
        
        # Update diffraction pattern FFT if window is open
        self.update_fft_windows('diffraction')
    
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
    
    # Plot focus tracking and FFT methods
    def setup_plot_focus_tracking(self):
        """Setup plot widgets dictionary and click tracking for FFT focus"""
        # Define all plot widgets that can be FFT'd
        self.plot_widgets = {
            'eels_image': self.eels_plot,
            'ndata_image': self.ndata_plot,
            'spectrum': self.spectrum_plot,
            'scan_overview': self.scan_plot,
            'diffraction': self.diffraction_plot,
            'virtual_bf': self.bf_plot,
            'virtual_df': self.df_plot
        }
        
        # Add click handlers to track focus
        for plot_name, plot_widget in self.plot_widgets.items():
            if plot_widget is not None:
                # Store original mouse press event
                original_mouse_press = plot_widget.mousePressEvent
                # Create new mouse press handler that tracks focus
                plot_widget.mousePressEvent = lambda event, name=plot_name, original=original_mouse_press: self.on_plot_clicked(name, event, original)
    
    def on_plot_clicked(self, plot_name, event, original_handler):
        """Handle plot click - set as active for FFT and update visual focus"""
        self.set_active_plot(plot_name)
        # Call original mouse press handler
        if original_handler:
            original_handler(event)
    
    def set_active_plot(self, plot_name):
        """Set active plot and update visual indicators"""
        # Clear previous focus styling
        if self.active_plot and self.active_plot in self.plot_widgets:
            old_widget = self.plot_widgets[self.active_plot]
            if old_widget is not None:
                old_widget.setStyleSheet("")
        
        # Set new focus
        self.active_plot = plot_name
        if plot_name in self.plot_widgets:
            new_widget = self.plot_widgets[plot_name]
            if new_widget is not None:
                # Apply blue focus border
                new_widget.setStyleSheet("border: 2px solid #0078d4;")
                
        # Update status bar to show which plot has focus
        if hasattr(self, 'statusBar'):
            plot_display_names = {
                'eels_image': 'EELS Image',
                'ndata_image': 'Ndata Image',
                'spectrum': 'Spectrum', 
                'scan_overview': 'Scan Overview',
                'diffraction': 'Diffraction Pattern',
                'virtual_bf': 'Virtual BF',
                'virtual_df': 'Virtual DF'
            }
            display_name = plot_display_names.get(plot_name, plot_name)
            self.statusBar().showMessage(f"Active plot for FFT: {display_name}", 3000)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # Track Shift key for snap-to-center functionality
        if event.key() == Qt.Key_Shift:
            self._shift_pressed = True
        
        # Handle Ctrl+F for FFT
        if event.key() == Qt.Key_F and event.modifiers() == Qt.ControlModifier:
            self.trigger_fft()
        else:
            # Call parent handler for other keys
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle keyboard key releases"""
        # Track Shift key release
        if event.key() == Qt.Key_Shift:
            self._shift_pressed = False
        
        # Call parent handler
        super().keyReleaseEvent(event)
    
    def trigger_fft(self):
        """Trigger FFT analysis on the currently active plot"""
        if not self.active_plot:
            QMessageBox.information(self, "FFT Analysis", 
                                  "Please click on a plot first to select it for FFT analysis.")
            return
        
        # Get data from active plot
        data = self.get_active_plot_data()
        if data is None:
            QMessageBox.warning(self, "FFT Analysis", 
                              "No data available for FFT analysis in the selected plot.")
            return
        
        # Perform FFT and display
        self.perform_fft_analysis(data, self.active_plot)
    
    def get_active_plot_data(self):
        """Get data from the currently active plot"""
        if not self.active_plot:
            return None
            
        try:
            if self.active_plot == 'eels_image' and self.eels_data is not None:
                # Return current EELS integrated image - match display logic
                energy_pixels = self.eels_data.shape[3]  # Energy is in dimension 3
                start_idx = max(0, int(self.current_energy_range[0]))
                end_idx = min(energy_pixels, int(self.current_energy_range[1]))
                return np.sum(self.eels_data[:, :, :, start_idx:end_idx], axis=(2, 3))
                
            elif self.active_plot == 'ndata_image' and self.ndata_data is not None:
                # Return ndata image data
                return self.ndata_data
                
            elif self.active_plot == 'spectrum' and self.eels_data is not None:
                # Spectrum is 1D data - not suitable for 2D FFT
                return None
                
            elif self.active_plot == 'scan_overview' and self.stem4d_data is not None:
                # Return scan overview (integrated diffraction)
                return np.sum(self.stem4d_data, axis=(2, 3))
                
            elif self.active_plot == 'diffraction' and self.stem4d_data is not None:
                # Return current diffraction pattern
                if hasattr(self, 'scan_cursor'):
                    pos = self.scan_cursor.pos()
                    scan_x, scan_y = int(pos[0]), int(pos[1])
                    scan_x = max(0, min(scan_x, self.stem4d_data.shape[1] - 1))
                    scan_y = max(0, min(scan_y, self.stem4d_data.shape[0] - 1))
                    return self.stem4d_data[scan_y, scan_x, :, :]
                else:
                    # Return center diffraction pattern
                    center_y, center_x = self.stem4d_data.shape[0] // 2, self.stem4d_data.shape[1] // 2
                    return self.stem4d_data[center_y, center_x, :, :]
                    
            elif self.active_plot == 'virtual_bf' and hasattr(self, 'bf_virtual_image'):
                # Return current BF virtual image
                return self.bf_virtual_image
                
            elif self.active_plot == 'virtual_df' and hasattr(self, 'df_virtual_image'):
                # Return current DF virtual image
                return self.df_virtual_image
                
        except Exception as e:
            print(f"Error getting data for {self.active_plot}: {e}")
            return None
        
        return None
    
    def get_active_plot_data_with_roi(self, plot_name, window_id):
        """Get data from the specified plot, optionally cropped by FFT ROI"""
        # Get the base data using the existing method
        saved_active_plot = self.active_plot
        self.active_plot = plot_name
        data = self.get_active_plot_data()
        self.active_plot = saved_active_plot
        
        if data is None:
            return None
        
        # Check if there's an FFT ROI for this plot/window
        roi_key = (plot_name, window_id)
        if roi_key not in self.fft_rois:
            # No ROI, return full data
            return data
        
        try:
            # Get ROI geometry
            fft_roi = self.fft_rois[roi_key]
            roi_pos = fft_roi.pos()
            roi_size = fft_roi.size()
            
            # Convert ROI coordinates to array indices
            # ROI gives us: pos = [x, y], size = [width, height]
            x_start = int(max(0, roi_pos[0]))
            y_start = int(max(0, roi_pos[1]))
            x_end = int(min(data.shape[1], roi_pos[0] + roi_size[0]))
            y_end = int(min(data.shape[0], roi_pos[1] + roi_size[1]))
            
            # Ensure we have a valid region
            if x_start >= x_end or y_start >= y_end:
                print(f"Warning: Invalid ROI region for {plot_name}")
                return data
            
            # Extract ROI region
            roi_data = data[y_start:y_end, x_start:x_end]
            
            print(f"FFT ROI extraction: Full shape {data.shape} → ROI shape {roi_data.shape}")
            return roi_data
            
        except Exception as e:
            print(f"Error extracting ROI data for {plot_name}: {e}")
            return data  # Fallback to full data
    
    def perform_fft_analysis(self, data, plot_name):
        """Perform FFT analysis and display results"""
        if data is None:
            return
        
        try:
            # Perform 2D FFT
            fft_data = np.fft.fft2(data)
            fft_shifted = np.fft.fftshift(fft_data)
            
            # Create FFT display window with complex data
            self.show_fft_window(fft_shifted, plot_name)
            
        except Exception as e:
            QMessageBox.critical(self, "FFT Analysis Error", 
                               f"Error performing FFT analysis: {str(e)}")
    
    def perform_fft_analysis_with_roi(self, plot_name, window_id):
        """Perform FFT analysis with ROI support"""
        # Get data (with ROI if applicable)
        data = self.get_active_plot_data_with_roi(plot_name, window_id)
        
        if data is None:
            return None
        
        try:
            # Perform 2D FFT
            fft_data = np.fft.fft2(data)
            fft_shifted = np.fft.fftshift(fft_data)
            return fft_shifted
            
        except Exception as e:
            print(f"Error performing FFT analysis: {e}")
            return None
    
    def show_fft_window(self, fft_complex, source_plot):
        """Display FFT results in a popup window with controls"""
        # Generate unique window ID for FFT ROI tracking
        self.fft_roi_counter += 1
        window_id = self.fft_roi_counter
        
        # Close existing window for this plot if it exists
        if source_plot in self.fft_windows:
            try:
                old_window = self.fft_windows[source_plot]
                # Clean up any existing FFT ROI for this plot
                if 'window_id' in old_window:
                    self.cleanup_fft_roi(source_plot, old_window['window_id'])
                old_window['window'].close()
            except:
                pass
        
        # Create popup window
        fft_window = QMainWindow()
        fft_window.setWindowTitle(f"FFT Analysis - {source_plot}")
        fft_window.resize(1000, 700)
        
        # Create central widget with layout
        central_widget = QWidget()
        layout = QHBoxLayout(central_widget)
        
        # Create plot widget with proper aspect ratio
        fft_plot = pg.PlotWidget()
        fft_plot.setLabel('left', 'Frequency Y')
        fft_plot.setLabel('bottom', 'Frequency X')
        fft_plot.setAspectLocked(True)  # Lock aspect ratio to prevent stretching
        
        # Display FFT data
        fft_image_item = pg.ImageItem()
        # Apply current colormap to FFT window
        current_colormap = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'viridis'
        fft_image_item.setColorMap(pg.colormap.get(current_colormap))
        fft_plot.addItem(fft_image_item)
        
        # Create controls panel
        controls_widget = QWidget()
        controls_widget.setFixedWidth(200)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Display mode controls
        display_group = QGroupBox("Display Mode")
        display_layout = QVBoxLayout(display_group)
        
        display_button_group = QButtonGroup()
        magnitude_radio = QRadioButton("Magnitude (Absolute)")
        magnitude_radio.setChecked(True)  # Default
        real_radio = QRadioButton("Real Part")
        phase_radio = QRadioButton("Phase")
        
        display_button_group.addButton(magnitude_radio, 0)
        display_button_group.addButton(real_radio, 1) 
        display_button_group.addButton(phase_radio, 2)
        
        display_layout.addWidget(magnitude_radio)
        display_layout.addWidget(real_radio)
        display_layout.addWidget(phase_radio)
        
        # Scale controls
        scale_group = QGroupBox("Scale")
        scale_layout = QVBoxLayout(scale_group)
        
        scale_button_group = QButtonGroup()
        log_scale_radio = QRadioButton("Log Scale")
        log_scale_radio.setChecked(True)  # Default
        linear_scale_radio = QRadioButton("Linear Scale")
        
        scale_button_group.addButton(log_scale_radio, 0)
        scale_button_group.addButton(linear_scale_radio, 1)
        
        scale_layout.addWidget(log_scale_radio)
        scale_layout.addWidget(linear_scale_radio)
        
        # ROI controls
        roi_group = QGroupBox("Region Selection")
        roi_layout = QVBoxLayout(roi_group)
        
        roi_checkbox = QCheckBox("Use ROI")
        roi_checkbox.setChecked(False)  # Default to full image
        roi_layout.addWidget(roi_checkbox)
        
        # Add groups to controls
        controls_layout.addWidget(display_group)
        controls_layout.addWidget(scale_group)
        controls_layout.addWidget(roi_group)
        controls_layout.addStretch()  # Push controls to top
        
        # Add to main layout
        layout.addWidget(fft_plot, 1)  # Plot takes most space
        layout.addWidget(controls_widget)
        
        fft_window.setCentralWidget(central_widget)
        
        # Function to update display
        def update_fft_display(new_fft_complex=None):
            nonlocal fft_complex
            if new_fft_complex is not None:
                fft_complex = new_fft_complex
                
            # Get current display mode
            display_mode = display_button_group.checkedId()
            use_log_scale = scale_button_group.checkedId() == 0
            
            if display_mode == 0:  # Magnitude
                data = np.abs(fft_complex)
                title_suffix = "Magnitude"
            elif display_mode == 1:  # Real
                data = np.real(fft_complex)
                title_suffix = "Real Part"
            else:  # Phase
                data = np.angle(fft_complex)
                title_suffix = "Phase"
            
            if use_log_scale and display_mode != 2:  # Don't log scale phase
                if display_mode == 0:  # Magnitude
                    data = np.log10(data + 1e-10)
                    scale_suffix = " (log scale)"
                else:  # Real part
                    # For real part, handle negative values differently
                    data = np.sign(data) * np.log10(np.abs(data) + 1e-10)
                    scale_suffix = " (signed log scale)"
            else:
                scale_suffix = " (linear scale)" if display_mode != 2 else ""
            
            fft_image_item.setImage(data.T)
            fft_plot.setTitle(f'FFT {title_suffix}{scale_suffix} - Source: {source_plot}')
            fft_plot.autoRange()
        
        # Function to handle ROI checkbox
        def on_roi_checkbox_changed(checked):
            if checked:
                # Create FFT ROI on the source plot
                self.create_fft_roi(source_plot, window_id)
                # Update FFT with initial ROI data
                self.on_fft_roi_changed(source_plot, window_id)
            else:
                # Remove FFT ROI
                self.remove_fft_roi(source_plot, window_id)
                # Update FFT back to full data
                update_fft_display()
        
        # Connect controls
        display_button_group.buttonClicked.connect(update_fft_display)
        scale_button_group.buttonClicked.connect(update_fft_display)
        roi_checkbox.toggled.connect(on_roi_checkbox_changed)
        
        # Initial display
        update_fft_display()
        
        # Store window reference and update function
        self.fft_windows[source_plot] = {
            'window': fft_window,
            'window_id': window_id,
            'update_func': update_fft_display
        }
        
        # Handle window close event to clean up ROI
        def on_fft_window_close():
            self.cleanup_fft_roi(source_plot, window_id)
            # Remove from tracking
            if source_plot in self.fft_windows:
                del self.fft_windows[source_plot]
        
        # Connect close event
        fft_window.closeEvent = lambda event: (on_fft_window_close(), event.accept())
        
        # Show window
        fft_window.show()
    
    def create_fft_roi(self, plot_name, window_id):
        """Create an FFT ROI on the specified plot"""
        if plot_name not in self.plot_widgets:
            print(f"Warning: Plot {plot_name} not found")
            return
        
        plot_widget = self.plot_widgets[plot_name]
        roi_key = (plot_name, window_id)
        
        # Don't create if already exists
        if roi_key in self.fft_rois:
            return
        
        # Create ROI with orange color and distinct appearance
        fft_roi = pg.ROI([50, 50], [100, 100], pen='orange', movable=True, resizable=True, removable=False)
        
        # Add corner handles for resizing
        fft_roi.addScaleHandle([1, 1], [0, 0])    # Bottom-right handle
        fft_roi.addScaleHandle([0, 0], [1, 1])    # Top-left handle  
        fft_roi.addScaleHandle([1, 0], [0, 1])    # Top-right handle
        fft_roi.addScaleHandle([0, 1], [1, 0])    # Bottom-left handle
        
        # Connect ROI change signal for real-time FFT updates
        fft_roi.sigRegionChanged.connect(lambda: self.on_fft_roi_changed(plot_name, window_id))
        
        # Add to plot and track
        plot_widget.addItem(fft_roi)
        self.fft_rois[roi_key] = fft_roi
        
        print(f"Created FFT ROI for {plot_name}, window {window_id}")
    
    def remove_fft_roi(self, plot_name, window_id):
        """Remove an FFT ROI from the specified plot"""
        roi_key = (plot_name, window_id)
        
        if roi_key not in self.fft_rois:
            return
        
        plot_widget = self.plot_widgets[plot_name]
        fft_roi = self.fft_rois[roi_key]
        
        # Remove from plot
        plot_widget.removeItem(fft_roi)
        
        # Remove from tracking
        del self.fft_rois[roi_key]
        
        print(f"Removed FFT ROI for {plot_name}, window {window_id}")
    
    def cleanup_fft_roi(self, plot_name, window_id):
        """Clean up FFT ROI when window is closed"""
        self.remove_fft_roi(plot_name, window_id)
    
    def on_fft_roi_changed(self, plot_name, window_id):
        """Handle FFT ROI changes - update FFT in real-time"""
        if plot_name not in self.fft_windows:
            return
        
        try:
            # Get new FFT data with updated ROI
            fft_shifted = self.perform_fft_analysis_with_roi(plot_name, window_id)
            
            if fft_shifted is not None:
                # Update the FFT display
                self.fft_windows[plot_name]['update_func'](fft_shifted)
                
        except Exception as e:
            print(f"Error updating FFT after ROI change for {plot_name}: {e}")
    
    def update_fft_windows(self, plot_names=None):
        """Update FFT windows with current data"""
        if plot_names is None:
            # Update all open FFT windows
            plot_names = list(self.fft_windows.keys())
        elif isinstance(plot_names, str):
            # Convert single plot name to list
            plot_names = [plot_names]
        
        for plot_name in plot_names:
            if plot_name in self.fft_windows:
                try:
                    # Get window ID for ROI-aware data extraction
                    window_id = self.fft_windows[plot_name]['window_id']
                    
                    # Get current data for this plot (with ROI if applicable)
                    data = self.get_active_plot_data_with_roi(plot_name, window_id)
                    
                    if data is not None:
                        # Perform FFT
                        fft_data = np.fft.fft2(data)
                        fft_shifted = np.fft.fftshift(fft_data)
                        
                        # Update the window
                        self.fft_windows[plot_name]['update_func'](fft_shifted)
                        
                except Exception as e:
                    print(f"Error updating FFT window for {plot_name}: {e}")
    
    # Conversion tab methods
    def browse_input_file(self):
        """Browse for input MIB file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select MIB file to convert",
            "",
            "MIB files (*.mib);;All files (*.*)"
        )
        
        if filename:
            self.input_file_path.setText(filename)
            self.analyze_input_file(filename)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder"
        )
        
        if folder:
            self.output_folder_path.setText(folder)
            self.update_output_filename()
    
    def analyze_input_file(self, filename):
        """Analyze the selected input file and update UI"""
        try:
            # Get file information
            file_info = get_data_file_info(filename)
            
            if file_info['compatible']:
                # Display file info
                shape = file_info['shape']
                size_gb = file_info['size_gb']
                info_text = (f"Size: {size_gb:.2f} GB | "
                           f"Scan: {shape[0]}×{shape[1]} | "
                           f"Detector: {shape[2]}×{shape[3]} | "
                           f"Frames: {shape[0] * shape[1]:,}")
                
                self.file_info_label.setText(info_text)
                self.file_info_label.setStyleSheet("color: #000;")
                
                # Update experiment type and processing options
                self.update_experiment_type_display(file_info)
                
                # Enable conversion button
                self.convert_btn.setEnabled(True)
                
                # Update output filename
                self.update_output_filename()
                
                # Update preview
                self.update_conversion_preview()
                
            else:
                self.file_info_label.setText(f"Error: {file_info.get('error', 'Invalid file')}")
                self.file_info_label.setStyleSheet("color: #cc0000;")
                self.convert_btn.setEnabled(False)
                
        except Exception as e:
            self.file_info_label.setText(f"Error analyzing file: {str(e)}")
            self.file_info_label.setStyleSheet("color: #cc0000;")
            self.convert_btn.setEnabled(False)
    
    def update_output_filename(self):
        """Update the output filename based on input file"""
        input_path = self.input_file_path.text()
        if input_path:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_name = f"{base_name}.emd"
            self.output_filename.setText(output_name)
    
    def update_compression_level(self):
        """Update compression level availability based on algorithm"""
        algorithm = self.compression_combo.currentText()
        
        if algorithm == 'gzip':
            self.compression_level_combo.setEnabled(True)
        else:
            self.compression_level_combo.setEnabled(False)
        
        self.update_conversion_preview()
    
    def update_conversion_preview(self):
        """Update the conversion preview information"""
        input_path = self.input_file_path.text()
        if not input_path:
            return
        
        try:
            file_info = get_data_file_info(input_path)
            if not file_info['compatible']:
                return
            
            input_size_gb = file_info['size_gb']
            compression_algo = self.compression_combo.currentText()
            
            # Estimate compression ratios based on our benchmark data
            compression_ratios = {
                'gzip': {'1': 20.1, '6': 26.4, '9': 28.9},
                'szip': {'default': 23.5},
                'lzf': {'default': 14.0},
                'none': {'default': 0.8}
            }
            
            if compression_algo == 'gzip':
                level = self.compression_level_combo.currentText()
                ratio = compression_ratios['gzip'].get(level, 26.4)
            else:
                ratio = compression_ratios.get(compression_algo, {}).get('default', 1.0)
            
            # Calculate data processing impact
            processing_options = self.get_processing_options()
            original_shape = file_info['shape']
            processed_shape, processing_reduction = calculate_processed_size(original_shape, processing_options)
            
            # Calculate total size after processing and compression
            processed_size_gb = input_size_gb / processing_reduction
            final_size_gb = processed_size_gb / ratio if ratio > 1 else processed_size_gb * 1.25
            
            # Calculate total reduction factor
            total_reduction = input_size_gb / final_size_gb
            
            # Estimate time (processing adds some overhead)
            processing_time_factor = 1.0
            if processing_options['bin_factor'] > 1 or processing_options['sum_y']:
                processing_time_factor = 1.2  # 20% overhead for processing
            
            estimated_time = max(5, int(input_size_gb * 10 * processing_time_factor))
            
            # Build preview text
            preview_lines = [f"Input size: {input_size_gb:.2f} GB ({original_shape})"]
            
            if processing_reduction > 1:
                preview_lines.append(f"After processing: {processed_size_gb:.2f} GB ({processed_shape}) • {processing_reduction:.1f}x reduction")
            
            preview_lines.extend([
                f"Final output size: {final_size_gb:.2f} GB ({total_reduction:.1f}x total reduction)",
                f"Estimated time: ~{estimated_time} seconds"
            ])
            
            # Add processing description
            if processing_options['sum_y'] and processing_options['bin_factor'] > 1:
                preview_lines.append(f"Processing: {processing_options['bin_factor']}x{processing_options['bin_factor']} binning → Y-sum")
            elif processing_options['sum_y']:
                preview_lines.append("Processing: Sum in Y direction")
            elif processing_options['bin_factor'] > 1:
                method = processing_options['bin_method'].capitalize()
                preview_lines.append(f"Processing: {processing_options['bin_factor']}x{processing_options['bin_factor']} binning ({method})")
            
            preview_text = "\n".join(preview_lines)
            self.preview_label.setText(preview_text)
            self.preview_label.setStyleSheet("color: #000; padding: 10px;")
            
        except Exception as e:
            self.preview_label.setText(f"Preview unavailable: {str(e)}")
            self.preview_label.setStyleSheet("color: #cc0000; padding: 10px;")
    
    def start_conversion(self):
        """Start the MIB to EMD conversion process using threaded worker"""
        input_path = self.input_file_path.text()
        if not input_path or not os.path.exists(input_path):
            QMessageBox.warning(self, "Error", "Please select a valid input file.")
            return
        
        # Determine output path
        output_folder = self.output_folder_path.text()
        if not output_folder:
            output_folder = os.path.dirname(input_path)
        
        output_filename = self.output_filename.text()
        if not output_filename:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_filename = f"{base_name}.emd"
        
        output_path = os.path.join(output_folder, output_filename)
        
        # Check if output file exists
        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self, 
                "File Exists", 
                f"The file {output_filename} already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # Setup conversion parameters
        compression = self.compression_combo.currentText()
        if compression == 'none':
            compression = None
        
        compression_level = None
        if compression == 'gzip':
            compression_level = int(self.compression_level_combo.currentText())
        
        # Setup UI for conversion
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.conversion_progress.setVisible(True)
        self.conversion_progress.setRange(0, 100)
        self.conversion_progress.setValue(0)
        self.conversion_status.setText("Initializing conversion...")
        
        # Get processing options from UI
        processing_options = self.get_processing_options()
        
        # Create worker and thread
        self.conversion_worker = ConversionWorker(
            input_path, output_path, compression, compression_level, 
            processing_options=processing_options
        )
        self.conversion_thread = QThread()
        
        # Move worker to thread
        self.conversion_worker.moveToThread(self.conversion_thread)
        
        # Connect signals
        self.conversion_worker.progress_updated.connect(self.on_conversion_progress)
        self.conversion_worker.conversion_finished.connect(self.on_conversion_finished)
        self.conversion_worker.conversion_failed.connect(self.on_conversion_failed)
        self.conversion_worker.log_message_signal.connect(self.log_message)  # Qt-safe logging
        
        # Log conversion start
        self.log_message(f"Starting conversion: {os.path.basename(input_path)} → {output_filename}")
        
        # Connect thread signals
        self.conversion_thread.started.connect(self.conversion_worker.run_conversion)
        self.conversion_thread.finished.connect(self.conversion_thread.deleteLater)
        
        # Start the thread
        self.conversion_thread.start()
    
    def on_conversion_progress(self, progress, status):
        """Handle progress updates from conversion worker"""
        self.conversion_progress.setValue(progress)
        self.conversion_status.setText(status)
    
    def on_conversion_finished(self, stats):
        """Handle successful completion of conversion"""
        output_path = self.conversion_worker.output_path
        
        # Update UI with success
        final_status = (
            f"Conversion completed! {stats['input_size_gb']:.2f} GB → "
            f"{stats['output_size_gb']:.2f} GB ({stats['compression_ratio']:.1f}x) "
            f"in {stats.get('actual_time', stats['total_time_s']):.1f}s"
        )
        self.conversion_status.setText(final_status)
        
        # Log success
        self.log_message(
            f"Conversion completed: {os.path.basename(output_path)} - "
            f"{stats['compression_ratio']:.1f}x compression in "
            f"{stats.get('actual_time', stats['total_time_s']):.1f}s", 
            "SUCCESS"
        )
        
        # Enable "Open Converted File" button
        self.last_converted_file = output_path
        self.open_converted_btn.setEnabled(True)
        
        # Clean up thread
        self.cleanup_conversion_thread()
    
    def on_conversion_failed(self, error_message):
        """Handle conversion failure"""
        self.conversion_progress.setValue(0)
        self.conversion_status.setText(f"Conversion failed: {error_message}")
        
        # Log error
        self.log_message(f"Conversion failed: {error_message}", "ERROR")
        
        QMessageBox.critical(self, "Conversion Error", f"Failed to convert file:\n{error_message}")
        
        # Clean up thread
        self.cleanup_conversion_thread()
    
    def cleanup_conversion_thread(self):
        """Clean up conversion thread and reset UI state"""
        # Reset UI state
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        # Clean up thread if it exists
        if hasattr(self, 'conversion_thread') and self.conversion_thread:
            if self.conversion_thread.isRunning():
                self.conversion_thread.quit()
                self.conversion_thread.wait()
            self.conversion_thread = None
            self.conversion_worker = None
    
    def cancel_conversion(self):
        """Cancel the ongoing conversion"""
        if hasattr(self, 'conversion_worker') and self.conversion_worker:
            # Signal the worker to cancel
            self.conversion_worker.cancel()
            
            # Log cancellation
            self.log_message("User cancelled conversion", "WARNING")
            
            # Update UI immediately
            self.conversion_status.setText("Cancelling conversion...")
            self.cancel_btn.setEnabled(False)
            
            # Wait a moment for worker to respond, then force cleanup
            QTimer.singleShot(2000, self.force_cancel_cleanup)  # 2 second timeout
    
    def force_cancel_cleanup(self):
        """Force cleanup after cancel timeout"""
        if hasattr(self, 'conversion_thread') and self.conversion_thread:
            if self.conversion_thread.isRunning():
                self.conversion_thread.terminate()  # Force terminate if still running
                self.conversion_thread.wait(1000)   # Wait up to 1 second
        
        # Update UI
        self.conversion_status.setText("Conversion cancelled")
        self.conversion_progress.setValue(0)
        self.cleanup_conversion_thread()
    
    def open_converted_file(self):
        """Open the converted EMD file in the viewer"""
        if self.last_converted_file and os.path.exists(self.last_converted_file):
            try:
                # Load the converted file using the existing loader
                raw_data = load_data_file(self.last_converted_file)
                
                # Detect experiment type based on data shape
                experiment_type, exp_info = detect_experiment_type(raw_data.shape)
                self.log_message(f"Opening converted file - Detected: {experiment_type} - {exp_info['detector_type']}")
                
                if experiment_type == "EELS":
                    # Flip the energy axis for EELS data and store
                    # After transpose, energy axis is now the last dimension (index 3) for EELS
                    self.eels_data = raw_data[:, :, :, ::-1]
                    
                    self.eels_filename = os.path.basename(self.last_converted_file)
                    self.eels_label.setText(f"EELS File: {self.eels_filename} (EMD)")
                    
                    # Auto-switch to EELS tab
                    self.tab_widget.setCurrentIndex(0)
                    
                elif experiment_type == "4D_STEM":
                    # Store as 4D STEM data (no energy axis flip needed)
                    self.stem4d_data = raw_data
                    
                    # Log 4D STEM file loading
                    self.log_message(f"Opened converted 4D STEM data: {os.path.basename(self.last_converted_file)} (EMD)")
                    
                    # Auto-switch to 4D STEM tab
                    self.tab_widget.setCurrentIndex(1)
                    
                else:
                    # Unknown data type - default to EELS behavior
                    self.eels_data = raw_data[:, :, :, ::-1]
                    self.eels_filename = os.path.basename(self.last_converted_file)
                    self.eels_label.setText(f"Unknown Data: {self.eels_filename} (EMD)")
                    self.tab_widget.setCurrentIndex(0)
                
                # Initialize ROI to center of image
                if experiment_type == "EELS" and self.eels_data is not None:
                    h, w = self.eels_data.shape[:2]
                elif experiment_type == "4D_STEM" and self.stem4d_data is not None:
                    h, w = self.stem4d_data.shape[:2]
                else:
                    # Fallback for unknown data type
                    h, w = self.eels_data.shape[:2] if self.eels_data is not None else (100, 100)
                if self.roi_mode:
                    roi_size = min(w, h) // 3
                    roi_x = (w - roi_size) // 2
                    roi_y = (h - roi_size) // 2
                    self.current_roi = (roi_x, roi_y, roi_size, roi_size, 0.0)
                else:
                    self.current_roi = (w // 2, h // 2)
                
                # Check compatibility with ndata
                if self.ndata_data is not None:
                    self.check_compatibility()
                
                self.update_displays()
                self.setup_roi_positions()
                self.update_4d_displays()
                
                # Switch to EELS tab to show the loaded data
                self.tab_widget.setCurrentIndex(0)
                
                self.status_bar.showMessage(f"Loaded converted EMD file: {self.eels_filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load converted file:\n{str(e)}")
    
    # Data processing UI methods
    def update_experiment_type_display(self, file_info):
        """Update the experiment type display and processing options"""
        exp_type = file_info.get('experiment_type', 'UNKNOWN')
        processing_info = file_info.get('processing_options', {})
        
        # Update experiment type label
        detector_type = processing_info.get('detector_type', 'Unknown detector')
        self.experiment_type_label.setText(f"Auto-detected: {exp_type} ({detector_type})")
        self.experiment_type_label.setStyleSheet("color: #000; font-weight: bold;")
        
        # Show processing options
        self.processing_options_widget.setVisible(True)
        
        # Configure UI based on experiment type
        if exp_type == "EELS":
            self.sum_y_radio.setEnabled(processing_info.get('can_sum_y', False))
            self.binning_radio.setEnabled(False)
            self.bin_factor_combo.setEnabled(False)
            self.bin_method_combo.setEnabled(False)
            self.advanced_radio.setEnabled(processing_info.get('can_sum_y', False))
            
            # Select recommended processing
            if processing_info.get('recommended_processing') == 'sum_y':
                self.sum_y_radio.setChecked(True)
            else:
                self.no_processing_radio.setChecked(True)
                
        elif exp_type == "4D_STEM":
            self.sum_y_radio.setEnabled(False)
            self.binning_radio.setEnabled(True)
            self.bin_factor_combo.setEnabled(True)
            self.bin_method_combo.setEnabled(True)
            self.advanced_radio.setEnabled(False)
            
            # Update bin factor options
            valid_factors = processing_info.get('valid_bin_factors', [1, 2, 4, 8])
            self.update_bin_factor_options(valid_factors)
            
            # Select recommended processing
            if processing_info.get('recommended_processing') == 'bin_2x2':
                self.binning_radio.setChecked(True)
            else:
                self.no_processing_radio.setChecked(True)
                
        else:  # UNKNOWN
            self.sum_y_radio.setEnabled(False)
            self.binning_radio.setEnabled(False)
            self.bin_factor_combo.setEnabled(False)
            self.bin_method_combo.setEnabled(False)
            self.advanced_radio.setEnabled(False)
            self.no_processing_radio.setChecked(True)
    
    def update_bin_factor_options(self, valid_factors):
        """Update the bin factor combo box with valid options"""
        self.bin_factor_combo.clear()
        for factor in valid_factors:
            if factor > 1:  # Skip 1x1 (no binning)
                self.bin_factor_combo.addItem(f"{factor}x{factor}")
    
    def update_processing_ui(self):
        """Update UI elements based on selected processing option"""
        # Enable/disable combo boxes based on selected radio button
        binning_selected = self.binning_radio.isChecked() or self.advanced_radio.isChecked()
        self.bin_factor_combo.setEnabled(binning_selected)
        self.bin_method_combo.setEnabled(binning_selected)
        
        # Update preview
        self.update_conversion_preview()
    
    def get_processing_options(self):
        """Get the current processing options from UI"""
        processing_options = {
            'sum_y': False,
            'bin_factor': 1,
            'bin_method': 'mean'
        }
        
        if self.sum_y_radio.isChecked():
            processing_options['sum_y'] = True
        elif self.binning_radio.isChecked():
            # Parse bin factor from "4x4" format
            bin_text = self.bin_factor_combo.currentText()
            if 'x' in bin_text:
                processing_options['bin_factor'] = int(bin_text.split('x')[0])
            processing_options['bin_method'] = self.bin_method_combo.currentText().lower()
        elif self.advanced_radio.isChecked():
            # Both operations
            processing_options['sum_y'] = True
            bin_text = self.bin_factor_combo.currentText()
            if 'x' in bin_text:
                processing_options['bin_factor'] = int(bin_text.split('x')[0])
            processing_options['bin_method'] = self.bin_method_combo.currentText().lower()
        
        return processing_options

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