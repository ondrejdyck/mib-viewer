#!/usr/bin/env python3
"""
Enhanced ConversionWorker with multithreaded pipeline integration

This module provides a drop-in replacement for the existing ConversionWorker
that uses the new SmartDataReader + ProcessingPipeline for much better performance
and real progress tracking.

Key Enhancements:
- Real progress from pipeline stages (not simulated)
- Automatic strategy selection (pipeline vs fallback)
- Enhanced error handling and cancellation support
- Detailed progress breakdown (Loading, Processing, Writing)
- Performance metrics integration
"""

import time
from PyQt5.QtCore import QObject, pyqtSignal

# Import our fixed original adaptive converter (restored parallelization)
from ..io.adaptive_converter import AdaptiveMibEmdConverter
from ..io.mib_loader import get_data_file_info


class EnhancedConversionWorker(QObject):
    """
    Enhanced worker class for multithreaded MIB/EMD conversion
    
    This worker provides the same Qt signals as the original ConversionWorker
    but uses the new multithreaded pipeline for much better performance and
    real progress tracking.
    
    Signals:
    --------
    progress_updated(int, str) : Progress percentage and status message
    conversion_finished(dict) : Conversion statistics on success
    conversion_failed(str) : Error message on failure
    log_message_signal(str, str) : Log messages with level
    stage_progress_updated(str, int, str) : Stage name, percentage, message
    """
    
    # Qt signals - same as original ConversionWorker for compatibility
    progress_updated = pyqtSignal(int, str)     # progress percentage, status message
    conversion_finished = pyqtSignal(dict)      # conversion statistics
    conversion_failed = pyqtSignal(str)         # error message
    log_message_signal = pyqtSignal(str, str)   # message, level
    
    # Enhanced signals for more detailed progress
    stage_progress_updated = pyqtSignal(str, int, str)  # stage, percentage, message
    performance_updated = pyqtSignal(dict)      # real-time performance stats
    
    def __init__(self, input_path, output_path, compression, compression_level, 
                 processing_options=None, log_callback=None, max_workers=None):
        """
        Initialize Enhanced ConversionWorker
        
        Parameters:
        -----------
        input_path : str
            Path to input MIB or EMD file
        output_path : str
            Path for output EMD file  
        compression : str
            HDF5 compression algorithm
        compression_level : int
            Compression level
        processing_options : dict, optional
            Data processing options (binning, summing, etc.)
        log_callback : callable, optional
            Log callback function (for compatibility)
        max_workers : int, optional
            Maximum worker threads for pipeline
        """
        super().__init__()
        
        self.input_path = input_path
        self.output_path = output_path
        self.compression = compression
        self.compression_level = compression_level
        self.processing_options = processing_options or {}
        self.log_callback = log_callback
        self.max_workers = max_workers
        self.cancelled = False
        
        # Performance tracking
        self._stage_progress = {
            'analysis': 0,
            'loading': 0, 
            'processing': 0,
            'writing': 0
        }
        self._current_stage = 'analysis'
        self._conversion_start_time = None
        
        # Create fixed original adaptive converter (restored parallelization)
        self.converter = AdaptiveMibEmdConverter(
            compression=compression,
            compression_level=compression_level,
            max_workers=max_workers,
            progress_callback=self._pipeline_progress_callback,
            log_callback=self.qt_safe_log,
            verbose=True  # Enable detailed logging for GUI
        )
    
    def qt_safe_log(self, message, level="INFO"):
        """Qt-safe logging that emits signal instead of direct callback"""
        # Always emit to GUI log - this is the primary purpose
        self.log_message_signal.emit(message, level)
        
        # Also call original callback if provided (for backward compatibility)
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                pass  # Don't let callback errors break the conversion
                
    
    def cancel(self):
        """Cancel the conversion process"""
        self.cancelled = True
        self.qt_safe_log("Conversion cancelled by user", "WARNING")
    
    def run_conversion(self):
        """Run the enhanced conversion process with real progress updates"""
        try:
            self._conversion_start_time = time.time()
            
            # Phase 1: File Analysis (0-10%)
            self._current_stage = 'analysis'
            self._update_overall_progress(5, "Analyzing input file...")
            
            if self.cancelled:
                return
            
            # Get file information for progress estimation
            file_info = get_data_file_info(self.input_path)
            file_size_gb = file_info['size_gb']
            
            self.qt_safe_log(f"File analysis: {file_size_gb:.2f} GB", "INFO")
            self._update_stage_progress('analysis', 100, f"File analyzed: {file_size_gb:.2f} GB")
            self._update_overall_progress(10, f"File analyzed: {file_size_gb:.2f} GB")
            
            if self.cancelled:
                return
            
            # Phase 2-4: Conversion with real pipeline progress (10-100%)
            self._current_stage = 'conversion'
            self._update_overall_progress(15, "Starting enhanced conversion...")
            
            # Run conversion with our adaptive converter
            stats = self.converter.convert_to_emd(
                self.input_path,
                self.output_path,
                processing_options=self.processing_options
            )
            
            if self.cancelled:
                return
            
            # Phase 5: Completion (100%)
            self._update_overall_progress(100, "Conversion completed successfully!")
            
            # Add timing and performance information
            end_time = time.time()
            total_time = end_time - self._conversion_start_time
            stats['gui_total_time'] = total_time
            stats['file_size_gb'] = file_size_gb
            
            # Calculate GUI-specific performance metrics
            if 'pipeline_throughput_mb_s' in stats:
                stats['gui_throughput_description'] = f"Pipeline: {stats['pipeline_throughput_mb_s']:.1f} MB/s"
            else:
                stats['gui_throughput_description'] = f"Standard: {(file_size_gb*1024)/total_time:.1f} MB/s"
            
            # Emit performance update
            self.performance_updated.emit({
                'total_time': total_time,
                'throughput': stats.get('pipeline_throughput_mb_s', 0),
                'compression_ratio': stats.get('compression_ratio', 1.0),
                'efficiency': stats.get('parallelization_efficiency', 0)
            })
            
            self.qt_safe_log(f"Conversion completed in {total_time:.1f}s", "SUCCESS")
            
            # Emit success signal
            self.conversion_finished.emit(stats)
            
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            self.qt_safe_log(error_msg, "ERROR")
            self.conversion_failed.emit(error_msg)
    
    def _pipeline_progress_callback(self, percentage, message):
        """Handle progress updates from the pipeline"""
        if self.cancelled:
            return
            
        # Parse pipeline messages to determine stage
        message_lower = message.lower()
        
        if 'loading' in message_lower or 'analyzing' in message_lower:
            stage = 'loading'
            stage_progress = percentage
        elif 'processing' in message_lower or 'binning' in message_lower:
            stage = 'processing' 
            stage_progress = percentage
        elif 'writing' in message_lower:
            stage = 'writing'
            stage_progress = percentage
        else:
            stage = 'conversion'
            stage_progress = percentage
        
        # Update stage progress
        self._stage_progress[stage] = stage_progress
        self._update_stage_progress(stage, stage_progress, message)
        
        # Calculate overall progress (10% for analysis + 90% for conversion)
        conversion_weight = 0.9
        overall_progress = 10 + int(percentage * conversion_weight)
        overall_progress = max(15, min(99, overall_progress))  # Keep in 15-99 range
        
        # Create detailed status message
        detailed_message = f"Pipeline {stage.capitalize()}: {message}"
        
        self._update_overall_progress(overall_progress, detailed_message)
    
    def _update_overall_progress(self, percentage, message):
        """Update overall progress and emit signal"""
        if not self.cancelled:
            self.progress_updated.emit(percentage, message)
    
    def _update_stage_progress(self, stage, percentage, message):
        """Update stage-specific progress and emit signal"""
        if not self.cancelled:
            self.stage_progress_updated.emit(stage, percentage, message)
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if self._conversion_start_time:
            elapsed_time = time.time() - self._conversion_start_time
            return {
                'elapsed_time': elapsed_time,
                'current_stage': self._current_stage,
                'stage_progress': self._stage_progress.copy()
            }
        return {}


class ConversionWorkerFactory:
    """Factory for creating conversion workers with backward compatibility"""
    
    @staticmethod
    def create_worker(input_path, output_path, compression, compression_level,
                     processing_options=None, log_callback=None, use_enhanced=True):
        """
        Create a conversion worker
        
        Parameters:
        -----------
        use_enhanced : bool
            If True, use EnhancedConversionWorker (default)
            If False, use original ConversionWorker for compatibility
        """
        
        if use_enhanced:
            return EnhancedConversionWorker(
                input_path=input_path,
                output_path=output_path,
                compression=compression,
                compression_level=compression_level,
                processing_options=processing_options,
                log_callback=log_callback
            )
        else:
            # Import and return original worker for fallback
            from .mib_viewer_pyqtgraph import ConversionWorker
            return ConversionWorker(
                input_path=input_path,
                output_path=output_path,
                compression=compression,
                compression_level=compression_level,
                processing_options=processing_options,
                log_callback=log_callback
            )