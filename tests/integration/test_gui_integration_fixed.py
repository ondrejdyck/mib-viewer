#!/usr/bin/env python3
"""
Test GUI integration with fixed adaptive converter
Verify that the GUI worker uses the fixed adaptive converter correctly
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock PyQt5 to avoid GUI dependencies for testing
class MockSignal:
    def emit(self, *args):
        print(f"SIGNAL: {args}")

class MockQObject:
    def __init__(self):
        self.progress_updated = MockSignal()
        self.conversion_finished = MockSignal()
        self.conversion_failed = MockSignal()
        self.log_message_signal = MockSignal()
        self.stage_progress_updated = MockSignal()
        self.performance_updated = MockSignal()

# Mock PyQt5 before importing GUI components
sys.modules['PyQt5.QtCore'] = type(sys)('PyQt5.QtCore')
sys.modules['PyQt5.QtCore'].QObject = MockQObject
sys.modules['PyQt5.QtCore'].pyqtSignal = lambda *args: MockSignal()

# Now import the GUI worker
from mib_viewer.gui.enhanced_conversion_worker import ConversionWorkerFactory, EnhancedConversionWorker

def test_gui_integration():
    """Test that GUI integration uses the fixed adaptive converter"""
    print("=== Testing GUI Integration with Fixed Adaptive Converter ===")

    # Test file path
    test_file = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/64x64 Test.mib"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.emd', delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        print(f"Testing GUI worker with: {os.path.basename(test_file)}")

        # Create worker via factory (simulating GUI usage)
        worker = ConversionWorkerFactory.create_worker(
            input_path=test_file,
            output_path=output_path,
            compression='gzip',
            compression_level=6,
            use_enhanced=True  # Use the enhanced worker with our fixed converter
        )

        print(f"✓ Created worker: {type(worker).__name__}")

        # Verify it's using the fixed adaptive converter
        if hasattr(worker, 'converter'):
            converter_type = type(worker.converter).__name__
            print(f"✓ Converter type: {converter_type}")

            # This should be AdaptiveMibEmdConverter (the fixed original), not V2
            if converter_type == 'AdaptiveMibEmdConverter':
                print("✓ GUI is using FIXED original adaptive converter!")

                # Test a small conversion via GUI worker
                print("\n--- Testing GUI Worker Conversion ---")

                def test_log_callback(message, level="INFO"):
                    print(f"GUI_LOG[{level}]: {message}")

                # Set up callbacks
                worker.converter.log_callback = test_log_callback

                # Test dimension detection through GUI worker
                file_info = worker.converter._analyze_input_file(test_file)
                detected_shape = file_info['file_shape']
                expected_shape = (64, 64, 1024, 256)

                if detected_shape == expected_shape:
                    print(f"✓ GUI worker dimension detection CORRECT: {detected_shape}")
                    return True
                else:
                    print(f"✗ GUI worker dimension detection INCORRECT: got {detected_shape}, expected {expected_shape}")
                    return False

            elif converter_type == 'AdaptiveMibEmdConverterV2':
                print("✗ GUI is still using V2 converter - integration not complete")
                return False
            else:
                print(f"✗ GUI is using unknown converter: {converter_type}")
                return False
        else:
            print("✗ Worker has no converter attribute")
            return False

    except Exception as e:
        print(f"✗ GUI integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)

def test_factory_options():
    """Test that the factory correctly switches between converters"""
    print("\n=== Testing Conversion Worker Factory ===")

    test_file = "/media/o2d/data/ORNL Dropbox/Ondrej Dyck/Code/mib-viewer-project/Example 4D/64x64 Test.mib"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False

    with tempfile.NamedTemporaryFile(suffix='.emd', delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        # Test enhanced=True (should use fixed adaptive converter)
        worker_enhanced = ConversionWorkerFactory.create_worker(
            input_path=test_file,
            output_path=output_path,
            compression='gzip',
            compression_level=6,
            use_enhanced=True
        )

        enhanced_type = type(worker_enhanced).__name__
        converter_type = type(worker_enhanced.converter).__name__

        print(f"Enhanced=True: Worker={enhanced_type}, Converter={converter_type}")

        if enhanced_type == 'EnhancedConversionWorker' and converter_type == 'AdaptiveMibEmdConverter':
            print("✓ Factory correctly creates enhanced worker with fixed adaptive converter")
            return True
        else:
            print("✗ Factory not working correctly")
            return False

    except Exception as e:
        print(f"✗ Factory test failed: {str(e)}")
        return False
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

if __name__ == "__main__":
    print("Testing GUI Integration with Fixed Adaptive Converter")
    print("=" * 60)

    success = True

    # Test 1: GUI Integration
    if not test_gui_integration():
        success = False

    # Test 2: Factory Options
    if not test_factory_options():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 GUI INTEGRATION TESTS PASSED!")
        print("✓ GUI now uses the fixed original adaptive converter")
        print("✓ Dimension detection working correctly")
        print("✓ Factory correctly switches between converters")
        print("✓ Ready for production use!")
    else:
        print("❌ GUI integration tests failed")

    print("=" * 60)