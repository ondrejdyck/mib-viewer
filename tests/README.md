# MIB Viewer Test Suite

This directory contains the comprehensive testing framework for the MIB Viewer project, organized by test type and purpose.

## Directory Structure

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interactions  
├── performance/          # Performance benchmarking and validation tests
├── debug/                # Debug utilities and diagnostic scripts
└── patches/              # GUI integration patches and modification scripts
```

## Test Categories

### Unit Tests (`unit/`)
Tests for individual components in isolation:
- `test_smart_data_reader.py` - Tests SmartDataReader functionality and chunking strategies

### Integration Tests (`integration/`)
Tests for component interactions and data flow:
- `test_chunked_processing.py` - Tests chunked processing pipeline end-to-end

### Performance Tests (`performance/`)
Performance benchmarking and validation:
- `test_enhanced_converter.py` - Compares enhanced vs original converter performance
- `test_8gb_file.py` - Large file processing performance validation

### Debug Utilities (`debug/`)
Diagnostic and debugging scripts:
- `debug_conversion_performance.py` - Investigates multithreading performance issues
- `debug_data_corruption.py` - Compares chunked vs in-memory processing outputs

### Patches (`patches/`)
GUI integration and modification scripts:
- `apply_gui_enhancement.py` - Applies enhanced conversion pipeline to main GUI
- `gui_integration_patch.py` - Demonstrates enhanced pipeline GUI integration

## Running Tests

### Individual Test Files
```bash
# Unit tests
python tests/unit/test_smart_data_reader.py

# Integration tests
python tests/integration/test_chunked_processing.py

# Performance tests
python tests/performance/test_enhanced_converter.py <file_path>
python tests/performance/test_8gb_file.py <large_file_path>

# Debug utilities
python tests/debug/debug_conversion_performance.py <file_path>
python tests/debug/debug_data_corruption.py <file1> <file2>
```

### Performance Comparison
To compare old vs new approaches:
```bash
# Run performance debugging
python tests/debug/debug_conversion_performance.py /path/to/test/file.mib

# Test enhanced converter
python tests/performance/test_enhanced_converter.py /path/to/test/file.mib
```

## Migration Notes

**Moved from `experiments/` directory:**
- All files starting with `test_*` → appropriate test category
- All files starting with `debug_*` → `debug/`
- GUI patch scripts → `patches/`

**Remaining in `experiments/`:**
- Research and experimental code
- Super resolution features
- New feature prototypes

This organization provides clear separation between testing/validation (this directory) and research/experimentation (experiments directory).