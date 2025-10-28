# Changelog

All notable changes to MIB Viewer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-10-28

### Added
- **Super-Resolution Feature**: Complete 4D STEM super-resolution reconstruction workflow
  - Interactive 5-step GUI workflow (Select BF → Compute FFT → Correlations → Shift Maps → Reconstruct)
  - Memory-efficient HDF5 caching system for large datasets
  - FFT Explorer for interactive Fourier-space visualization
  - Real-time progress tracking and status updates
  
- **Intelligent Cache Detection**: Prevents duplicate cache file creation
  - Automatic detection of existing cache files when starting Step 2
  - Interactive dialog with Load/Recompute/Cancel options
  - Cache metadata display (size, date, completion status)
  - Parameter validation with mismatch warnings
  
- **Documentation**: Comprehensive super-resolution documentation
  - Release notes with usage instructions
  - Technical implementation details
  - Troubleshooting guide
  - Known limitations and future enhancements

### Changed
- Reorganized project structure:
  - Moved development documentation to `docs/superres/`
  - Archived test scripts and outputs to `archive/` (gitignored)
  - Updated `.gitignore` for cleaner repository

### Fixed
- Cache file path handling for files loaded from any directory
- FFT explorer initialization when resuming from cache
- Duplicate cache file creation issue (50+ GB files)

### Technical Details
- Cache files use HDF5 format with incremental computation support
- Typical cache size: 50-100 GB for 256×256 scans
- Memory-efficient chunked processing
- Compatible with EMD format 4D STEM data

### Known Limitations
- ADF/DF super-resolution implemented but not GUI-integrated (experimental)
- Requires 8+ GB RAM for small datasets, 16+ GB recommended
- Cache files are large - ensure sufficient disk space

## [1.0.12] - 2025-10-XX

### Fixed
- Multiple critical bugs
- Spectrum plot disappearing when zoomed due to Qt painter path overflow

### Changed
- Updated multi-platform release workflow for versioned binaries

## [1.0.11] - 2025-XX-XX

### Changed
- Versioned binary builds
- Linux build workflow defaults to Ubuntu 22.04

## [1.0.0] - 2025-XX-XX

### Added
- Initial release of MIB Viewer
- Support for MIB and EMD file formats
- EELS data visualization
- Virtual detector functionality
- Basic 4D STEM data viewing

---

[1.1.0]: https://github.com/ondrejdyck/mib-viewer/compare/v1.0.12...v1.1.0
[1.0.12]: https://github.com/ondrejdyck/mib-viewer/compare/v1.0.11...v1.0.12
[1.0.11]: https://github.com/ondrejdyck/mib-viewer/compare/v1.0.0...v1.0.11
[1.0.0]: https://github.com/ondrejdyck/mib-viewer/releases/tag/v1.0.0
