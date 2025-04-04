# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete project structure with core modules, utilities, and models
- Comprehensive test suite with pytest
- Configuration management system using YAML files
- Detailed documentation including API, configuration, and testing guides
- Command-line interface for all major operations
- GitHub Actions workflows for testing and deployment
- Issue templates for bug reports and feature requests

### Core Modules
- Mask generation module with UNet-based segmentation
- Mask application module for processing and overlaying masks
- Analysis module for statistical analysis and visualization
- Data utilities for dataset handling and augmentation
- Image utilities for processing and transformation
- Logging utilities for comprehensive logging

### Models
- UNet implementation for image segmentation
- Support for different input sizes and batch sizes
- Gradient computation and device compatibility

### Configuration
- Default configuration for all operations
- Training-specific configuration
- Mask generation configuration
- Mask application configuration
- Analysis configuration
- Environment variable overrides

### Testing
- Test runner script with coverage reporting
- Core module tests (mask generation, application, analysis)
- Utility module tests (data, image, logging)
- Model tests (UNet)
- Test fixtures and common utilities
- Continuous integration with GitHub Actions

### Documentation
- API documentation for all modules and functions
- Configuration guide with examples
- Testing guide with best practices
- Updated README with project structure and usage
- Contributing guidelines
- License information

### Scripts
- Training script with early stopping and checkpointing
- Mask generation script with post-processing
- Mask application script with overlay options
- Analysis script with metrics and plots
- Test runner script with coverage options

## [0.1.0] - 2024-04-04

### Added
- Initial project setup
- Basic UNet implementation
- Core image processing functionality
- Basic documentation structure

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A 