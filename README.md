# Lateral Flow Assay (LFA) Image Analysis

A Python package for analyzing and processing lateral flow assay (LFA) images using deep learning.

## Features

- **Image Segmentation**: Uses UNet architecture for accurate mask generation
- **Mask Processing**: Tools for applying and analyzing masks
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Analysis Tools**: Statistical analysis and visualization
- **Configuration Management**: Flexible YAML-based configuration
- **Testing Framework**: Comprehensive test suite with coverage reporting
- **Logging**: Detailed logging for all operations
- **CLI Tools**: Command-line interface for all major operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lateral_flow_assay_LFA.git
cd lateral_flow_assay_LFA
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training

```bash
python scripts/train.py --config config/training_config.yaml
```

### Mask Generation

```bash
python scripts/generate_masks.py --config config/mask_generation_config.yaml
```

### Mask Application

```bash
python scripts/apply_masks.py --config config/mask_application_config.yaml
```

### Analysis

```bash
python scripts/analyze_results.py --config config/analysis_config.yaml
```

### Testing

```bash
# Run all tests
python scripts/run_tests.py

# Run specific tests
python scripts/run_tests.py tests/test_core/test_mask_generation.py

# Run tests with coverage
python scripts/run_tests.py -c
```

## Project Structure

```
lateral_flow_assay_LFA/
├── lfa/
│   ├── core/
│   │   ├── mask_generation.py
│   │   ├── mask_application.py
│   │   └── analysis.py
│   ├── utils/
│   │   ├── data_utils.py
│   │   ├── image_utils.py
│   │   └── logging_utils.py
│   ├── models/
│   │   └── unet.py
│   └── config/
│       ├── default_config.yaml
│       ├── training_config.yaml
│       ├── mask_generation_config.yaml
│       ├── mask_application_config.yaml
│       └── analysis_config.yaml
├── scripts/
│   ├── train.py
│   ├── generate_masks.py
│   ├── apply_masks.py
│   ├── analyze_results.py
│   └── run_tests.py
├── tests/
│   ├── test_core/
│   ├── test_utils/
│   ├── test_models/
│   └── conftest.py
├── docs/
│   ├── api.md
│   ├── configuration.md
│   └── testing.md
├── requirements.txt
└── README.md
```

## Documentation

- [API Documentation](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Testing Guide](docs/testing.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing

Run the test suite:
```bash
python scripts/run_tests.py
```

Generate coverage report:
```bash
python scripts/run_tests.py -c
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UNet architecture based on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Inspired by various open-source medical image analysis projects 