# Testing Guide

## Overview

The project uses pytest for testing. Tests are organized in the `tests` directory with the following structure:

```
tests/
├── test_core/
│   ├── test_mask_generation.py
│   ├── test_mask_application.py
│   └── test_analysis.py
├── test_utils/
│   ├── test_data_utils.py
│   ├── test_image_utils.py
│   └── test_logging_utils.py
├── test_models/
│   └── test_unet.py
└── conftest.py
```

## Running Tests

### Using the Test Runner Script

The easiest way to run tests is using the provided test runner script:

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test files
python scripts/run_tests.py tests/test_core/test_mask_generation.py

# Run tests with verbose output
python scripts/run_tests.py -v

# Run tests with coverage report
python scripts/run_tests.py -c
```

### Using pytest Directly

You can also run tests directly using pytest:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_core/test_mask_generation.py

# Run specific test functions
pytest tests/test_core/test_mask_generation.py::test_generate_mask

# Run tests with coverage
pytest --cov=lfa --cov-report=term-missing
```

## Test Categories

### Core Tests

#### Mask Generation Tests
- Test mask generation with different input images
- Test edge cases (empty images, invalid inputs)
- Test post-processing steps
- Test logging functionality

#### Mask Application Tests
- Test mask application with different image sizes
- Test overlay functionality
- Test output formats
- Test error handling

#### Analysis Tests
- Test metric calculations
- Test plot generation
- Test statistical analysis
- Test comparison functions

### Utility Tests

#### Data Utilities Tests
- Test data loading
- Test data preprocessing
- Test augmentation transforms
- Test dataset class

#### Image Utilities Tests
- Test image loading/saving
- Test image processing
- Test mask operations
- Test metric calculations

#### Logging Utilities Tests
- Test logging setup
- Test log file rotation
- Test log levels
- Test log formatting

### Model Tests

#### UNet Tests
- Test model initialization
- Test forward pass
- Test different input sizes
- Test different batch sizes
- Test gradient computation
- Test device compatibility

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def sample_image():
    # Returns a sample test image
    pass

@pytest.fixture
def sample_mask():
    # Returns a sample test mask
    pass

@pytest.fixture
def unet_model():
    # Returns an initialized UNet model
    pass
```

## Writing Tests

### Test Structure

Each test should follow this structure:

```python
def test_functionality():
    # Setup
    setup_data = ...
    
    # Execute
    result = function_to_test(setup_data)
    
    # Assert
    assert expected_condition
```

### Best Practices

1. Test both success and failure cases
2. Use meaningful test names
3. Keep tests independent
4. Use fixtures for common setup
5. Test edge cases
6. Include docstrings
7. Use type hints
8. Follow PEP 8 style

### Example Test

```python
def test_generate_mask(sample_image, unet_model):
    """Test mask generation with valid input."""
    # Setup
    image = sample_image
    model = unet_model
    
    # Execute
    mask = generate_mask(image, model)
    
    # Assert
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 255))
```

## Coverage

### Generating Coverage Reports

```bash
# Generate terminal report
pytest --cov=lfa --cov-report=term-missing

# Generate HTML report
pytest --cov=lfa --cov-report=html:coverage_report
```

### Coverage Goals

- Aim for >90% code coverage
- Focus on critical paths
- Document uncovered code
- Regularly update coverage reports

## Continuous Integration

Tests are automatically run on GitHub Actions for:
- Push to main branch
- Pull requests
- Scheduled runs

See `.github/workflows/test.yml` for configuration.

## Debugging Tests

### Common Issues

1. Test Dependencies
   - Ensure all required packages are installed
   - Check virtual environment

2. Resource Issues
   - Memory leaks
   - File permissions
   - Disk space

3. Environment Variables
   - Check required environment variables
   - Test environment setup

### Debugging Tools

1. pytest Options
   ```bash
   # Show print statements
   pytest -s
   
   # Show local variables on failure
   pytest --showlocals
   
   # Run specific test with debugger
   pytest --pdb
   ```

2. Logging
   - Use debug logging
   - Check log files
   - Monitor system resources

## Test Maintenance

1. Regular Updates
   - Update tests with code changes
   - Review test coverage
   - Remove obsolete tests

2. Performance
   - Optimize slow tests
   - Use appropriate fixtures
   - Clean up resources

3. Documentation
   - Keep test docs updated
   - Document test requirements
   - Maintain test examples 