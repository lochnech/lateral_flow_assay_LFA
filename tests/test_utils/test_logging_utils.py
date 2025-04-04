import os
import sys
import pytest
import logging
from pathlib import Path
import shutil

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lfa.utils.logging_utils import setup_logging

def test_setup_logging(tmp_path):
    """Test logging setup"""
    # Setup logging
    log_dir = tmp_path / "logs"
    logger = setup_logging(str(log_dir), name="test_logger")
    
    # Check logger properties
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    
    # Check that log directory was created
    assert log_dir.exists()
    
    # Check that log file was created
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) == 1
    
    # Test logging
    test_message = "Test log message"
    logger.info(test_message)
    
    # Verify log file contents
    with open(log_files[0], 'r') as f:
        log_content = f.read()
        assert test_message in log_content

def test_setup_logging_with_existing_dir(tmp_path):
    """Test logging setup with existing directory"""
    # Create log directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    # Setup logging
    logger = setup_logging(str(log_dir), name="test_logger")
    
    # Check that new log file was created
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) == 1

def test_setup_logging_with_custom_level(tmp_path):
    """Test logging setup with custom log level"""
    # Setup logging with DEBUG level
    logger = setup_logging(str(tmp_path / "logs"), name="test_logger", level=logging.DEBUG)
    
    # Check logger level
    assert logger.level == logging.DEBUG
    
    # Test logging at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Verify log file contents
    log_file = next((tmp_path / "logs").glob("*.log"))
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert "Debug message" in log_content
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content

def test_setup_logging_with_invalid_dir():
    """Test logging setup with invalid directory"""
    # Try to setup logging in a non-existent parent directory
    with pytest.raises(OSError):
        setup_logging("/nonexistent/path/logs", name="test_logger")

def test_setup_logging_with_readonly_dir(tmp_path):
    """Test logging setup with readonly directory"""
    # Create a readonly directory
    log_dir = tmp_path / "logs"
    log_dir.mkdir(mode=0o555)
    
    # Try to setup logging
    with pytest.raises(OSError):
        setup_logging(str(log_dir), name="test_logger")

def test_setup_logging_with_multiple_loggers(tmp_path):
    """Test setting up multiple loggers"""
    # Setup first logger
    logger1 = setup_logging(str(tmp_path / "logs1"), name="logger1")
    logger1.info("Message from logger1")
    
    # Setup second logger
    logger2 = setup_logging(str(tmp_path / "logs2"), name="logger2")
    logger2.info("Message from logger2")
    
    # Verify both log files exist
    assert len(list((tmp_path / "logs1").glob("*.log"))) == 1
    assert len(list((tmp_path / "logs2").glob("*.log"))) == 1
    
    # Verify log contents
    log_file1 = next((tmp_path / "logs1").glob("*.log"))
    log_file2 = next((tmp_path / "logs2").glob("*.log"))
    
    with open(log_file1, 'r') as f:
        assert "Message from logger1" in f.read()
    
    with open(log_file2, 'r') as f:
        assert "Message from logger2" in f.read()

def test_setup_logging_with_rotation(tmp_path):
    """Test logging with file rotation"""
    # Setup logging with small maxBytes to force rotation
    log_dir = tmp_path / "logs"
    logger = setup_logging(str(log_dir), name="test_logger", maxBytes=1000, backupCount=2)
    
    # Write enough log messages to trigger rotation
    for i in range(100):
        logger.info(f"Test message {i}")
    
    # Verify that rotation occurred
    log_files = sorted(log_dir.glob("*.log*"))
    assert len(log_files) > 1  # Should have at least one backup file
    
    # Verify that backupCount is respected
    assert len(log_files) <= 3  # Original + 2 backups
