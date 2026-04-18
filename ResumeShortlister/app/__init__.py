"""
Application Factory

This module contains the application factory pattern for creating Flask app instances.
It handles configuration loading, extension initialization, and blueprint registration.

Environment variables from a local .env file are loaded early so that configuration
classes (which read os.environ at import time) see the correct values. This prevents
errors like missing SECRET_KEY in production when running entrypoints that do not
explicitly call load_dotenv (e.g. app.py or a WSGI server importing the package).
"""

from flask import Flask
import logging.config
import os

# Load environment variables BEFORE importing config that reads them.
try:
    import importlib
    # Only import python-dotenv if it's installed to avoid static import errors in editors/environments
    if importlib.util.find_spec('dotenv') is not None: # type: ignore
        load_dotenv = importlib.import_module('dotenv').load_dotenv
        load_dotenv()  # safe to call multiple times; silently ignored if .env absent
except Exception:
    # Fail silently; config classes will still rely on real environment vars.
    pass

from app.config.settings import get_config
from app.extensions import init_extensions
from app.controllers import register_blueprints
from app.utils.error_handlers import register_error_handlers

def create_app(config_name='development'):
    """
    Application factory function that creates and configures a Flask application.
    
    Args:
        config_name (str): Configuration environment name ('development', 'production', 'testing')
        
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Ensure required directories exist
    _create_required_directories(app)
    
    # Setup logging
    _setup_logging(app)
    
    # Initialize extensions
    init_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Log application startup
    app.logger.info(f"RSART application created with config: {config_name}")
    
    return app

def _create_required_directories(app):
    """
    Create required directories if they don't exist.
    
    Args:
        app: Flask application instance
    """
    directories = [
        app.config.get('UPLOAD_FOLDER'),
        app.config.get('JOB_DESCRIPTIONS_FOLDER'),
        app.config.get('LOG_FILE').parent if app.config.get('LOG_FILE') else None
    ]
    
    for directory in directories:
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

def _setup_logging(app):
    """
    Setup application logging configuration.
    
    Args:
        app: Flask application instance
    """
    if app.config.get('LOGGING_CONFIG') and os.path.exists(app.config['LOGGING_CONFIG']):
        logging.config.fileConfig(app.config['LOGGING_CONFIG'])
    else:
        # Fallback logging configuration
        logging.basicConfig(
            level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )