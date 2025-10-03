"""MIB Data Analysis Suite

A comprehensive toolkit for analyzing MIB EELS data and 4D STEM datasets 
with interactive visualization and format conversion capabilities.
"""

__version__ = "1.0.12"

def main():
    """Main entry point for the MIB viewer GUI application."""
    from .main import main as main_func
    main_func()