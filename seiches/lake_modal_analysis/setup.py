"""
Setup script for lake_modal_analysis package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lake_modal_analysis",
    version="1.0.0",
    author="Carlos Rozas Rojas (original MATLAB), Python conversion 2025",
    author_email="crozas@ing.uchile.cl",
    description="Modal analysis of internal waves in stratified lakes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lake_modal_analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    include_package_data=True,
    keywords="oceanography limnology internal waves seiches modal analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/lake_modal_analysis/issues",
        "Source": "https://github.com/yourusername/lake_modal_analysis",
    },
)
