"""
Setup script for SkySim5000 Analysis Tools
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="skysim5000-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Analysis tools for SkySim5000 galaxy catalog",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/skysim5000-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "astropy>=5.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipython>=7.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "skysim5000-lf=skysim5000_analysis.scripts.calculate_lf:main",
            "skysim5000-colors=skysim5000_analysis.scripts.analyze_colors:main",
        ],
    },
    include_package_data=True,
    package_data={
        "skysim5000_analysis": ["data/*.txt", "data/*.csv"],
    },
)