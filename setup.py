"""
Setup script for Adaptive Speculative Decoding.

This package implements a multi-stage adaptive inference pipeline for Large Language Models
with dynamic quality prediction and cost optimization.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
def get_version():
    init_path = Path(__file__).parent / "src" / "__init__.py"
    if init_path.exists():
        with open(init_path, 'r') as f:
            content = f.read()
            match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
            if match:
                return match.group(1)
    return "2.0.0"

# Read long description from README
def get_long_description():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def get_requirements(filename='requirements.txt'):
    req_path = Path(__file__).parent / filename
    if req_path.exists():
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Development requirements
dev_requirements = [
    # Testing
    'pytest>=7.0.0',
    'pytest-asyncio>=0.21.0',
    'pytest-cov>=4.0.0',
    'pytest-mock>=3.10.0',
    'pytest-xdist>=3.0.0',
    
    # Code quality
    'black>=23.0.0',
    'isort>=5.12.0',
    'flake8>=6.0.0',
    'mypy>=1.0.0',
    'pylint>=2.17.0',
    
    # Documentation
    'sphinx>=6.0.0',
    'sphinx-rtd-theme>=1.2.0',
    'myst-parser>=1.0.0',
    
    # Development tools
    'pre-commit>=3.0.0',
    'jupyter>=1.0.0',
    'ipython>=8.0.0',
]

# Research requirements (for experiments and evaluation)
research_requirements = [
    'datasets>=3.0.0',
    'evaluate>=0.4.0',
    'wandb>=0.15.0',
    'tensorboard>=2.13.0',
    'plotly>=5.14.0',
    'bokeh>=3.1.0',
]

# Production requirements (for deployment)
production_requirements = [
    'gunicorn>=21.0.0',
    'uvicorn[standard]>=0.22.0',
    'prometheus-client>=0.17.0',
    'sentry-sdk>=1.28.0',
]

setup(
    # Basic package information
    name="adaptive-speculative-decoding",
    version=get_version(),
    author="Adaptive SD Research Team",
    author_email="research@adaptive-sd.ai",
    description="Multi-stage adaptive inference pipeline for Large Language Models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/sa2shun/adaptive-speculative-decoding",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=get_requirements(),
    extras_require={
        'dev': dev_requirements,
        'research': research_requirements,
        'production': production_requirements,
        'all': dev_requirements + research_requirements + production_requirements,
    },
    
    # Package data
    package_data={
        "adaptive_sd": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.html",
            "static/*",
        ]
    },
    include_package_data=True,
    
    # Entry points
    entry_points={
        'console_scripts': [
            'adaptive-sd-server=adaptive_sd.cli.server:main',
            'adaptive-sd-train=adaptive_sd.cli.train:main',
            'adaptive-sd-evaluate=adaptive_sd.cli.evaluate:main',
            'adaptive-sd-benchmark=adaptive_sd.cli.benchmark:main',
            'adaptive-sd-config=adaptive_sd.cli.config:main',
        ],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords="llm, inference, optimization, adaptive, speculative-decoding, machine-learning",
    
    project_urls={
        "Bug Reports": "https://github.com/sa2shun/adaptive-speculative-decoding/issues",
        "Source": "https://github.com/sa2shun/adaptive-speculative-decoding",
        "Documentation": "https://adaptive-sd.readthedocs.io/",
        "Research Paper": "https://arxiv.org/abs/2024.adaptive-sd",
    },
    
    # Installation options
    zip_safe=False,
    
    # Testing
    test_suite='tests',
    tests_require=dev_requirements,
    
    # Additional metadata
    license="Apache License 2.0",
    platforms=["any"],
)