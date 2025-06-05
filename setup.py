"""
Setup configuration for Adaptive Speculative Decoding
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-speculative-decoding",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-stage Draft-Verify pipeline with input-dependent depth optimization for LLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptive-speculative-decoding",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "black>=24.10.0",
            "flake8>=7.1.0",
            "mypy>=1.13.0",
            "pre-commit>=4.0.0",
        ],
        "onnx": [
            "onnx>=1.17.0",
            "onnxruntime>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-sd-server=src.serving.server:main",
            "adaptive-sd-train=scripts.train_predictor:main",
            "adaptive-sd-evaluate=experiments.evaluate_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)