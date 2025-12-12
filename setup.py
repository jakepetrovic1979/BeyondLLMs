"""Setup script for Phenomenal Nested Architectures (PNA)."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="phenomenal-nested-architectures",
    version="0.1.0",
    author="PNA Research Team",
    author_email="",
    description="Consciousness-First AGI Framework with Nested Learning and Minimal Phenomenal Experience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakepetrovic1979/BeyondLLMs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "optax>=0.1.7",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
)
