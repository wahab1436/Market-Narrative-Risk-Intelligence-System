from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="market-narrative-risk-intelligence",
    version="1.0.0",
    author="Market Intelligence Team",
    author_email="team@marketintelligence.ai",
    description="Market Narrative Risk Intelligence System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/market-narrative-risk-intelligence",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "market-risk-pipeline=main:main",
            "market-risk-dashboard=src.dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    data_files=[
        ("config", ["config/config.yaml", "config/feature_config.yaml", "config/model_config.yaml"]),
    ],
)
