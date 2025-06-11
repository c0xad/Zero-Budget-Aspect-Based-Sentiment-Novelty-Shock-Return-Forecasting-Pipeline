"""Setup script for ABSSENT package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Zero-Budget Aspect-Based Sentiment & Novelty-Shock Return Forecasting Pipeline"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="abssent",
    version="0.1.0",
    author="Research Team",
    author_email="research@abssent.ai",
    description="Zero-Budget Aspect-Based Sentiment & Novelty-Shock Return Forecasting Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abssent/abssent",
    project_urls={
        "Bug Reports": "https://github.com/abssent/abssent/issues",
        "Source": "https://github.com/abssent/abssent",
        "Documentation": "https://abssent.readthedocs.io",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.0.280",
            "black>=23.7.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "ml": [
            "mlflow>=2.5.0",
        ],
        "web": [
            "streamlit>=1.25.0",
        ],
        "all": [
            "mlflow>=2.5.0",
            "streamlit>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "abssent=abssent.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "abssent": ["data/*.yaml", "data/*.json"],
    },
    zip_safe=False,
) 