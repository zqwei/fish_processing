from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).parent / "README.md"


setup(
    name="fish_proc",
    version="1.0.1",
    description="Utilities for analyzing imaging and behavioral data from zebrafish",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Ziqiang Wei",
    author_email="weiz@janelia.hhmi.org",
    url="https://github.com/zqwei/fish_processing",
    packages=find_packages(include=["fish_proc", "fish_proc.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0",
        "scipy>=1.13",
        "matplotlib>=3.8",
        "h5py>=3.10",
        "dask[array]>=2024.8",
        "distributed>=2024.8",
        "zarr>=2.17",
        "scikit-image>=0.23",
        "scikit-learn>=1.4",
        "networkx>=3.3",
        "psutil>=5.9",
        "tqdm>=4.66",
        "dipy>=1.9",
    ],
    extras_require={
        "spike": [
            "pandas>=2.2",
            "statsmodels>=0.14",
            "tensorflow>=2.18; python_version < '3.13'",
        ],
        "dev": [
            "pytest>=8.0",
        ],
        "all": [
            "pandas>=2.2",
            "statsmodels>=0.14",
            "tensorflow>=2.18; python_version < '3.13'",
            "pytest>=8.0",
        ],
    },
    zip_safe=False,
)
