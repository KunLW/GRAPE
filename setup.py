from setuptools import find_packages, setup


setup(
    name="grape-verge",
    version="0.1.0",
    description="Modular quantum control engine with perturbative expansion evolution.",
    packages=find_packages(include=["quantum_control", "quantum_control.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
        ],
    },
)
