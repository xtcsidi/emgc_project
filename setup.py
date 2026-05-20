from setuptools import setup, find_packages

setup(
    name="emgc",
    version="0.1.0",
    description="Elastic Memory-Gated Compression toolkit for Decentralized Federated Learning",
    author="Sidi Koka",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pynvml",
        "psutil",
        "flwr"
    ],
    extras_require={
        "full": ["bitsandbytes"]
    }
)
