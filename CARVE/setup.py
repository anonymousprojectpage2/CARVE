from setuptools import setup, find_packages

setup(
    name="carve",
    version="1.0.0",
    description="CARVE: Streaming Continual Merge for VLA Models",
    author="Anonymous (NeurIPS 2026 submission)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "safetensors>=0.4.0",
        "accelerate>=0.30.0",
        "einops",
        "numpy",
        "tqdm",
        "pyyaml",
    ],
)
