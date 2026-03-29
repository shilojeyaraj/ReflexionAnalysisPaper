"""Setup script for the reflexion_memory_study package."""

from setuptools import setup, find_packages

setup(
    name="reflexion_memory_study",
    version="0.1.0",
    description="Comparative study of persistent memory backends for Reflexion agents",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "datasets>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
    ],
)
