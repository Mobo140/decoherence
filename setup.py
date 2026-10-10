"""Setup script for quantum decoherence prediction project."""
from setuptools import setup, find_packages

setup(
    name='quantum-decoherence-prediction',
    version='0.1.0',
    description=(
        'Predicting quantum decoherence times with a physics-informed '
        'PyTorch model'
    ),
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'qutip>=5.0.0',
        'torch>=2.0.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'fastapi>=0.110.0',
        'pydantic>=2.0.0',
        'uvicorn>=0.27.0',
    ],
    extras_require={
        # notebooks/figures.ipynb
        'notebook': ['matplotlib>=3.7.0', 'pandas>=2.0.0'],
        # tests/ (fastapi's TestClient needs httpx)
        'test': ['pytest', 'httpx>=0.27.0'],
    },
    python_requires='>=3.11',
)
