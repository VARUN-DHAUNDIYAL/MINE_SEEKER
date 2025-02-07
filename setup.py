# setup.py
from setuptools import setup, find_packages

setup(
    name='RF-Landmine-Detection',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'tqdm', 'scikit-learn', 'pyyaml',
        'torch', 'torchvision', 'torchaudio', 'uhd', 'onnx', 'onnxruntime',
        'tensorrt', 'paho-mqtt', 'Flask'
    ],
    entry_points={
        'console_scripts': [
            'rf-landmine=src.main:main'
        ],
    },
)
