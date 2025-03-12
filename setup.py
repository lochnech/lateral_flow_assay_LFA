from setuptools import setup, find_packages

setup(
    name="lateral_flow_assay",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "albumentations",
        "opencv-python",
        "numpy",
        "Pillow",
        "matplotlib",
    ],
) 