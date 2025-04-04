from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="lfa-analysis",
    version="1.0.0",
    author="Keegan Spell",
    author_email="ks2398s@missouristate.edu",
    description="Lateral Flow Assay Analysis using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lochnech/lateral_flow_assay_LFA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lfa-train=lfa.scripts.train:main",
            "lfa-generate=lfa.scripts.generate_masks:main",
            "lfa-apply=lfa.scripts.apply_masks:main",
            "lfa-analyze=lfa.scripts.analyze_results:main",
        ],
    },
) 