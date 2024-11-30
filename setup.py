from setuptools import setup, find_packages

setup(
    name="feature_selection",
    version="1.0.0",
    description="A project to explore the effects of different parameters on dataset noise and feature selection using animations and plots.",
    url="https://github.com/vHalenka/Feature-Selection",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm",
        "opencv-python",
        "setuptools",
        "jupyter",
        "seaborn"
    ],
    extras_require={
        "dev": [
            "black",
            "flake8"
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "generate-animations=GenerateAnimations:main",
            "generate-plots=GeneratePlots:main",
        ],
    },
)
