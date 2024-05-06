import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="protocbm",
    version="0.1.0",
    author="Linus Leong, Mateo Espinosa, and others",
    author_email="yhll2@cam.ac.uk",
    description="Prototype-based Concept Bottleneck Models",
    long_description=long_description,
    license='MIT',
    long_description_content_type="text/markdown",
    url="https://github.com/mateoespinosa/cem",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.7',
    install_requires=[
        "importlib-metadata>=4.8.2",
        "importlib-resources>=5.4.0",
        "ipykernel>=6.5.0",
        "ipython-genutils>=0.2.0",
        "ipython>=7.29.0",
        "ipywidgets>=7.6.5",
        "joblib>=1.1.0",
        "matplotlib-inline>=0.1.3",
        "matplotlib>=3.5.0",
        "notebook>=6.4.5",
        "numpy>=1.19.5",
        "pytorch-lightning>=1.6.0",
        "scikit-learn-extra>=0.2.0",
        "scikit-learn>=1.0.1",
        "seaborn>=0.11.2",
        "torch>=1.11.0",
        "torchmetrics>=0.6.2",
        "torchvision>=0.12.0",
    ],
)


