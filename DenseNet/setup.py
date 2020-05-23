import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DenseNet-armhzjz",
    version="0.1.1",
    author="Armando Hernandez",
    author_email="armhzjz@pm.me",
    description="A DenseNet implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/armhzjz/DenseNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

