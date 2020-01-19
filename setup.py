import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
]

setuptools.setup(
    name="glimpy",
    version="0.1.0",
    author="Kyle Safran",
    author_email="ksafran356@gmail.com",
    description="Generalized Linear Models in Python",
    long_description=long_description,
    download_url="https://github.com/KSafran/glimpy/archive/0.1.0.tar.gz",
    long_description_content_type="text/markdown",
    url="https://github.com/KSafran/glimpy",
    packages=['glimpy'],
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
        "statsmodels"
    ],
    python_requires='>=3.6',
    license="MIT"
)
