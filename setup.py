import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rationalnets",
    version="0.0.1",
    install_requires=[
        "jax",
        "flax"
    ],
    author="yonesuke",
    author_email="13e.e.c.13@gmail.com",
    description="JAX/Flax implementation of rational neural nets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yonesuke/RationalNets",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)