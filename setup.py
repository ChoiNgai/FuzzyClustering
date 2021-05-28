import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FuzzyClustering",
    version="0.0.1",
    author="CAIWEI",
    author_email="caiwei-email@qq.com",
    description="Fuzzy clustering algorithm toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChoiNgai/FuzzyClustering",
    project_urls={
        "Bug Tracker": "https://github.com/ChoiNgai/FuzzyClustering/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)