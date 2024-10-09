from setuptools import setup, find_packages

setup(
    name="mirrorverse",
    version="0.0.1",
    author="Marcel Gietzmann-Sanders",
    author_email="marcelsanders96@gmail.com",
    packages=find_packages(include=["mirrorverse", "mirrorverse*"]),
    install_requires=[
        "h3==3.7.7",
        "shapely==2.0.6",
        "geopandas==1.0.1",
        "plotly==5.24.1",
        "nbformat==5.10.4",
    ],
)