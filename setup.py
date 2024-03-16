from setuptools import setup, find_packages

setup(
    name="mirrorverse",
    version="0.0.1",
    author="Marcel Gietzmann-Sanders",
    author_email="marcelsanders96@gmail.com",
    packages=find_packages(include=["mirrorverse", "mirrorverse*"]),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "train_chinook=mirrorverse.chinook.train:main",
            "simulate_chinook=mirrorverse.chinook.simulate:main",
            "chinook_states=mirrorverse.chinook.states:main",
        ]
    },
)
