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
            "chinook_train=mirrorverse.chinook.train:main",
            "chinook_simulate=mirrorverse.chinook.simulate:main",
            "chinook_states=mirrorverse.chinook.states:main",
            "mirrorverse_bundle_models=mirrorverse.utils:bundle_models",
        ]
    },
)
