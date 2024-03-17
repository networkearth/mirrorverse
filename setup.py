from setuptools import setup, find_packages

setup(
    name="mirrorverse",
    version="0.0.1",
    author="Marcel Gietzmann-Sanders",
    author_email="marcelsanders96@gmail.com",
    packages=find_packages(include=["mirrorverse", "mirrorverse*"]),
    install_requires=[
        "pandas==2.1.2",
        "numpy==1.24.4",
        "scikit-learn==1.3.2",
        "h3==3.7.6",
        "geopy==2.4.1",
        "dvc==3.48.4",
        "click==8.1.7",
        "tqdm==4.66.1",
        "graphviz==0.20.1",
    ],
    entry_points={
        "console_scripts": [
            "chinook_train=mirrorverse.chinook.train:main",
            "chinook_simulate=mirrorverse.chinook.simulate:main",
            "chinook_states=mirrorverse.chinook.states:main",
            "mirrorverse_bundle_models=mirrorverse.utils:bundle_models",
            "mirrorverse_graph_decision_tree=mirrorverse.graph:main",
        ]
    },
)
