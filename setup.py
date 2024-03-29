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
        "Sphinx==7.2.6",
        "SQLAlchemy==2.0.28",
        "Fiona==1.8.22",
        "geopandas==0.14.3",
        "eralchemy2==1.3.8",
        "netCDF4==1.6.5",
        "earthengine-api==0.1.390",
    ],
    entry_points={
        "console_scripts": [
            "chinook_train=mirrorverse.chinook.train:main",
            "chinook_simulate=mirrorverse.chinook.simulate:main",
            "chinook_states=mirrorverse.chinook.states:main",
            "chinook_db=mirrorverse.chinook.db:main",
            "mirrorverse_bundle_models=mirrorverse.utils:bundle_models",
            "mirrorverse_graph_decision_tree=mirrorverse.graph:main",
            "mirrorverse_upload_facts=mirrorverse.warehouse.commands:upload_facts",
            "mirrorverse_enumerate_missing_dimensions=mirrorverse.warehouse.etls.missing_dimensions:enumerate_missing_dimensions",
            "mirrorverse_upload_dimensions=mirrorverse.warehouse.commands:upload_dimensions",
            "mirrorverse_prep_cwt_query=mirrorverse.warehouse.etls.missing_dimensions:prep_cwt_query",
            "mirrorverse_build_erd=mirrorverse.warehouse.commands:build_erd",
            "mirrorverse_file_import=mirrorverse.docks.commands:file_import",
        ]
    },
)
