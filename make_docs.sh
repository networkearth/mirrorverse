#!/bin/bash

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

rm docs/mirrorverse*
rm -rf docs/_build
sphinx-apidoc -o docs . -f
cd docs
make html
cd ..