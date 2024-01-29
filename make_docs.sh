#!/bin/bash

rm docs/mirrorverse*
rm -rf docs/_build
sphinx-apidoc -o docs . -f
cd docs
make html
cd ..
