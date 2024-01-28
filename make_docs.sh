#!/bin/bash

rm docs/mirrorverse*
sphinx-apidoc -o docs . -f
cd docs
make html
cd ..
