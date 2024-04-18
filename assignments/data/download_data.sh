#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
mkdir "$BASEDIR/housenumbers"
cd "$BASEDIR/housenumbers"
wget -c http://ufldl.stanford.edu/housenumbers/train_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
