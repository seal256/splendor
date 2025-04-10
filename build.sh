#!/bin/bash

BUILD_TYPE="Release" # Release or Debug
if [ "$#" -eq 1 ]; then
    BUILD_TYPE=$1
fi

mkdir build
mkdir bin

cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
cmake --build . -j 8

cd ..
rm ./splendor
cp ./bin/splendor .
