#!/bin/bash

cd ../src
make clean
make

cd ../examples/optix-homogcube/
gdb --args ../../src/bin/mmc --compute optix -f optix.json -A 0 -b 1 -F bin

