#!/bin/sh
compute-sanitizer --tool memcheck --leak-check=full ../../src/bin/mmc --compute optix -f optix.json -b 1 -F bin $@
