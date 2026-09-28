Standalone CI branch: reproduces the tetgen "Invalid PLC! A point and a subface
intersect" failure seen on macOS in demo_mcxyz_skinvessel_shallowvessel.m, by
running its Cork surfboolean + tetgen meshing step with iso2mesh's macOS binaries.
Results are pushed to the ci-mac-cork-repro-results branch.
