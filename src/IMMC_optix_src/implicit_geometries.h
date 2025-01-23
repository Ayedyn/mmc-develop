#ifndef _IMPLICIT_GEOMETRIES_
#define _IMPLICIT_GEOMETRIES_

namespace mcx {
	    // implicit curve represented with two vertices and width
        struct ImplicitCurve {
                float3 vertex1;
                float3 vertex2;
                float width;
        };

        // implicit sphere represented with position and radius
        struct ImplicitSphere {
                float3 position;
                float radius;
        };
}
#endif
