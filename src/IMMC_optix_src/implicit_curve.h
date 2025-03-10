#include <optix.h>
#include <iostream>

namespace mcx {
// implicit curve represented with two vertices and width
struct ImplicitCurve {
    float3 vertex1;
    float3 vertex2;
    float width;
};
}
