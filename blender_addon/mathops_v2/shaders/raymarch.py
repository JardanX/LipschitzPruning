from .. import runtime


UNIFORMS_SOURCE = """
struct MathOPSV2ViewParams {
  vec4 cameraPosition;
  vec4 surfaceDistanceCounts;
  vec4 instructionDebugSpecular;
  vec4 sceneLayout;
  vec4 renderBoundsMin;
  vec4 renderBoundsMax;
  vec4 pruningBoundsMinGridSize;
  vec4 pruningBoundsMaxEnabled;
  vec4 viewRow0;
  vec4 viewRow1;
  vec4 viewRow2;
  vec4 viewRow3;
};
"""


VERTEX_SOURCE = """
void main()
{
  uvInterp = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}
"""


COMMON_SOURCE = """\
#define PRIMITIVE_STRIDE %d
#define MAX_STACK %d
#define MIRROR_WARP_PACK_SCALE 256
#define WARP_ROWS_PER_ENTRY 9
#define WARP_KIND_MIRROR 1
#define WARP_KIND_GRID 2
#define WARP_KIND_RADIAL 3
#define MIRROR_AXIS_X 1
#define MIRROR_AXIS_Y 2
#define MIRROR_AXIS_Z 4
#define MIRROR_SIDE_X 8
#define MIRROR_SIDE_Y 16
#define MIRROR_SIDE_Z 32
#define WARP_PI 3.14159265358979323846

vec4 loadRow(int row)
{
  return texelFetch(sceneData, ivec2(0, row), 0);
}

ivec2 linearCoord(ivec2 size, int index)
{
  int width = max(size.x, 1);
  return ivec2(index %% width, index / width);
}

float loadScalar(sampler2D textureSampler, int index)
{
  return texelFetch(textureSampler, linearCoord(textureSize(textureSampler, 0), index), 0).x;
}

vec4 loadPolygonRow(int index)
{
  return texelFetch(polygonData, linearCoord(textureSize(polygonData, 0), index), 0);
}

vec2 loadPolygonPoint(int index)
{
  return loadPolygonRow(index).xy;
}

vec2 loadBezierPoint(int pointOffset, int index)
{
  return loadPolygonRow(pointOffset + index * 2).xy;
}

vec2 loadBezierHandleRight(int pointOffset, int index)
{
  return loadPolygonRow(pointOffset + index * 2).zw;
}

vec2 loadBezierHandleLeft(int pointOffset, int index)
{
  return loadPolygonRow(pointOffset + index * 2 + 1).xy;
}

vec3 inferno(float t)
{
  const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
  const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
  const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

vec3 primitiveScale(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return max(loadRow(baseRow + 4).xyz, vec3(1e-6));
}

vec4 primitiveExtra0(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return loadRow(baseRow + 5);
}

vec4 primitiveExtra1(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return loadRow(baseRow + 6);
}

vec4 primitiveExtra2(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return loadRow(baseRow + 7);
}

float surfaceEpsilonValue()
{
  return mathops.surfaceDistanceCounts.x;
}

float maxDistanceValue()
{
  return mathops.surfaceDistanceCounts.y;
}

int maxStepsValue()
{
  return int(mathops.surfaceDistanceCounts.z + 0.5);
}

int primitiveCountValue()
{
  return int(mathops.surfaceDistanceCounts.w + 0.5);
}

int instructionCountValue()
{
  return int(mathops.instructionDebugSpecular.x + 0.5);
}

int warpRowCountValue()
{
  return int(mathops.sceneLayout.x + 0.5);
}

int instructionBaseValue()
{
  return primitiveCountValue() * PRIMITIVE_STRIDE + warpRowCountValue();
}

int pruningGridSizeValue()
{
  return int(mathops.pruningBoundsMinGridSize.w + 0.5);
}

bool pruningEnabledValue()
{
  return mathops.pruningBoundsMaxEnabled.w > 0.5;
}

vec3 pruningBoundsMinValue()
{
  return mathops.pruningBoundsMinGridSize.xyz;
}

vec3 pruningBoundsMaxValue()
{
  return mathops.pruningBoundsMaxEnabled.xyz;
}

vec3 renderBoundsMinValue()
{
  return mathops.renderBoundsMin.xyz;
}

vec3 renderBoundsMaxValue()
{
  return mathops.renderBoundsMax.xyz;
}

int debugModeValue()
{
  return int(mathops.instructionDebugSpecular.y + 0.5);
}

float debugVizMaxValue()
{
  return max(mathops.instructionDebugSpecular.z, 1.0);
}

bool showSpecularValue()
{
  return mathops.instructionDebugSpecular.w > 0.5;
}

float gammaValue()
{
  return max(0.001, mathops.viewRow3.w);
}

int viewFlagsValue()
{
  return int(mathops.cameraPosition.w + 0.5);
}

bool orthographicViewValue()
{
  return (viewFlagsValue() & 1) != 0;
}

bool disableSurfaceShadingValue()
{
  return (viewFlagsValue() & 2) != 0;
}

float sabs(float x, float k)
{
  if (k <= 0.0001) {
    return abs(x);
  }
  float ax = abs(x);
  if (ax >= k) {
    return ax;
  }
  float t = ax / k;
  float t2 = t * t;
  float t3 = t2 * t;
  float t4 = t2 * t2;
  return k * (0.25 + 1.5 * t2 - t3 + 0.25 * t4);
}

float foldFiniteGridAxis(float axisValue, float origin, float primitiveCenter, float spacing, int count, float blend)
{
  if (count <= 1 || spacing <= 1e-6) {
    return axisValue;
  }
  float base = (primitiveCenter - origin) / spacing;
  float centerOffset = 0.5 * float(count - 1);
  float shifted = (axisValue - origin) / spacing - base + centerOffset;
  float id = clamp(round(shifted), 0.0, float(count - 1));
  float local = shifted - id;
  float dR = (0.5 - local) * spacing;
  float pullR = dR - sabs(dR, blend);
  float dL = (0.5 + local) * spacing;
  float pullL = dL - sabs(dL, blend);
  if (id < 0.5) {
    pullL = 0.0;
  }
  if (id > float(count) - 1.5) {
    pullR = 0.0;
  }
  local += (pullR - pullL) / spacing;
  return origin + (local + base) * spacing;
}

vec3 worldToPrimitiveLocal(vec4 row0, vec4 row1, vec4 row2, vec3 worldPoint)
{
  vec4 world = vec4(worldPoint, 1.0);
  return vec3(dot(row0, world), dot(row1, world), dot(row2, world));
}

vec3 primitiveLocalToWorld(vec4 row0, vec4 row1, vec4 row2, vec3 localPoint)
{
  vec3 basisX = vec3(row0.x, row1.x, row2.x);
  vec3 basisY = vec3(row0.y, row1.y, row2.y);
  vec3 basisZ = vec3(row0.z, row1.z, row2.z);
  vec3 origin = -(basisX * row0.w + basisY * row1.w + basisZ * row2.w);
  return origin + (basisX * localPoint.x) + (basisY * localPoint.y) + (basisZ * localPoint.z);
}

vec4 unpackWarpRotation(vec3 packedXYZ)
{
  return normalize(vec4(packedXYZ, sqrt(max(1.0 - dot(packedXYZ, packedXYZ), 0.0))));
}

vec3 rotateByQuaternion(vec4 rotation, vec3 point)
{
  vec3 axis = rotation.xyz;
  float scalar = rotation.w;
  return (2.0 * dot(axis, point) * axis)
       + ((scalar * scalar - dot(axis, axis)) * point)
       + (2.0 * scalar * cross(axis, point));
}

vec3 safeArrayScale(vec3 scale)
{
  return max(abs(scale), vec3(1e-6));
}

vec3 worldToArrayLocal(vec3 worldPoint, vec3 origin, vec4 rotation, vec3 scale, vec3 branchCenter)
{
  return branchCenter + rotateByQuaternion(vec4(-rotation.xyz, rotation.w), worldPoint - origin) / safeArrayScale(scale);
}

vec3 applyInstanceOffsetInverse(vec3 point, vec3 branchCenter, vec3 offsetLocation, vec4 offsetRotation, vec3 offsetScale)
{
  vec3 centered = point - (branchCenter + offsetLocation);
  vec3 local = rotateByQuaternion(vec4(-offsetRotation.xyz, offsetRotation.w), centered) / safeArrayScale(offsetScale);
  return branchCenter + local;
}

vec3 applyRadialArray(vec3 worldPoint, vec3 origin, vec3 primitiveCenter, float radius, int count, float blend)
{
  if (count <= 1 || radius <= 1e-6) {
    return worldPoint;
  }
  vec3 q = worldPoint - origin;
  vec3 baseOffset = primitiveCenter - origin + vec3(radius, 0.0, 0.0);
  float baseAngle = atan(baseOffset.y, baseOffset.x);
  float baseRadius = max(length(baseOffset.xy), 1e-4);
  float sector = (2.0 * WARP_PI) / float(count);
  float angle = atan(q.y, q.x);
  float angleRel = mod(angle - baseAngle + WARP_PI, 2.0 * WARP_PI) - WARP_PI;
  float normA = angleRel / sector;
  float id = round(normA);
  float local = normA - id;
  float arc = sector * baseRadius;
  float dR = (0.5 - local) * arc;
  float pullR = dR - sabs(dR, blend);
  float dL = (0.5 + local) * arc;
  float pullL = dL - sabs(dL, blend);
  local += (pullR - pullL) / arc;
  float foldA = local * sector + baseAngle;
  float r = length(q.xy);
  vec3 qFold = vec3(r * cos(foldA), r * sin(foldA), q.z);
  return primitiveCenter + (qFold - baseOffset);
}

vec3 applyWarpStack(int baseRow, vec3 worldPoint, int limitCount, out float distanceScale)
{
  int packedWarpInfo = int(loadRow(baseRow + 4).w + 0.5);
  int warpBase = primitiveCountValue() * PRIMITIVE_STRIDE;
  int warpOffset = packedWarpInfo / MIRROR_WARP_PACK_SCALE;
  int warpCount = packedWarpInfo - warpOffset * MIRROR_WARP_PACK_SCALE;
  int applyCount = (limitCount < 0) ? warpCount : min(limitCount, warpCount);
  vec3 warpedPoint = worldPoint;
  distanceScale = 1.0;
  for (int index = 0; index < applyCount; index++) {
    int warpRow = warpBase + warpOffset + index * WARP_ROWS_PER_ENTRY;
    vec4 warp0 = loadRow(warpRow);
    vec4 warp1 = loadRow(warpRow + 1);
    vec4 warp2 = loadRow(warpRow + 2);
    vec4 warp3 = loadRow(warpRow + 3);
    vec4 warp4 = loadRow(warpRow + 4);
    vec4 warp5 = loadRow(warpRow + 5);
    vec4 warp6 = loadRow(warpRow + 6);
    vec4 warp7 = loadRow(warpRow + 7);
    vec4 warp8 = loadRow(warpRow + 8);
    int warpKind = int(warp0.x + 0.5);
    if (warpKind == WARP_KIND_MIRROR) {
      int flags = int(warp0.y + 0.5);
      float blend = max(warp0.z, 0.0);
      vec3 origin = warp1.xyz;
      if ((flags & MIRROR_AXIS_X) != 0) {
        float side = ((flags & MIRROR_SIDE_X) != 0) ? 1.0 : -1.0;
        warpedPoint.x = origin.x + side * sabs(side * (warpedPoint.x - origin.x), blend);
      }
      if ((flags & MIRROR_AXIS_Y) != 0) {
        float side = ((flags & MIRROR_SIDE_Y) != 0) ? 1.0 : -1.0;
        warpedPoint.y = origin.y + side * sabs(side * (warpedPoint.y - origin.y), blend);
      }
      if ((flags & MIRROR_AXIS_Z) != 0) {
        float side = ((flags & MIRROR_SIDE_Z) != 0) ? 1.0 : -1.0;
        warpedPoint.z = origin.z + side * sabs(side * (warpedPoint.z - origin.z), blend);
      }
      continue;
    }
    if (warpKind == WARP_KIND_GRID) {
      ivec3 counts = ivec3(int(warp0.y + 0.5), int(warp0.z + 0.5), int(warp0.w + 0.5));
      vec3 spacing = abs(warp1.xyz);
      float blend = max(warp1.w, 0.0);
      vec3 origin = warp2.xyz;
      vec4 rotation = unpackWarpRotation(warp3.xyz);
      vec3 scale = warp4.xyz;
      vec3 branchCenter = warp5.xyz;
      vec3 offsetLocation = warp6.xyz;
      vec4 offsetRotation = unpackWarpRotation(warp7.xyz);
      vec3 offsetScale = warp8.xyz;
      distanceScale *= max(min(scale.x, min(scale.y, scale.z)), 1e-6);
      distanceScale *= max(min(offsetScale.x, min(offsetScale.y, offsetScale.z)), 1e-6);
      vec3 arrayPoint = worldToArrayLocal(warpedPoint, origin, rotation, scale, branchCenter);
      vec3 offsetCenter = branchCenter + offsetLocation;
      arrayPoint.x = foldFiniteGridAxis(arrayPoint.x, offsetCenter.x, offsetCenter.x, spacing.x, counts.x, blend);
      arrayPoint.y = foldFiniteGridAxis(arrayPoint.y, offsetCenter.y, offsetCenter.y, spacing.y, counts.y, blend);
      arrayPoint.z = foldFiniteGridAxis(arrayPoint.z, offsetCenter.z, offsetCenter.z, spacing.z, counts.z, blend);
      warpedPoint = applyInstanceOffsetInverse(arrayPoint, branchCenter, offsetLocation, offsetRotation, offsetScale);
      continue;
    }
    if (warpKind == WARP_KIND_RADIAL) {
      int count = int(warp0.y + 0.5);
      float blend = max(warp0.z, 0.0);
      float radius = max(warp0.w, 0.0);
      vec3 repeatOrigin = warp1.xyz;
      vec3 fieldOrigin = warp2.xyz;
      vec4 rotation = unpackWarpRotation(warp3.xyz);
      vec3 scale = warp4.xyz;
      vec3 branchCenter = warp5.xyz;
      vec3 offsetLocation = warp6.xyz;
      vec4 offsetRotation = unpackWarpRotation(warp7.xyz);
      vec3 offsetScale = warp8.xyz;
      distanceScale *= max(min(scale.x, min(scale.y, scale.z)), 1e-6);
      distanceScale *= max(min(offsetScale.x, min(offsetScale.y, offsetScale.z)), 1e-6);
      vec3 arrayPoint = worldToArrayLocal(warpedPoint, fieldOrigin, rotation, scale, branchCenter);
      arrayPoint = applyRadialArray(arrayPoint, repeatOrigin, branchCenter + offsetLocation, radius, count, blend);
      warpedPoint = applyInstanceOffsetInverse(arrayPoint, branchCenter, offsetLocation, offsetRotation, offsetScale);
    }
  }
  return warpedPoint;
}

vec3 toLocalPoint(int primitiveIndex, vec3 worldPoint, out float distanceScale)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 row0 = loadRow(baseRow + 1);
  vec4 row1 = loadRow(baseRow + 2);
  vec4 row2 = loadRow(baseRow + 3);
  vec3 warpedPoint = applyWarpStack(baseRow, worldPoint, -1, distanceScale);
  return worldToPrimitiveLocal(row0, row1, row2, warpedPoint);
}

float sdEllipsoid(vec3 p, vec3 r)
{
  float k0 = length(p / r);
  float k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / k1;
}

float sdBox(vec3 p, vec3 b)
{
  vec3 q = abs(p) - b;
  return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdRoundBox2D(vec2 p, vec2 b, vec4 r)
{
  float rx = (p.x > 0.0) ? r.x : r.z;
  float ry = (p.x > 0.0) ? r.y : r.w;
  float rc = (p.y > 0.0) ? rx : ry;
  vec2 q = abs(p) - b + rc;
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - rc;
}

float sdChamferBox2D(vec2 p, vec2 b, vec4 r)
{
  float rx = (p.x > 0.0) ? r.x : r.z;
  float ry = (p.x > 0.0) ? r.y : r.w;
  float rc = (p.y > 0.0) ? rx : ry;
  vec2 q = abs(p) - b;
  if (rc < 0.001) {
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
  }
  float dCham = (q.x + q.y + rc) * 0.70710678;
  float dBox = max(q.x, q.y);
  float d = max(dBox, dCham);
  if (d <= 0.0) {
    return d;
  }
  if (dBox <= 0.0) {
    return dCham;
  }
  if (q.y <= -rc || q.x <= -rc) {
    return length(max(q, vec2(0.0)));
  }
  float t = (-q.x + q.y + rc) / (2.0 * rc);
  if (t <= 0.0) {
    return length(q - vec2(0.0, -rc));
  }
  if (t >= 1.0) {
    return length(q - vec2(-rc, 0.0));
  }
  return dCham;
}

float sdAdvancedBox(vec3 p, vec3 size, vec4 corners, float edgeTop, float edgeBot, float tapTop, float tapBot, int cornerMode, int edgeMode, float taperZ)
{
  float zn = clamp(p.z / max(taperZ, 0.001), -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  vec2 sz = size.xy * tapFactor;
  float slope = length(size.xy) * (tapTop + tapBot) / (2.0 * max(taperZ, 0.001));
  float lipschitz = sqrt(1.0 + slope * slope);
  float maxR = min(sz.x, sz.y);
  vec4 r = corners * maxR;
  float d2d = (cornerMode == 0) ? sdRoundBox2D(p.xy, sz, r) : sdChamferBox2D(p.xy, sz, r);
  float maxRFace = min(sz.x, sz.y);
  float dz = abs(p.z) - size.z;
  float edgeR = (p.z > 0.0) ? edgeTop * min(maxRFace, size.z) : edgeBot * min(maxRFace, size.z);
  if (edgeR > 0.001) {
    if (edgeMode == 0) {
      vec2 dd = vec2(d2d + edgeR, dz + edgeR);
      return (min(max(dd.x, dd.y), 0.0) + length(max(dd, vec2(0.0))) - edgeR) / lipschitz;
    }
    float base = max(d2d, dz);
    float cham = (d2d + dz + edgeR) * 0.70710678;
    float dd = max(base, cham);
    if (dd <= 0.0) {
      return dd / lipschitz;
    }
    if (d2d <= 0.0 && dz <= 0.0) {
      return cham / lipschitz;
    }
    if (dz <= -edgeR) {
      return d2d / lipschitz;
    }
    if (d2d <= -edgeR) {
      return dz / lipschitz;
    }
    float tc2 = (-d2d + dz + edgeR) / (2.0 * edgeR);
    if (tc2 <= 0.0) {
      return length(vec2(d2d, dz + edgeR)) / lipschitz;
    }
    if (tc2 >= 1.0) {
      return length(vec2(d2d + edgeR, dz)) / lipschitz;
    }
    return cham / lipschitz;
  }
  vec2 dd = vec2(d2d, dz);
  return (length(max(dd, vec2(0.0))) + min(max(dd.x, dd.y), 0.0)) / lipschitz;
}

float sdCylinder(vec3 p, float r, float halfHeight)
{
  vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, halfHeight);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}

float sdAdvancedCylinder(vec3 p, float radius, float halfHeight, float bevelTop, float bevelBottom, float taper, int bevelMode)
{
  float taperH = max(halfHeight, 0.001);
  float zn = clamp(p.z / taperH, -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapTop = max(taper, 0.0);
  float tapBot = max(-taper, 0.0);
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  float scaledRadius = radius * tapFactor;
  float slope = radius * (tapTop + tapBot) / (2.0 * taperH);
  float lipschitz = sqrt(1.0 + slope * slope);
  float d2d = length(p.xy) - scaledRadius;
  float dz = abs(p.z) - halfHeight;
  float edgeBevel = (p.z > 0.0) ? bevelTop : bevelBottom;
  float edgeR = min(max(edgeBevel, 0.0), max(min(scaledRadius, halfHeight) - 0.001, 0.0));
  if (edgeR > 0.001) {
    if (bevelMode == 0) {
      vec2 dd = vec2(d2d + edgeR, dz + edgeR);
      return (min(max(dd.x, dd.y), 0.0) + length(max(dd, vec2(0.0))) - edgeR) / lipschitz;
    }
    float base = max(d2d, dz);
    float cham = (d2d + dz + edgeR) * 0.70710678;
    float dd = max(base, cham);
    if (dd <= 0.0) {
      return dd / lipschitz;
    }
    if (d2d <= 0.0 && dz <= 0.0) {
      return cham / lipschitz;
    }
    if (dz <= -edgeR) {
      return d2d / lipschitz;
    }
    if (d2d <= -edgeR) {
      return dz / lipschitz;
    }
    float tc2 = (-d2d + dz + edgeR) / (2.0 * edgeR);
    if (tc2 <= 0.0) {
      return length(vec2(d2d, dz + edgeR)) / lipschitz;
    }
    if (tc2 >= 1.0) {
      return length(vec2(d2d + edgeR, dz)) / lipschitz;
    }
    return cham / lipschitz;
  }
  vec2 dd = vec2(d2d, dz);
  return (length(max(dd, vec2(0.0))) + min(max(dd.x, dd.y), 0.0)) / lipschitz;
}

float sdTorus(vec3 p, vec2 t)
{
  vec2 q = vec2(length(p.xy) - t.x, p.z);
  return length(q) - t.y;
}

float sdCappedTorus(vec3 p, vec2 sc, float ra, float rb)
{
  p.x = abs(p.x);
  float k = (sc.y * p.x > sc.x * p.y) ? dot(p.xy, sc) : length(p.xy);
  return sqrt(dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
}

float sdCone(vec3 p, float bottomRadius, float topRadius, float halfHeight)
{
  vec2 q = vec2(length(p.xy), p.z);
  vec2 k1 = vec2(topRadius, halfHeight);
  vec2 k2 = vec2(topRadius - bottomRadius, 2.0 * halfHeight);
  vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? bottomRadius : topRadius), abs(q.y) - halfHeight);
  vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / max(dot(k2, k2), 1e-12), 0.0, 1.0);
  float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

float sdAdvancedCone(vec3 p, float bottomRadius, float topRadius, float halfHeight, float bevel, int bevelMode)
{
  float radial = length(p.xy);
  vec2 edge = vec2(topRadius - bottomRadius, 2.0 * halfHeight);
  float edgeLen = max(length(edge), 1e-12);
  float dside = (edge.y * (radial - bottomRadius) - edge.x * (p.z + halfHeight)) / edgeLen;
  float dz = abs(p.z) - halfHeight;
  float edgeR = min(max(bevel, 0.0), max(halfHeight - 0.001, 0.0));
  if (edgeR > 0.001) {
    if (bevelMode == 0) {
      vec2 dd = vec2(dside + edgeR, dz + edgeR);
      return min(max(dd.x, dd.y), 0.0) + length(max(dd, vec2(0.0))) - edgeR;
    }
    float base = max(dside, dz);
    float cham = (dside + dz + edgeR) * 0.70710678;
    float dd = max(base, cham);
    if (dd <= 0.0) {
      return dd;
    }
    if (dside <= 0.0 && dz <= 0.0) {
      return cham;
    }
    if (dz <= -edgeR) {
      return dside;
    }
    if (dside <= -edgeR) {
      return dz;
    }
    float tc2 = (-dside + dz + edgeR) / (2.0 * edgeR);
    if (tc2 <= 0.0) {
      return length(vec2(dside, dz + edgeR));
    }
    if (tc2 >= 1.0) {
      return length(vec2(dside + edgeR, dz));
    }
    return cham;
  }
  return sdCone(p, bottomRadius, topRadius, halfHeight);
}

float sdRoundCone(vec3 p, float bottomRadius, float topRadius, float halfHeight)
{
  vec2 q = vec2(length(p.xy), p.z + halfHeight);
  float totalHeight = max(2.0 * halfHeight, 1e-6);
  if (abs(bottomRadius - topRadius) >= totalHeight) {
    return (bottomRadius >= topRadius) ? (length(q) - bottomRadius) : (length(q - vec2(0.0, totalHeight)) - topRadius);
  }
  float slope = (bottomRadius - topRadius) / totalHeight;
  float axis = sqrt(max(1.0 - slope * slope, 1e-12));
  float k = -slope * q.x + axis * q.y;
  if (k <= 0.0) {
    return length(q) - bottomRadius;
  }
  if (k >= axis * totalHeight) {
    return length(q - vec2(0.0, totalHeight)) - topRadius;
  }
  return axis * q.x + slope * q.y - bottomRadius;
}

vec2 capsuleEndRadii(float radius, float taper)
{
  float tapTop = max(taper, 0.0);
  float tapBot = max(-taper, 0.0);
  return vec2(radius * max(1.0 - tapBot, 0.0), radius * max(1.0 - tapTop, 0.0));
}

float sdCapsule(vec3 p, float r, float h, float taper)
{
  vec2 radii = capsuleEndRadii(r, taper);
  return sdRoundCone(p, radii.x, radii.y, h);
}

float sdRegularPolygon2D(vec2 p, float radius, int sides)
{
  float an = WARP_PI / float(max(sides, 3));
  float innerRadius = radius * cos(an);
  float halfEdge = radius * sin(an);
  p = vec2(-p.y, p.x);
  float bn = an * floor((atan(p.y, p.x) + an) / an / 2.0) * 2.0;
  vec2 cs = vec2(cos(bn), sin(bn));
  p = vec2(cs.x * p.x + cs.y * p.y, -cs.y * p.x + cs.x * p.y);
  return length(p - vec2(innerRadius, clamp(p.y, -halfEdge, halfEdge))) * sign(p.x - innerRadius);
}

float sdStarPolygon2D(vec2 p, float radius, int sides, float star)
{
  if (star < 0.001) {
    return sdRegularPolygon2D(p, radius, sides);
  }
  float an = WARP_PI / float(max(sides, 3));
  float innerRadius = radius * cos(an) * max(1.0 - star, 0.01);
  float angle = atan(p.y, p.x);
  float bn = floor((angle + an) / (2.0 * an)) * 2.0 * an;
  vec2 cs = vec2(cos(bn), sin(bn));
  vec2 q = vec2(cs.x * p.x + cs.y * p.y, -cs.y * p.x + cs.x * p.y);
  q.y = abs(q.y);
  vec2 a = vec2(radius, 0.0);
  vec2 b = vec2(innerRadius * cos(an), innerRadius * sin(an));
  vec2 ab = b - a;
  float t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
  float dist = length(q - (a + ab * t));
  float crossValue = ab.x * (q.y - a.y) - ab.y * (q.x - a.x);
  return (crossValue > 0.0) ? -dist : dist;
}

float sdAdvancedNgon(vec3 p, float radius, float halfHeight, int sides, float corner, float edgeTop, float edgeBot, float tapTop, float tapBot, int edgeMode, float taperH, float star)
{
  float zn = clamp(p.z / max(taperH, 0.001), -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  float scaledR = radius * tapFactor;
  float slope = radius * (tapTop + tapBot) / (2.0 * max(taperH, 0.001));
  float lipschitz = sqrt(1.0 + slope * slope);
  float an = WARP_PI / float(max(sides, 3));
  float apothem = scaledR * cos(an);
  float bevelR = corner * apothem;
  float innerR = scaledR - bevelR / max(cos(an), 0.001);
  float d2d = (star > 0.001) ? sdStarPolygon2D(p.xy, innerR, sides, star) - bevelR : sdRegularPolygon2D(p.xy, innerR, sides) - bevelR;
  float dz = abs(p.z) - halfHeight;
  float edgeR = (p.z > 0.0) ? edgeTop * min(apothem, halfHeight) : edgeBot * min(apothem, halfHeight);
  if (edgeR > 0.001) {
    if (edgeMode == 0) {
      vec2 dd = vec2(d2d + edgeR, dz + edgeR);
      return (min(max(dd.x, dd.y), 0.0) + length(max(dd, vec2(0.0))) - edgeR) / lipschitz;
    }
    float base = max(d2d, dz);
    float cham = (d2d + dz + edgeR) * 0.70710678;
    float dd = max(base, cham);
    if (dd <= 0.0) {
      return dd / lipschitz;
    }
    if (d2d <= 0.0 && dz <= 0.0) {
      return cham / lipschitz;
    }
    if (dz <= -edgeR) {
      return d2d / lipschitz;
    }
    if (d2d <= -edgeR) {
      return dz / lipschitz;
    }
    float tc2 = (-d2d + dz + edgeR) / (2.0 * edgeR);
    if (tc2 <= 0.0) {
      return length(vec2(d2d, dz + edgeR)) / lipschitz;
    }
    if (tc2 >= 1.0) {
      return length(vec2(d2d + edgeR, dz)) / lipschitz;
    }
    return cham / lipschitz;
  }
  vec2 dd = vec2(d2d, dz);
  return (length(max(dd, vec2(0.0))) + min(max(dd.x, dd.y), 0.0)) / lipschitz;
}

float sdPolygon2D(vec2 p, int pointOffset, int pointCount)
{
  if (pointCount < 3) {
    return 1e20;
  }
  float d = 1e20;
  int winding = 0;
  for (int i = 0; i < pointCount; i++) {
    vec2 a = loadPolygonPoint(pointOffset + i);
    vec2 b = loadPolygonPoint(pointOffset + ((i + 1) %% pointCount));
    vec2 edge = b - a;
    vec2 rel = p - a;
    float edgeLenSq = max(dot(edge, edge), 1e-12);
    float t = clamp(dot(rel, edge) / edgeLenSq, 0.0, 1.0);
    d = min(d, length(rel - edge * t));
    float crossValue = edge.x * rel.y - edge.y * rel.x;
    if (a.y <= p.y && b.y > p.y && crossValue > 0.0) {
      winding++;
    }
    if (a.y > p.y && b.y <= p.y && crossValue < 0.0) {
      winding--;
    }
  }
  return (winding != 0) ? -d : d;
}

vec2 cubicBezierPoint2D(vec2 a, vec2 b, vec2 c, vec2 d, float t)
{
  float invT = 1.0 - t;
  float invT2 = invT * invT;
  float t2 = t * t;
  return a * (invT2 * invT) + b * (3.0 * invT2 * t) + c * (3.0 * invT * t2) + d * (t2 * t);
}

float sdBezierPolygon2D(vec2 p, int pointOffset, int pointCount)
{
  if (pointCount < 3) {
    return 1e20;
  }
  const int BEZIER_STEPS = 8;
  float d = 1e20;
  int winding = 0;
  for (int i = 0; i < pointCount; i++) {
    int j = (i + 1) %% pointCount;
    vec2 p0 = loadBezierPoint(pointOffset, i);
    vec2 p1 = loadBezierHandleRight(pointOffset, i);
    vec2 p2 = loadBezierHandleLeft(pointOffset, j);
    vec2 p3 = loadBezierPoint(pointOffset, j);
    vec2 a = p0;
    for (int step = 1; step <= BEZIER_STEPS; step++) {
      float t = float(step) / float(BEZIER_STEPS);
      vec2 b = cubicBezierPoint2D(p0, p1, p2, p3, t);
      vec2 edge = b - a;
      vec2 rel = p - a;
      float edgeLenSq = max(dot(edge, edge), 1e-12);
      float s = clamp(dot(rel, edge) / edgeLenSq, 0.0, 1.0);
      d = min(d, length(rel - edge * s));
      float crossValue = edge.x * rel.y - edge.y * rel.x;
      if (a.y <= p.y && b.y > p.y && crossValue > 0.0) {
        winding++;
      }
      if (a.y > p.y && b.y <= p.y && crossValue < 0.0) {
        winding--;
      }
      a = b;
    }
  }
  return (winding != 0) ? -d : d;
}

float sdPolygonInterpolated2D(vec2 p, int pointOffset, int pointCount, int interpolationMode)
{
  return (interpolationMode != 0)
    ? sdBezierPolygon2D(p, pointOffset, pointCount)
    : sdPolygon2D(p, pointOffset, pointCount);
}

float sdPrism(float d2d, float zValue, float halfHeight)
{
  vec2 d = vec2(d2d, abs(zValue) - halfHeight);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}

float sdAdvancedPolygon(vec3 p, float halfHeight, int pointOffset, int pointCount, float edgeTop, float edgeBot, float tapTop, float tapBot, int interpolationMode, int edgeMode, float taperH)
{
  float zn = clamp(p.z / max(taperH, 0.001), -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  float slope = (tapTop + tapBot) / (2.0 * max(taperH, 0.001));
  float lipschitz = sqrt(1.0 + slope * slope);
  float d2d = sdPolygonInterpolated2D(p.xy / tapFactor, pointOffset, pointCount, interpolationMode) * tapFactor;
  float dz = abs(p.z) - halfHeight;
  float edgeR = (p.z > 0.0) ? edgeTop * halfHeight : edgeBot * halfHeight;
  if (edgeR > 0.001) {
    if (edgeMode == 0) {
      vec2 dd = vec2(d2d + edgeR, dz + edgeR);
      return (min(max(dd.x, dd.y), 0.0) + length(max(dd, vec2(0.0))) - edgeR) / lipschitz;
    }
    float base = max(d2d, dz);
    float cham = (d2d + dz + edgeR) * 0.70710678;
    float dd = max(base, cham);
    if (dd <= 0.0) {
      return dd / lipschitz;
    }
    if (d2d <= 0.0 && dz <= 0.0) {
      return cham / lipschitz;
    }
    if (dz <= -edgeR) {
      return d2d / lipschitz;
    }
    if (d2d <= -edgeR) {
      return dz / lipschitz;
    }
    float tc2 = (-d2d + dz + edgeR) / (2.0 * edgeR);
    if (tc2 <= 0.0) {
      return length(vec2(d2d, dz + edgeR)) / lipschitz;
    }
    if (tc2 >= 1.0) {
      return length(vec2(d2d + edgeR, dz)) / lipschitz;
    }
    return cham / lipschitz;
  }
  vec2 dd = vec2(d2d, dz);
  return (length(max(dd, vec2(0.0))) + min(max(dd.x, dd.y), 0.0)) / lipschitz;
}

float kernel(float x, float k)
{
  if (k <= 0.0) {
    return 0.0;
  }
  float m = max(0.0, k - x);
  return m * m * 0.25 / k;
}

float opSign(int op)
{
  return (op == 1) ? 1.0 : -1.0;
}

float applySignedOp(int op, float a, float b, float blend)
{
  float s = opSign(op);
  return s * (min(s * a, s * b) - kernel(abs(a - b), blend));
}

float opSmoothUnion(float a, float b, float k)
{
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float opSmoothSubtract(float a, float b, float k)
{
  float h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
  return mix(a, -b, h) + k * h * (1.0 - h);
}

float opSmoothIntersect(float a, float b, float k)
{
  float h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) + k * h * (1.0 - h);
}

float applyOp(int op, float a, float b, float blend)
{
  if (op == 1) {
    if (blend <= 1e-6) {
      return min(a, b);
    }
    return opSmoothUnion(a, b, blend);
  }
  if (op == 2) {
    if (blend <= 1e-6) {
      return max(a, -b);
    }
    return opSmoothSubtract(a, b, blend);
  }
  if (op == 3) {
    if (blend <= 1e-6) {
      return max(a, b);
    }
    return opSmoothIntersect(a, b, blend);
  }
  return b;
}

int activeNodeIndex(float encoded)
{
  return int(abs(encoded) + 0.5) - 1;
}

bool activeNodeSign(float encoded)
{
  return encoded >= 0.0;
}

float evalPrimitive(int primitiveIndex, vec3 worldPoint);

float evalPrimitiveLocalDistance(int primitiveType,
                                 vec4 meta,
                                 vec4 extra0,
                                 vec4 extra1,
                                 vec4 extra2,
                                 vec3 localPoint,
                                 vec3 scale,
                                 float minScale)
{
  float distanceValue = 1e20;
  float bevel = max(extra0.x, 0.0);
  float postBevel = bevel;
  if (primitiveType == 0) {
    distanceValue = sdEllipsoid(localPoint, vec3(meta.y) * scale);
  }
  else if (primitiveType == 1) {
    float tapTop = max(extra1.w, 0.0);
    float tapBot = max(-extra1.w, 0.0);
    if (extra0.y + extra0.z + extra0.w + extra1.x + extra1.y + extra1.z + tapTop + tapBot > 0.001) {
      distanceValue = sdAdvancedBox(localPoint, meta.yzw * scale, vec4(extra0.y, extra0.z, extra0.w, extra1.x), extra1.y, extra1.z, tapTop, tapBot, int(extra2.x + 0.5), int(extra2.y + 0.5), meta.w * scale.z);
    }
    else {
      distanceValue = sdBox(localPoint, meta.yzw * scale);
    }
  }
  else if (primitiveType == 2) {
    vec3 scaledPoint = localPoint / scale;
    float bevelTop = max(extra0.x, 0.0);
    float bevelBottom = max(extra0.y, 0.0);
    distanceValue = ((max(bevelTop, bevelBottom) + abs(extra0.z)) > 0.001)
      ? sdAdvancedCylinder(scaledPoint, meta.y, meta.z, bevelTop, bevelBottom, extra0.z, int(extra0.w + 0.5)) * minScale
      : sdCylinder(scaledPoint, meta.y, meta.z) * minScale;
    postBevel = 0.0;
  }
  else if (primitiveType == 3) {
    float angle = clamp(extra0.y, 0.0, 6.283185307179586);
    distanceValue = ((6.283185307179586 - angle) > 0.001)
      ? sdCappedTorus(localPoint / scale, vec2(sin(0.5 * angle), cos(0.5 * angle)), meta.y, meta.z) * minScale
      : sdTorus(localPoint / scale, vec2(meta.y, meta.z)) * minScale;
    postBevel = 0.0;
  }
  else if (primitiveType == 4) {
    vec3 scaledPoint = localPoint / scale;
    distanceValue = (bevel > 0.001)
      ? sdAdvancedCone(scaledPoint, meta.y, meta.z, meta.w, bevel, int(extra0.y + 0.5)) * minScale
      : sdCone(scaledPoint, meta.y, meta.z, meta.w) * minScale;
    postBevel = 0.0;
  }
  else if (primitiveType == 5) {
    distanceValue = sdCapsule(localPoint / scale, meta.y, meta.z, extra0.y) * minScale;
    postBevel = 0.0;
  }
  else if (primitiveType == 6) {
    vec3 scaledPoint = localPoint / scale;
    float tapTop = max(extra1.x, 0.0);
    float tapBot = max(-extra1.x, 0.0);
    if (extra0.y + extra0.z + extra0.w + tapTop + tapBot + extra1.z > 0.001) {
      distanceValue = sdAdvancedNgon(scaledPoint, meta.y, meta.z, int(meta.w + 0.5), extra0.y, extra0.z, extra0.w, tapTop, tapBot, int(extra1.y + 0.5), meta.z, extra1.z) * minScale;
    }
    else {
      distanceValue = sdPrism(sdRegularPolygon2D(scaledPoint.xy, meta.y, int(meta.w + 0.5)), scaledPoint.z, meta.z) * minScale;
    }
    postBevel = 0.0;
  }
  else if (primitiveType == 7) {
    vec3 scaledPoint = localPoint / scale;
    float tapTop = max(extra0.w, 0.0);
    float tapBot = max(-extra0.w, 0.0);
    if (extra0.y + extra0.z + tapTop + tapBot > 0.001) {
      distanceValue = sdAdvancedPolygon(scaledPoint, meta.w, int(meta.y + 0.5), int(meta.z + 0.5), extra0.y, extra0.z, tapTop, tapBot, int(extra1.w + 0.5), int(extra1.x + 0.5), meta.w) * minScale;
    }
    else {
      distanceValue = sdPrism(sdPolygonInterpolated2D(scaledPoint.xy, int(meta.y + 0.5), int(meta.z + 0.5), int(extra1.w + 0.5)), scaledPoint.z, meta.w) * minScale;
    }
  }
  return distanceValue - postBevel;
}

vec4 normalizeDistanceGradient(float distanceValue, vec3 gradient)
{
  float lenSq = dot(gradient, gradient);
  if (lenSq > 1e-12) {
    gradient *= inversesqrt(lenSq);
  }
  else {
    gradient = vec3(0.0);
  }
  return vec4(distanceValue, gradient);
}

vec4 negateDistanceGradient(vec4 value)
{
  return vec4(-value.x, -value.yzw);
}

vec4 sdgEllipsoid(vec3 p, vec3 r)
{
  vec3 pr = p / r;
  float k0 = length(pr);
  vec3 pr2 = p / (r * r);
  float k1 = length(pr2);
  float dist = k0 * (k0 - 1.0) / max(k1, 1e-8);
  vec3 grad = pr2 / max(k1, 1e-8);
  return vec4(dist, grad);
}

vec4 sdgBox(vec3 p, vec3 b)
{
  vec3 w = abs(p) - b;
  vec3 s = sign(p);
  float g = max(w.x, max(w.y, w.z));
  vec3 q = max(w, vec3(0.0));
  float l = length(q);
  vec4 f = (g > 0.0)
      ? vec4(l, q / max(l, 1e-8))
      : vec4(g,
             vec3(
               (w.x == g) ? 1.0 : 0.0,
               (w.y == g) ? 1.0 : 0.0,
               (w.z == g) ? 1.0 : 0.0));
  return vec4(f.x, f.yzw * s);
}

vec4 sdgCylinder(vec3 p, vec3 size)
{
  vec2 e = max(size.xy, vec2(0.001));
  vec2 pn = p.xy / e;
  float rn = max(length(pn), 1e-6);
  vec2 g = pn / (e * rn);
  float gl = max(length(g), 1e-6);
  float radial = (rn - 1.0) / gl;
  float vertical = abs(p.z) - size.z;

  vec3 du = vec3(g / gl, 0.0);
  vec3 dv = vec3(0.0, 0.0, sign(p.z));

  vec2 dd = vec2(radial, vertical);
  vec2 hh = max(dd, vec2(0.0));
  float fl = length(hh);
  float fg = max(dd.x, dd.y);

  if (fg <= 0.0) {
    return vec4(fg, (dd.x > dd.y) ? du : dv);
  }
  return vec4(fl, (hh.x * du + hh.y * dv) / max(fl, 1e-8));
}

vec4 sdgCone(vec3 p, float r, float h)
{
  vec2 k = vec2(-r, 2.0 * h);
  float m = dot(k, k);
  float l = length(p.xy);
  vec2 q = vec2(-l, h - p.z);
  vec2 a = vec2(l - min(l, (p.z < 0.0) ? r : 0.0), abs(p.z) - h);
  float tRaw = dot(q, k) / m;
  vec2 b = k * clamp(tRaw, 0.0, 1.0) - q;
  float s = (b.x < 0.0 && a.y < 0.0) ? -1.0 : 1.0;
  float la = dot(a, a);
  float lb = dot(b, b);
  float dist = s * sqrt(min(la, lb));

  vec2 radial = p.xy / max(l, 1e-8);
  vec2 closest;
  if (la < lb) {
    float rCap = (p.z < 0.0) ? r : 0.0;
    closest = vec2(min(l, rCap), sign(p.z) * h);
  }
  else {
    float tc = clamp(tRaw, 0.0, 1.0);
    closest = vec2(tc * r, h - 2.0 * tc * h);
  }

  vec2 delta = vec2(l, p.z) - closest;
  float dl2 = length(delta);
  vec3 grad;
  if (dl2 > 1e-6) {
    grad = s * vec3(radial * delta.x, delta.y) / dl2;
  }
  else {
    grad = s * vec3(radial * k.y, -k.x) / sqrt(m);
  }
  return vec4(dist, grad);
}

vec4 sdgCapsule(vec3 p, float r, float h)
{
  vec3 ba = vec3(0.0, 0.0, 2.0 * h);
  vec3 pa = vec3(p.x, p.y, p.z + h);
  float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  vec3 q = pa - t * ba;
  float d = length(q);
  return vec4(d - r, q / max(d, 1e-8));
}

vec4 sdgTorus(vec3 p, vec2 t)
{
  float h = length(p.xy);
  vec3 raw = p * vec3(h - t.x, h - t.x, h);
  float rl = length(raw);
  vec3 grad = (rl > 1e-8) ? raw / rl : vec3(0.0, 0.0, sign(p.z));
  vec2 q = vec2(h - t.x, p.z);
  float d = length(q) - t.y;
  return vec4(d, grad);
}

vec4 sdgCappedTorus(vec3 p, vec2 sc, float ra, float rb)
{
  vec3 ap = vec3(abs(p.x), p.y, p.z);
  float k = (sc.y * ap.x > sc.x * ap.y) ? dot(ap.xy, sc) : length(ap.xy);
  float d = sqrt(dot(p, p) + ra * ra - 2.0 * ra * k) - rb;

  vec3 grad;
  if (sc.y * ap.x > sc.x * ap.y) {
    grad = vec3(p.x - sign(p.x) * ra * sc.x,
                p.y - ra * sc.y,
                p.z);
  }
  else {
    float lxy = max(length(p.xy), 1e-8);
    grad = vec3(p.x * (1.0 - ra / lxy),
                p.y * (1.0 - ra / lxy),
                p.z);
  }
  return vec4(d, grad / max(length(grad), 1e-8));
}

vec4 makeDistanceGradient2D(float distanceValue, vec2 gradient)
{
  return vec4(distanceValue, gradient.x, gradient.y, 0.0);
}

float sabsDerivative(float x, float k)
{
  if (k <= 0.0001) {
    return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  }
  float ax = abs(x);
  if (ax >= k) {
    return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  }
  float t = ax / k;
  float deriv = 3.0 * t - 3.0 * t * t + t * t * t;
  return ((x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0)) * deriv;
}

float foldFiniteGridAxisDerivative(float axisValue, float origin, float primitiveCenter, float spacing, int count, float blend)
{
  if (count <= 1 || spacing <= 1e-6) {
    return 1.0;
  }
  float base = (primitiveCenter - origin) / spacing;
  float centerOffset = 0.5 * float(count - 1);
  float shifted = (axisValue - origin) / spacing - base + centerOffset;
  float id = clamp(round(shifted), 0.0, float(count - 1));
  float local = shifted - id;
  float dR = (0.5 - local) * spacing;
  float dL = (0.5 + local) * spacing;
  float pullRDeriv = 0.0;
  float pullLDeriv = 0.0;
  if (id <= float(count) - 1.5) {
    pullRDeriv = -(1.0 - sabsDerivative(dR, blend));
  }
  if (id >= 0.5) {
    pullLDeriv = 1.0 - sabsDerivative(dL, blend);
  }
  return 1.0 + pullRDeriv - pullLDeriv;
}

vec3 restoreRotationScaleGradient(vec3 gradient, vec4 rotation, vec3 scale)
{
  return rotateByQuaternion(rotation, gradient / safeArrayScale(scale));
}

vec4 sdgBox2D(vec2 p, vec2 b)
{
  vec2 w = abs(p) - b;
  vec2 s = sign(p);
  float g = max(w.x, w.y);
  vec2 q = max(w, vec2(0.0));
  float l = length(q);
  vec2 grad = (g > 0.0)
    ? q / max(l, 1e-8)
    : vec2((w.x == g) ? 1.0 : 0.0, (w.y == g) ? 1.0 : 0.0);
  return makeDistanceGradient2D((g > 0.0) ? l : g, grad * s);
}

vec4 sdgRoundBox2D(vec2 p, vec2 b, vec4 r)
{
  float rx = (p.x > 0.0) ? r.x : r.z;
  float ry = (p.x > 0.0) ? r.y : r.w;
  float rc = (p.y > 0.0) ? rx : ry;
  vec4 dg = sdgBox2D(p, b - vec2(rc));
  dg.x -= rc;
  return dg;
}

vec4 sdgChamferBox2D(vec2 p, vec2 b, vec4 r)
{
  float rx = (p.x > 0.0) ? r.x : r.z;
  float ry = (p.x > 0.0) ? r.y : r.w;
  float rc = (p.y > 0.0) ? rx : ry;
  if (rc < 0.001) {
    return sdgBox2D(p, b);
  }
  vec2 s = sign(p);
  vec2 q = abs(p) - b;
  float dCham = (q.x + q.y + rc) * 0.70710678;
  vec2 gCham = vec2(0.70710678) * s;
  float dBox = max(q.x, q.y);
  float d = max(dBox, dCham);
  if (d <= 0.0) {
    if (dBox > dCham) {
      vec2 gBox = vec2((q.x >= q.y) ? 1.0 : 0.0, (q.y > q.x) ? 1.0 : 0.0) * s;
      return makeDistanceGradient2D(dBox, gBox);
    }
    return makeDistanceGradient2D(dCham, gCham);
  }
  if (dBox <= 0.0) {
    return makeDistanceGradient2D(dCham, gCham);
  }
  if (q.y <= -rc || q.x <= -rc) {
    vec2 qq = max(q, vec2(0.0));
    float l = length(qq);
    vec2 grad = (l > 1e-8) ? qq / l : vec2(0.0);
    return makeDistanceGradient2D(l, grad * s);
  }
  float t = (-q.x + q.y + rc) / (2.0 * rc);
  if (t <= 0.0) {
    vec2 delta = q - vec2(0.0, -rc);
    float l = length(delta);
    vec2 grad = (l > 1e-8) ? delta / l : vec2(1.0, 0.0);
    return makeDistanceGradient2D(l, grad * s);
  }
  if (t >= 1.0) {
    vec2 delta = q - vec2(-rc, 0.0);
    float l = length(delta);
    vec2 grad = (l > 1e-8) ? delta / l : vec2(0.0, 1.0);
    return makeDistanceGradient2D(l, grad * s);
  }
  return makeDistanceGradient2D(dCham, gCham);
}

vec4 sdgRegularPolygon2D(vec2 p, float radius, int sides)
{
  float an = WARP_PI / float(max(sides, 3));
  float innerRadius = radius * cos(an);
  float halfEdge = radius * sin(an);
  vec2 u = vec2(-p.y, p.x);
  float bn = an * floor((atan(u.y, u.x) + an) / an / 2.0) * 2.0;
  vec2 cs = vec2(cos(bn), sin(bn));
  vec2 q = vec2(cs.x * u.x + cs.y * u.y, -cs.y * u.x + cs.x * u.y);
  vec2 closest = vec2(innerRadius, clamp(q.y, -halfEdge, halfEdge));
  vec2 delta = q - closest;
  float l = length(delta);
  float signValue = (q.x >= innerRadius) ? 1.0 : -1.0;
  vec2 gradQ = (l > 1e-8) ? (signValue * delta / l) : vec2(signValue, 0.0);
  vec2 gradU = vec2(cs.x * gradQ.x - cs.y * gradQ.y, cs.y * gradQ.x + cs.x * gradQ.y);
  vec2 gradP = vec2(gradU.y, -gradU.x);
  return makeDistanceGradient2D(l * signValue, gradP);
}

vec4 sdgStarPolygon2D(vec2 p, float radius, int sides, float star)
{
  if (star < 0.001) {
    return sdgRegularPolygon2D(p, radius, sides);
  }
  float an = WARP_PI / float(max(sides, 3));
  float innerRadius = radius * cos(an) * max(1.0 - star, 0.01);
  float angle = atan(p.y, p.x);
  float bn = floor((angle + an) / (2.0 * an)) * 2.0 * an;
  vec2 cs = vec2(cos(bn), sin(bn));
  vec2 q0 = vec2(cs.x * p.x + cs.y * p.y, -cs.y * p.x + cs.x * p.y);
  vec2 q = vec2(q0.x, abs(q0.y));
  vec2 a = vec2(radius, 0.0);
  vec2 b = vec2(innerRadius * cos(an), innerRadius * sin(an));
  vec2 ab = b - a;
  float t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
  vec2 closest = a + ab * t;
  vec2 delta = q - closest;
  float l = length(delta);
  float signValue = (ab.x * (q.y - a.y) - ab.y * (q.x - a.x) > 0.0) ? -1.0 : 1.0;
  vec2 gradQ = (l > 1e-8) ? (signValue * delta / l) : vec2(signValue, 0.0);
  vec2 gradQ0 = vec2(gradQ.x, gradQ.y * ((q0.y >= 0.0) ? 1.0 : -1.0));
  vec2 gradP = vec2(cs.x * gradQ0.x - cs.y * gradQ0.y, cs.y * gradQ0.x + cs.x * gradQ0.y);
  return makeDistanceGradient2D(signValue * l, gradP);
}

vec4 sdgPolygon2D(vec2 p, int pointOffset, int pointCount)
{
  if (pointCount < 3) {
    return vec4(1e20, 0.0, 0.0, 0.0);
  }
  float bestLenSq = 1e20;
  vec2 bestDelta = vec2(1.0, 0.0);
  int winding = 0;
  for (int i = 0; i < pointCount; i++) {
    vec2 a = loadPolygonPoint(pointOffset + i);
    vec2 b = loadPolygonPoint(pointOffset + ((i + 1) %% pointCount));
    vec2 edge = b - a;
    vec2 rel = p - a;
    float edgeLenSq = max(dot(edge, edge), 1e-12);
    float t = clamp(dot(rel, edge) / edgeLenSq, 0.0, 1.0);
    vec2 delta = rel - edge * t;
    float lenSq = dot(delta, delta);
    if (lenSq < bestLenSq) {
      bestLenSq = lenSq;
      bestDelta = delta;
    }
    float crossValue = edge.x * rel.y - edge.y * rel.x;
    if (a.y <= p.y && b.y > p.y && crossValue > 0.0) {
      winding++;
    }
    if (a.y > p.y && b.y <= p.y && crossValue < 0.0) {
      winding--;
    }
  }
  float l = sqrt(bestLenSq);
  float signValue = (winding != 0) ? -1.0 : 1.0;
  vec2 grad = (l > 1e-8) ? (signValue * bestDelta / l) : vec2(signValue, 0.0);
  return makeDistanceGradient2D(signValue * l, grad);
}

vec4 sdgExtrude2DGrad(vec4 dg2d, float pz, float halfH)
{
  float dz = abs(pz) - halfH;
  vec2 dd = vec2(dg2d.x, dz);
  vec2 hh = max(dd, vec2(0.0));
  float fl = length(hh);
  float fg = max(dd.x, dd.y);
  vec3 gz = vec3(0.0, 0.0, sign(pz));
  if (fg <= 0.0) {
    return (dd.x > dd.y) ? dg2d : vec4(fg, gz);
  }
  return vec4(fl, (hh.x * dg2d.yzw + hh.y * gz) / max(fl, 1e-8));
}

vec4 sdgRoundExtrude2DGrad(vec4 dg2d, float pz, float halfH, float edgeR, vec3 gEdgeR)
{
  float dz = abs(pz) - halfH;
  vec3 gz = vec3(0.0, 0.0, sign(pz));
  vec2 dd = vec2(dg2d.x + edgeR, dz + edgeR);
  if (max(dd.x, dd.y) <= 0.0) {
    return (dd.x > dd.y) ? dg2d : vec4(dz, gz);
  }
  vec2 hh = max(dd, vec2(0.0));
  float fl = max(length(hh), 1e-8);
  float cx = hh.x / fl;
  float cy = hh.y / fl;
  float cr = cx + cy - 1.0;
  return vec4(fl - edgeR, cx * dg2d.yzw + cy * gz + cr * gEdgeR);
}

vec4 sdgChamferExtrude2DGrad(vec4 dg2d, float pz, float halfH, float edgeR, vec3 gEdgeR)
{
  float dz = abs(pz) - halfH;
  vec3 gz = vec3(0.0, 0.0, sign(pz));
  float base = max(dg2d.x, dz);
  float cham = (dg2d.x + dz + edgeR) * 0.70710678;
  vec4 dgCham = vec4(cham, (dg2d.yzw + gz + gEdgeR) * 0.70710678);
  float dd = max(base, cham);
  if (dd <= 0.0) {
    if (base > cham) {
      return (dg2d.x > dz) ? dg2d : vec4(dz, gz);
    }
    return dgCham;
  }
  if (dg2d.x <= 0.0 && dz <= 0.0) {
    return dgCham;
  }
  if (dz <= -edgeR) {
    return dg2d;
  }
  if (dg2d.x <= -edgeR) {
    return vec4(dz, gz);
  }
  float tc2 = (-dg2d.x + dz + edgeR) / (2.0 * edgeR);
  if (tc2 <= 0.0) {
    vec2 q = vec2(dg2d.x, dz + edgeR);
    float l = max(length(q), 1e-8);
    return vec4(l, (q.x * dg2d.yzw + q.y * (gz + gEdgeR)) / l);
  }
  if (tc2 >= 1.0) {
    vec2 q = vec2(dg2d.x + edgeR, dz);
    float l = max(length(q), 1e-8);
    return vec4(l, (q.x * (dg2d.yzw + gEdgeR) + q.y * gz) / l);
  }
  return dgCham;
}

float taperFactorDerivative(float pz, float taperH, float tapTop, float tapBot)
{
  if (taperH <= 0.001 || abs(pz) >= taperH) {
    return 0.0;
  }
  return -(tapTop - tapBot) / (2.0 * taperH);
}

vec4 sdgAdvancedBox(vec3 p, vec3 size, vec4 corners, float edgeTop, float edgeBot, float tapTop, float tapBot, int cornerMode, int edgeMode, float taperZ)
{
  float taperH = max(taperZ, 0.001);
  float zn = clamp(p.z / taperH, -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  float tapDeriv = taperFactorDerivative(p.z, taperH, tapTop, tapBot);
  float lipschitz = sqrt(1.0 + pow(length(size.xy) * (tapTop + tapBot) / (2.0 * taperH), 2.0));
  float baseMaxR = min(size.x, size.y);
  vec2 u = p.xy / tapFactor;
  vec4 base2d = (cornerMode == 0)
    ? sdgRoundBox2D(u, size.xy, corners * baseMaxR)
    : sdgChamferBox2D(u, size.xy, corners * baseMaxR);
  float baseValue = base2d.x;
  vec4 dg2d = vec4(tapFactor * baseValue, base2d.y, base2d.z, tapDeriv * (baseValue - dot(base2d.yz, u)));
  float faceExtent = baseMaxR * tapFactor;
  float edgeScale = (faceExtent <= size.z) ? 1.0 : 0.0;
  float edgeCoeff = (p.z > 0.0) ? edgeTop : edgeBot;
  float edgeR = edgeCoeff * min(faceExtent, size.z);
  vec3 gEdgeR = vec3(0.0, 0.0, edgeCoeff * baseMaxR * tapDeriv * edgeScale);
  vec4 dg = (edgeR > 0.001)
    ? ((edgeMode == 0) ? sdgRoundExtrude2DGrad(dg2d, p.z, size.z, edgeR, gEdgeR)
                       : sdgChamferExtrude2DGrad(dg2d, p.z, size.z, edgeR, gEdgeR))
    : sdgExtrude2DGrad(dg2d, p.z, size.z);
  return vec4(dg.x / lipschitz, dg.yzw / lipschitz);
}

vec4 sdgAdvancedCylinder(vec3 p, float radius, float halfHeight, float bevelTop, float bevelBottom, float taper, int bevelMode)
{
  float taperH = max(halfHeight, 0.001);
  float tapTop = max(taper, 0.0);
  float tapBot = max(-taper, 0.0);
  float zn = clamp(p.z / taperH, -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  float tapDeriv = taperFactorDerivative(p.z, taperH, tapTop, tapBot);
  float lipschitz = sqrt(1.0 + pow(radius * (tapTop + tapBot) / (2.0 * taperH), 2.0));
  vec2 u = p.xy / tapFactor;
  float ul = length(u);
  float baseValue = ul - radius;
  vec2 baseGrad = (ul > 1e-8) ? (u / ul) : vec2(0.0);
  vec4 dg2d = vec4(tapFactor * baseValue, baseGrad.x, baseGrad.y, tapDeriv * (baseValue - dot(baseGrad, u)));
  float edgeBevel = (p.z > 0.0) ? bevelTop : bevelBottom;
  float edgeR = min(max(edgeBevel, 0.0), max(min(radius * tapFactor, halfHeight) - 0.001, 0.0));
  vec4 dg = (edgeR > 0.001)
    ? ((bevelMode == 0) ? sdgRoundExtrude2DGrad(dg2d, p.z, halfHeight, edgeR, vec3(0.0))
                        : sdgChamferExtrude2DGrad(dg2d, p.z, halfHeight, edgeR, vec3(0.0)))
    : sdgExtrude2DGrad(dg2d, p.z, halfHeight);
  return vec4(dg.x / lipschitz, dg.yzw / lipschitz);
}

vec4 sdgAdvancedNgon(vec3 p, float radius, float halfHeight, int sides, float corner, float edgeTop, float edgeBot, float tapTop, float tapBot, int edgeMode, float taperH, float star)
{
  float taper = max(taperH, 0.001);
  float zn = clamp(p.z / taper, -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  float tapDeriv = taperFactorDerivative(p.z, taper, tapTop, tapBot);
  float lipschitz = sqrt(1.0 + pow(radius * (tapTop + tapBot) / (2.0 * taper), 2.0));
  float an = WARP_PI / float(max(sides, 3));
  float apothemBase = radius * cos(an);
  float bevelBase = corner * apothemBase;
  float innerRadiusBase = radius * max(1.0 - corner, 0.0);
  vec2 u = p.xy / tapFactor;
  vec4 base2d = (star > 0.001)
    ? sdgStarPolygon2D(u, innerRadiusBase, sides, star)
    : sdgRegularPolygon2D(u, innerRadiusBase, sides);
  float baseValue = base2d.x - bevelBase;
  vec4 dg2d = vec4(tapFactor * baseValue, base2d.y, base2d.z, tapDeriv * (baseValue - dot(base2d.yz, u)));
  float faceExtent = apothemBase * tapFactor;
  float edgeScale = (faceExtent <= halfHeight) ? 1.0 : 0.0;
  float edgeCoeff = (p.z > 0.0) ? edgeTop : edgeBot;
  float edgeR = edgeCoeff * min(faceExtent, halfHeight);
  vec3 gEdgeR = vec3(0.0, 0.0, edgeCoeff * apothemBase * tapDeriv * edgeScale);
  vec4 dg = (edgeR > 0.001)
    ? ((edgeMode == 0) ? sdgRoundExtrude2DGrad(dg2d, p.z, halfHeight, edgeR, gEdgeR)
                       : sdgChamferExtrude2DGrad(dg2d, p.z, halfHeight, edgeR, gEdgeR))
    : sdgExtrude2DGrad(dg2d, p.z, halfHeight);
  return vec4(dg.x / lipschitz, dg.yzw / lipschitz);
}

vec4 sdgAdvancedPolygon(vec3 p, float halfHeight, int pointOffset, int pointCount, float edgeTop, float edgeBot, float tapTop, float tapBot, int edgeMode, float taperH)
{
  float taper = max(taperH, 0.001);
  float zn = clamp(p.z / taper, -1.0, 1.0);
  float t = (zn + 1.0) * 0.5;
  float tapFactor = max(1.0 - tapTop * t - tapBot * (1.0 - t), 0.001);
  float tapDeriv = taperFactorDerivative(p.z, taper, tapTop, tapBot);
  float lipschitz = sqrt(1.0 + pow((tapTop + tapBot) / (2.0 * taper), 2.0));
  vec2 u = p.xy / tapFactor;
  vec4 base2d = sdgPolygon2D(u, pointOffset, pointCount);
  float baseValue = base2d.x;
  vec4 dg2d = vec4(tapFactor * baseValue, base2d.y, base2d.z, tapDeriv * (baseValue - dot(base2d.yz, u)));
  float edgeR = ((p.z > 0.0) ? edgeTop : edgeBot) * halfHeight;
  vec4 dg = (edgeR > 0.001)
    ? ((edgeMode == 0) ? sdgRoundExtrude2DGrad(dg2d, p.z, halfHeight, edgeR, vec3(0.0))
                       : sdgChamferExtrude2DGrad(dg2d, p.z, halfHeight, edgeR, vec3(0.0)))
    : sdgExtrude2DGrad(dg2d, p.z, halfHeight);
  return vec4(dg.x / lipschitz, dg.yzw / lipschitz);
}

vec3 primitiveGradientToWorld(int baseRow, vec3 localGradient)
{
  vec4 row0 = loadRow(baseRow + 1);
  vec4 row1 = loadRow(baseRow + 2);
  vec4 row2 = loadRow(baseRow + 3);
  return row0.xyz * localGradient.x + row1.xyz * localGradient.y + row2.xyz * localGradient.z;
}

vec3 invertMirrorWarpGradient(vec3 gradient, vec3 point, int flags, float blend, vec3 origin)
{
  if ((flags & MIRROR_AXIS_X) != 0) {
    float side = ((flags & MIRROR_SIDE_X) != 0) ? 1.0 : -1.0;
    gradient.x *= sabsDerivative(side * (point.x - origin.x), blend);
  }
  if ((flags & MIRROR_AXIS_Y) != 0) {
    float side = ((flags & MIRROR_SIDE_Y) != 0) ? 1.0 : -1.0;
    gradient.y *= sabsDerivative(side * (point.y - origin.y), blend);
  }
  if ((flags & MIRROR_AXIS_Z) != 0) {
    float side = ((flags & MIRROR_SIDE_Z) != 0) ? 1.0 : -1.0;
    gradient.z *= sabsDerivative(side * (point.z - origin.z), blend);
  }
  return gradient;
}

vec3 invertGridWarpGradient(vec3 gradient, vec3 point, ivec3 counts, vec3 spacing, float blend, vec3 origin, vec4 rotation, vec3 scale, vec3 branchCenter, vec3 offsetLocation, vec4 offsetRotation, vec3 offsetScale)
{
  vec3 arrayPoint = worldToArrayLocal(point, origin, rotation, scale, branchCenter);
  vec3 offsetCenter = branchCenter + offsetLocation;
  vec3 gradArray = restoreRotationScaleGradient(gradient, offsetRotation, offsetScale);
  gradArray.x *= foldFiniteGridAxisDerivative(arrayPoint.x, offsetCenter.x, offsetCenter.x, spacing.x, counts.x, blend);
  gradArray.y *= foldFiniteGridAxisDerivative(arrayPoint.y, offsetCenter.y, offsetCenter.y, spacing.y, counts.y, blend);
  gradArray.z *= foldFiniteGridAxisDerivative(arrayPoint.z, offsetCenter.z, offsetCenter.z, spacing.z, counts.z, blend);
  return restoreRotationScaleGradient(gradArray, rotation, scale);
}

vec3 invertRadialWarpGradient(vec3 gradient, vec3 point, float radius, int count, float blend, vec3 repeatOrigin, vec3 fieldOrigin, vec4 rotation, vec3 scale, vec3 branchCenter, vec3 offsetLocation, vec4 offsetRotation, vec3 offsetScale)
{
  vec3 arrayPoint = worldToArrayLocal(point, fieldOrigin, rotation, scale, branchCenter);
  vec3 gradArray = restoreRotationScaleGradient(gradient, offsetRotation, offsetScale);
  if (count > 1 && radius > 1e-6) {
    vec3 q = arrayPoint - repeatOrigin;
    vec3 primitiveCenter = branchCenter + offsetLocation;
    vec3 baseOffset = primitiveCenter - repeatOrigin + vec3(radius, 0.0, 0.0);
    float baseAngle = atan(baseOffset.y, baseOffset.x);
    float baseRadius = max(length(baseOffset.xy), 1e-4);
    float sector = (2.0 * WARP_PI) / float(count);
    float angle = atan(q.y, q.x);
    float angleRel = mod(angle - baseAngle + WARP_PI, 2.0 * WARP_PI) - WARP_PI;
    float normA = angleRel / sector;
    float id = round(normA);
    float local = normA - id;
    float arc = sector * baseRadius;
    float dR = (0.5 - local) * arc;
    float dL = (0.5 + local) * arc;
    float foldDeriv = sabsDerivative(dR, blend) + sabsDerivative(dL, blend) - 1.0;
    local += ((dR - sabs(dR, blend)) - (dL - sabs(dL, blend))) / arc;
    float foldA = local * sector + baseAngle;
    vec2 eFoldR = vec2(cos(foldA), sin(foldA));
    vec2 eFoldT = vec2(-eFoldR.y, eFoldR.x);
    float gR = dot(gradArray.xy, eFoldR);
    float gT = dot(gradArray.xy, eFoldT);
    float qr = length(q.xy);
    vec2 eR = (qr > 1e-8) ? (q.xy / qr) : vec2(cos(angle), sin(angle));
    vec2 eT = vec2(-eR.y, eR.x);
    gradArray.xy = gR * eR + (gT * foldDeriv) * eT;
  }
  return restoreRotationScaleGradient(gradArray, rotation, scale);
}

vec4 evalPrimitiveGradient(int primitiveIndex, vec3 worldPoint)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 meta = loadRow(baseRow);
  vec4 extra0 = primitiveExtra0(primitiveIndex);
  vec4 extra1 = primitiveExtra1(primitiveIndex);
  vec4 extra2 = primitiveExtra2(primitiveIndex);
  vec4 scaleRow = loadRow(baseRow + 4);
  vec3 scale = max(scaleRow.xyz, vec3(1e-6));
  int packedWarpInfo = int(scaleRow.w + 0.5);
  int warpOffset = packedWarpInfo / MIRROR_WARP_PACK_SCALE;
  int warpCount = packedWarpInfo - warpOffset * MIRROR_WARP_PACK_SCALE;
  float distanceScale;
  vec3 localPoint = toLocalPoint(primitiveIndex, worldPoint, distanceScale);
  float minScale = max(min(scale.x, min(scale.y, scale.z)), 1e-6);
  int primitiveType = int(meta.x + 0.5);
  float baseDistance = evalPrimitiveLocalDistance(primitiveType, meta, extra0, extra1, extra2, localPoint, scale, minScale) * distanceScale;

  vec3 localGradient = vec3(0.0);
  if (primitiveType == 0) {
    localGradient = sdgEllipsoid(localPoint, vec3(meta.y) * scale).yzw;
  }
  else if (primitiveType == 1) {
    float tapTop = max(extra1.w, 0.0);
    float tapBot = max(-extra1.w, 0.0);
    vec4 dg = (extra0.y + extra0.z + extra0.w + extra1.x + extra1.y + extra1.z + tapTop + tapBot > 0.001)
      ? sdgAdvancedBox(localPoint, meta.yzw * scale, vec4(extra0.y, extra0.z, extra0.w, extra1.x), extra1.y, extra1.z, tapTop, tapBot, int(extra2.x + 0.5), int(extra2.y + 0.5), meta.w * scale.z)
      : sdgBox(localPoint, meta.yzw * scale);
    localGradient = dg.yzw;
  }
  else if (primitiveType == 2) {
    vec3 scaledPoint = localPoint / scale;
    float bevelTop = max(extra0.x, 0.0);
    float bevelBottom = max(extra0.y, 0.0);
    vec4 dg = ((max(bevelTop, bevelBottom) + abs(extra0.z)) > 0.001)
      ? sdgAdvancedCylinder(scaledPoint, meta.y, meta.z, bevelTop, bevelBottom, extra0.z, int(extra0.w + 0.5))
      : sdgCylinder(scaledPoint, vec3(meta.y, meta.y, meta.z));
    localGradient = dg.yzw * (vec3(minScale) / scale);
  }
  else if (primitiveType == 3) {
    float angle = clamp(extra0.y, 0.0, 6.283185307179586);
    localGradient = (((6.283185307179586 - angle) > 0.001)
      ? sdgCappedTorus(localPoint / scale, vec2(sin(0.5 * angle), cos(0.5 * angle)), meta.y, meta.z)
      : sdgTorus(localPoint / scale, vec2(meta.y, meta.z))).yzw * (vec3(minScale) / scale);
  }
  else if (primitiveType == 4) {
    localGradient = sdgCone(localPoint / scale, meta.y, meta.z).yzw * (vec3(minScale) / scale);
  }
  else if (primitiveType == 5) {
    localGradient = sdgCapsule(localPoint / scale, meta.y, meta.z).yzw * (vec3(minScale) / scale);
  }
  else if (primitiveType == 6) {
    vec3 scaledPoint = localPoint / scale;
    float tapTop = max(extra1.x, 0.0);
    float tapBot = max(-extra1.x, 0.0);
    vec4 dg = (extra0.y + extra0.z + extra0.w + tapTop + tapBot + extra1.z > 0.001)
      ? sdgAdvancedNgon(scaledPoint, meta.y, meta.z, int(meta.w + 0.5), extra0.y, extra0.z, extra0.w, tapTop, tapBot, int(extra1.y + 0.5), meta.z, extra1.z)
      : sdgExtrude2DGrad(sdgRegularPolygon2D(scaledPoint.xy, meta.y, int(meta.w + 0.5)), scaledPoint.z, meta.z);
    localGradient = dg.yzw * (vec3(minScale) / scale);
  }
  else if (primitiveType == 7) {
    vec3 scaledPoint = localPoint / scale;
    float tapTop = max(extra0.w, 0.0);
    float tapBot = max(-extra0.w, 0.0);
    vec4 dg = (extra0.y + extra0.z + tapTop + tapBot > 0.001)
      ? sdgAdvancedPolygon(scaledPoint, meta.w, int(meta.y + 0.5), int(meta.z + 0.5), extra0.y, extra0.z, tapTop, tapBot, int(extra1.x + 0.5), meta.w)
      : sdgExtrude2DGrad(sdgPolygon2D(scaledPoint.xy, int(meta.y + 0.5), int(meta.z + 0.5)), scaledPoint.z, meta.w);
    localGradient = dg.yzw * (vec3(minScale) / scale);
  }

  vec3 worldGradient = primitiveGradientToWorld(baseRow, localGradient);
  for (int index = warpCount - 1; index >= 0; index--) {
    float prefixScale;
    vec3 preWarpPoint = applyWarpStack(baseRow, worldPoint, index, prefixScale);
    int warpRow = primitiveCountValue() * PRIMITIVE_STRIDE + warpOffset + index * WARP_ROWS_PER_ENTRY;
    vec4 warp0 = loadRow(warpRow);
    vec4 warp1 = loadRow(warpRow + 1);
    vec4 warp2 = loadRow(warpRow + 2);
    vec4 warp3 = loadRow(warpRow + 3);
    vec4 warp4 = loadRow(warpRow + 4);
    vec4 warp5 = loadRow(warpRow + 5);
    vec4 warp6 = loadRow(warpRow + 6);
    vec4 warp7 = loadRow(warpRow + 7);
    vec4 warp8 = loadRow(warpRow + 8);
    int warpKind = int(warp0.x + 0.5);
    if (warpKind == WARP_KIND_MIRROR) {
      worldGradient = invertMirrorWarpGradient(worldGradient, preWarpPoint, int(warp0.y + 0.5), max(warp0.z, 0.0), warp1.xyz);
      continue;
    }
    if (warpKind == WARP_KIND_GRID) {
      worldGradient = invertGridWarpGradient(worldGradient, preWarpPoint, ivec3(int(warp0.y + 0.5), int(warp0.z + 0.5), int(warp0.w + 0.5)), abs(warp1.xyz), max(warp1.w, 0.0), warp2.xyz, unpackWarpRotation(warp3.xyz), warp4.xyz, warp5.xyz, warp6.xyz, unpackWarpRotation(warp7.xyz), warp8.xyz);
      continue;
    }
    if (warpKind == WARP_KIND_RADIAL) {
      worldGradient = invertRadialWarpGradient(worldGradient, preWarpPoint, max(warp0.w, 0.0), int(warp0.y + 0.5), max(warp0.z, 0.0), warp1.xyz, warp2.xyz, unpackWarpRotation(warp3.xyz), warp4.xyz, warp5.xyz, warp6.xyz, unpackWarpRotation(warp7.xyz), warp8.xyz);
    }
  }

  worldGradient *= distanceScale;
  return normalizeDistanceGradient(baseDistance, worldGradient);
}

vec4 applySignedOpGradient(int op, vec4 lhs, vec4 rhs, float blend)
{
  if (op == 1) {
    if (blend <= 1e-6) {
      return (lhs.x <= rhs.x) ? lhs : rhs;
    }
    float h = clamp(0.5 + 0.5 * (rhs.x - lhs.x) / blend, 0.0, 1.0);
    return vec4(
      mix(rhs.x, lhs.x, h) - blend * h * (1.0 - h),
      mix(rhs.yzw, lhs.yzw, h));
  }
  if (blend <= 1e-6) {
    return (lhs.x >= rhs.x) ? lhs : rhs;
  }
  float h = clamp(0.5 - 0.5 * (rhs.x - lhs.x) / blend, 0.0, 1.0);
  return vec4(
    mix(rhs.x, lhs.x, h) + blend * h * (1.0 - h),
    mix(rhs.yzw, lhs.yzw, h));
}

vec4 applyOpGradient(int op, vec4 lhs, vec4 rhs, float blend)
{
  if (op == 1) {
    if (blend <= 1e-6) {
      return (lhs.x <= rhs.x) ? lhs : rhs;
    }
    float h = clamp(0.5 + 0.5 * (rhs.x - lhs.x) / blend, 0.0, 1.0);
    return vec4(
      mix(rhs.x, lhs.x, h) - blend * h * (1.0 - h),
      mix(rhs.yzw, lhs.yzw, h));
  }
  if (op == 2) {
    if (blend <= 1e-6) {
      return (lhs.x >= -rhs.x) ? lhs : negateDistanceGradient(rhs);
    }
    float h = clamp(0.5 - 0.5 * (rhs.x + lhs.x) / blend, 0.0, 1.0);
    return vec4(
      mix(lhs.x, -rhs.x, h) + blend * h * (1.0 - h),
      mix(lhs.yzw, -rhs.yzw, h));
  }
  if (op == 3) {
    if (blend <= 1e-6) {
      return (lhs.x >= rhs.x) ? lhs : rhs;
    }
    float h = clamp(0.5 - 0.5 * (rhs.x - lhs.x) / blend, 0.0, 1.0);
    return vec4(
      mix(rhs.x, lhs.x, h) + blend * h * (1.0 - h),
      mix(rhs.yzw, lhs.yzw, h));
  }
  return rhs;
}

float evalPrimitive(int primitiveIndex, vec3 worldPoint)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 meta = loadRow(baseRow);
  vec4 extra0 = primitiveExtra0(primitiveIndex);
  vec4 extra1 = primitiveExtra1(primitiveIndex);
  vec4 extra2 = primitiveExtra2(primitiveIndex);
  float distanceScale;
  vec3 localPoint = toLocalPoint(primitiveIndex, worldPoint, distanceScale);
  vec3 scale = primitiveScale(primitiveIndex);
  float minScale = max(min(scale.x, min(scale.y, scale.z)), 1e-6);
  int primitiveType = int(meta.x + 0.5);
  return evalPrimitiveLocalDistance(primitiveType, meta, extra0, extra1, extra2, localPoint, scale, minScale) * distanceScale;
}

float evalInstructionRange(vec3 worldPoint, int instructionBase, int instructionCount)
{
  float stack[MAX_STACK];
  int stackPointer = 0;
  for (int index = 0; index < instructionCount; index++) {
    vec4 instruction = loadRow(instructionBase + index);
    int kind = int(instruction.x + 0.5);
    float distanceValue;
    if (kind == 0) {
      distanceValue = evalPrimitive(int(instruction.y + 0.5), worldPoint);
    }
    else {
      if (stackPointer < 2) {
        return 1e20;
      }
      float rhs = stack[stackPointer - 1];
      float lhs = stack[stackPointer - 2];
      stackPointer -= 2;
      distanceValue = applyOp(kind, lhs, rhs, max(instruction.w, 0.0));
    }
    if (stackPointer >= MAX_STACK) {
      return 1e20;
    }
    stack[stackPointer] = distanceValue;
    stackPointer++;
  }
  if (stackPointer == 0) {
    return 1e20;
  }
  return stack[stackPointer - 1];
}

vec4 evalInstructionRangeGradient(vec3 worldPoint, int instructionBase, int instructionCount, out bool valid)
{
  valid = true;
  vec4 stack[MAX_STACK];
  int stackPointer = 0;
  for (int index = 0; index < instructionCount; index++) {
    vec4 instruction = loadRow(instructionBase + index);
    int kind = int(instruction.x + 0.5);
    vec4 value;
    if (kind == 0) {
      value = evalPrimitiveGradient(int(instruction.y + 0.5), worldPoint);
    }
    else {
      if (stackPointer < 2) {
        valid = false;
        return vec4(1e20, vec3(0.0));
      }
      vec4 rhs = stack[stackPointer - 1];
      vec4 lhs = stack[stackPointer - 2];
      stackPointer -= 2;
      value = applyOpGradient(kind, lhs, rhs, max(instruction.w, 0.0));
    }
    if (stackPointer >= MAX_STACK) {
      valid = false;
      return vec4(1e20, vec3(0.0));
    }
    stack[stackPointer] = value;
    stackPointer++;
  }
  if (stackPointer == 0) {
    valid = false;
    return vec4(1e20, vec3(0.0));
  }
  return stack[stackPointer - 1];
}

int pruningCellIndex(vec3 worldPoint)
{
  int gridSize = max(pruningGridSizeValue(), 1);
  vec3 boundsMin = pruningBoundsMinValue();
  vec3 boundsMax = pruningBoundsMaxValue();
  vec3 boundsSize = max(boundsMax - boundsMin, vec3(1e-6));
  vec3 gridPos = clamp((worldPoint - boundsMin) / boundsSize, vec3(0.0), vec3(0.999999));
  ivec3 cellCoord = ivec3(floor(gridPos * float(gridSize)));

  uint x = uint(cellCoord.x) & 0x000003ffu;
  uint y = uint(cellCoord.y) & 0x000003ffu;
  uint z = uint(cellCoord.z) & 0x000003ffu;

  x = (x ^ (x << 16)) & 0xff0000ffu;
  x = (x ^ (x << 8)) & 0x0300f00fu;
  x = (x ^ (x << 4)) & 0x030c30c3u;
  x = (x ^ (x << 2)) & 0x09249249u;

  y = (y ^ (y << 16)) & 0xff0000ffu;
  y = (y ^ (y << 8)) & 0x0300f00fu;
  y = (y ^ (y << 4)) & 0x030c30c3u;
  y = (y ^ (y << 2)) & 0x09249249u;

  z = (z ^ (z << 16)) & 0xff0000ffu;
  z = (z ^ (z << 8)) & 0x0300f00fu;
  z = (z ^ (z << 4)) & 0x030c30c3u;
  z = (z ^ (z << 2)) & 0x09249249u;

  return int((x) | (y << 1) | (z << 2));
}

bool bboxIntersect(vec3 boxMin, vec3 boxMax, vec3 rayOrigin, vec3 rayDirection, out float travel, out float travelExit)
{
  float t0 = 0.0;
  float t1 = 1e30;
  for (int axis = 0; axis < 3; axis++) {
    float origin = rayOrigin[axis];
    float direction = rayDirection[axis];
    float axisMin = boxMin[axis];
    float axisMax = boxMax[axis];
    if (abs(direction) < 1e-8) {
      if (origin < axisMin || origin > axisMax) {
        travel = 0.0;
        travelExit = 0.0;
        return false;
      }
      continue;
    }
    float invDirection = 1.0 / direction;
    float axisT0 = (axisMin - origin) * invDirection;
    float axisT1 = (axisMax - origin) * invDirection;
    t0 = max(t0, min(axisT0, axisT1));
    t1 = min(t1, max(axisT0, axisT1));
    if (t1 <= t0) {
      travel = 0.0;
      travelExit = 0.0;
      return false;
    }
  }
  travel = t0;
  travelExit = t1;
  return true;
}

float evalActiveScene(vec3 worldPoint, out bool nearField, out int activeCount, out float cellValue)
{
  nearField = false;
  activeCount = -1;
  cellValue = 1e20;
  if (!pruningEnabledValue()) {
    return 1e20;
  }

  vec3 boundsMin = pruningBoundsMinValue();
  vec3 boundsMax = pruningBoundsMaxValue();
  if (any(lessThan(worldPoint, boundsMin)) || any(greaterThan(worldPoint, boundsMax))) {
    return 1e20;
  }

  int cellIndex = pruningCellIndex(worldPoint);
  activeCount = int(loadScalar(pruneCellCounts, cellIndex) + 0.5);
  cellValue = loadScalar(pruneCellErrors, cellIndex);
  if (activeCount <= 0) {
    return cellValue;
  }

  nearField = true;
  float stack[MAX_STACK];
  int stackPointer = 0;
  int cellOffset = int(loadScalar(pruneCellOffsets, cellIndex) + 0.5);
  int instructionBase = instructionBaseValue();
  for (int index = 0; index < activeCount; index++) {
    float activeNode = loadScalar(pruneActiveNodes, cellOffset + index);
    int instructionIndex = activeNodeIndex(activeNode);
    vec4 instruction = loadRow(instructionBase + instructionIndex);
    int kind = int(instruction.x + 0.5);
    float distanceValue;
    if (kind == 0) {
      distanceValue = evalPrimitive(int(instruction.y + 0.5), worldPoint);
    }
    else {
      if (stackPointer < 2) {
        return 1e20;
      }
      float rhs = stack[stackPointer - 1];
      float lhs = stack[stackPointer - 2];
      stackPointer -= 2;
      distanceValue = applySignedOp(kind, lhs, rhs, max(instruction.w, 0.0));
    }
    distanceValue *= activeNodeSign(activeNode) ? 1.0 : -1.0;
    if (stackPointer >= MAX_STACK) {
      return 1e20;
    }
    stack[stackPointer] = distanceValue;
    stackPointer++;
  }

  if (stackPointer == 0) {
    return 1e20;
  }
  return stack[stackPointer - 1];
}

float evalActiveSceneForCell(vec3 worldPoint, int cellIndex, out bool nearField)
{
  int activeCount = int(loadScalar(pruneCellCounts, cellIndex) + 0.5);
  if (activeCount <= 0) {
    nearField = false;
    return loadScalar(pruneCellErrors, cellIndex);
  }

  nearField = true;
  float stack[MAX_STACK];
  int stackPointer = 0;
  int cellOffset = int(loadScalar(pruneCellOffsets, cellIndex) + 0.5);
  int instructionBase = instructionBaseValue();
  for (int index = 0; index < activeCount; index++) {
    float activeNode = loadScalar(pruneActiveNodes, cellOffset + index);
    int instructionIndex = activeNodeIndex(activeNode);
    vec4 instruction = loadRow(instructionBase + instructionIndex);
    int kind = int(instruction.x + 0.5);
    float distanceValue;
    if (kind == 0) {
      distanceValue = evalPrimitive(int(instruction.y + 0.5), worldPoint);
    }
    else {
      if (stackPointer < 2) {
        return 1e20;
      }
      float rhs = stack[stackPointer - 1];
      float lhs = stack[stackPointer - 2];
      stackPointer -= 2;
      distanceValue = applySignedOp(kind, lhs, rhs, max(instruction.w, 0.0));
    }
    distanceValue *= activeNodeSign(activeNode) ? 1.0 : -1.0;
    if (stackPointer >= MAX_STACK) {
      return 1e20;
    }
    stack[stackPointer] = distanceValue;
    stackPointer++;
  }

  return (stackPointer == 0) ? 1e20 : stack[stackPointer - 1];
}

vec4 evalActiveSceneGradientForCell(vec3 worldPoint, int cellIndex, out bool valid)
{
  int activeCount = int(loadScalar(pruneCellCounts, cellIndex) + 0.5);
  if (activeCount <= 0) {
    valid = false;
    return vec4(1e20, vec3(0.0));
  }

  valid = true;
  vec4 stack[MAX_STACK];
  int stackPointer = 0;
  int cellOffset = int(loadScalar(pruneCellOffsets, cellIndex) + 0.5);
  int instructionBase = instructionBaseValue();
  for (int index = 0; index < activeCount; index++) {
    float activeNode = loadScalar(pruneActiveNodes, cellOffset + index);
    int instructionIndex = activeNodeIndex(activeNode);
    vec4 instruction = loadRow(instructionBase + instructionIndex);
    int kind = int(instruction.x + 0.5);
    vec4 value;
    if (kind == 0) {
      value = evalPrimitiveGradient(int(instruction.y + 0.5), worldPoint);
    }
    else {
      if (stackPointer < 2) {
        valid = false;
        return vec4(1e20, vec3(0.0));
      }
      vec4 rhs = stack[stackPointer - 1];
      vec4 lhs = stack[stackPointer - 2];
      stackPointer -= 2;
      value = applySignedOpGradient(kind, lhs, rhs, max(instruction.w, 0.0));
    }
    if (!activeNodeSign(activeNode)) {
      value = negateDistanceGradient(value);
    }
    if (stackPointer >= MAX_STACK) {
      valid = false;
      return vec4(1e20, vec3(0.0));
    }
    stack[stackPointer] = value;
    stackPointer++;
  }

  if (stackPointer == 0) {
    valid = false;
    return vec4(1e20, vec3(0.0));
  }
  return stack[stackPointer - 1];
}

float evalScene(vec3 worldPoint)
{
  bool nearField;
  int activeCount;
  float cellValue;
  float activeDistance = evalActiveScene(worldPoint, nearField, activeCount, cellValue);
  if (pruningEnabledValue()) {
    if (activeCount >= 0) {
      if (!nearField) {
        return cellValue;
      }
      if (activeCount > 0) {
        return activeDistance;
      }
    }
  }
  int instructionBase = instructionBaseValue();
  return evalInstructionRange(worldPoint, instructionBase, instructionCountValue());
}

float evalSceneExact(vec3 worldPoint)
{
  int instructionBase = instructionBaseValue();
  return evalInstructionRange(worldPoint, instructionBase, instructionCountValue());
}

float evalSceneTrace(vec3 worldPoint, out bool exactSample, out bool nearField, out int activeCount, out float cellValue)
{
  nearField = false;
  activeCount = -1;
  cellValue = 1e20;
  exactSample = true;

  float activeDistance = evalActiveScene(worldPoint, nearField, activeCount, cellValue);
  if (pruningEnabledValue() && activeCount >= 0) {
    if (!nearField) {
      exactSample = false;
      return cellValue;
    }
    if (activeCount > 0) {
      return activeDistance;
    }
  }
  return evalSceneExact(worldPoint);
}

vec3 estimateSurfaceNormal(vec3 worldPoint, vec3 rayDirection, int cellIndex)
{
  bool valid = false;
  vec4 gradientValue = vec4(1e20, vec3(0.0));
  if (pruningEnabledValue() && cellIndex >= 0) {
    gradientValue = evalActiveSceneGradientForCell(worldPoint, cellIndex, valid);
  }
  if (!valid) {
    int instructionBase = instructionBaseValue();
    gradientValue = evalInstructionRangeGradient(worldPoint, instructionBase, instructionCountValue(), valid);
  }
  if (valid) {
    float lenSq = dot(gradientValue.yzw, gradientValue.yzw);
    if (lenSq > 1e-12) {
      vec3 normal = gradientValue.yzw * inversesqrt(lenSq);
      if (dot(normal, -rayDirection) < 0.0) {
        normal = -normal;
      }
      return normal;
    }
  }
  return -rayDirection;
}

void computeRay(vec2 uv, out vec3 rayOrigin, out vec3 rayDirection)
{
  vec2 ndc = uv * 2.0 - 1.0;
  vec4 nearPoint = invViewProjectionMatrix * vec4(ndc, -1.0, 1.0);
  vec4 farPoint = invViewProjectionMatrix * vec4(ndc, 1.0, 1.0);
  nearPoint.xyz /= nearPoint.w;
  farPoint.xyz /= farPoint.w;
  rayDirection = normalize(farPoint.xyz - nearPoint.xyz);
  rayOrigin = orthographicViewValue() ? nearPoint.xyz : mathops.cameraPosition.xyz;
}
""" % (runtime.PRIMITIVE_TEXELS, runtime.MAX_STACK)


FRAGMENT_SOURCE = COMMON_SOURCE + """\
struct SurfaceInfo {
  float distanceValue;
  float outlineId;
};

SurfaceInfo combineSurfaceInfo(int op, SurfaceInfo lhs, SurfaceInfo rhs, float blend)
{
  if (op == 1) {
    if (blend <= 1e-6) {
      return (lhs.distanceValue <= rhs.distanceValue) ? lhs : rhs;
    }
    float h = clamp(0.5 + 0.5 * (rhs.distanceValue - lhs.distanceValue) / blend, 0.0, 1.0);
    return SurfaceInfo(
      mix(rhs.distanceValue, lhs.distanceValue, h) - blend * h * (1.0 - h),
      (h > 0.5) ? lhs.outlineId : rhs.outlineId
    );
  }
  if (op == 2) {
    if (blend <= 1e-6) {
      return (lhs.distanceValue >= -rhs.distanceValue) ? lhs : rhs;
    }
    float h = clamp(0.5 - 0.5 * (rhs.distanceValue + lhs.distanceValue) / blend, 0.0, 1.0);
    return SurfaceInfo(
      mix(lhs.distanceValue, -rhs.distanceValue, h) + blend * h * (1.0 - h),
      (h > 0.5) ? rhs.outlineId : lhs.outlineId
    );
  }
  if (op == 3) {
    if (blend <= 1e-6) {
      return (lhs.distanceValue >= rhs.distanceValue) ? lhs : rhs;
    }
    float h = clamp(0.5 - 0.5 * (rhs.distanceValue - lhs.distanceValue) / blend, 0.0, 1.0);
    return SurfaceInfo(
      mix(rhs.distanceValue, lhs.distanceValue, h) + blend * h * (1.0 - h),
      (h > 0.5) ? lhs.outlineId : rhs.outlineId
    );
  }
  return rhs;
}

float loadOutlineId(int primitiveIndex)
{
  return loadScalar(outlineData, primitiveIndex);
}

float surfaceOutlineId(vec3 worldPoint)
{
  SurfaceInfo stack[MAX_STACK];
  int stackPointer = 0;
  int instructionBase = instructionBaseValue();
  for (int index = 0; index < instructionCountValue(); index++) {
    vec4 instruction = loadRow(instructionBase + index);
    int kind = int(instruction.x + 0.5);
    SurfaceInfo info;
    if (kind == 0) {
      int primitiveIndex = int(instruction.y + 0.5);
      info = SurfaceInfo(evalPrimitive(primitiveIndex, worldPoint), loadOutlineId(primitiveIndex));
    }
    else {
      if (stackPointer < 2) {
        return 0.0;
      }
      SurfaceInfo rhs = stack[stackPointer - 1];
      SurfaceInfo lhs = stack[stackPointer - 2];
      stackPointer -= 2;
      info = combineSurfaceInfo(kind, lhs, rhs, max(instruction.w, 0.0));
    }
    if (stackPointer >= MAX_STACK) {
      return 0.0;
    }
    stack[stackPointer] = info;
    stackPointer++;
  }
  return (stackPointer == 0) ? 0.0 : stack[stackPointer - 1].outlineId;
}

vec3 backgroundColor(vec3 rayDirection)
{
  return mathops.sceneLayout.yzw;
}

vec2 matcapUV(vec3 normalView, vec3 viewDirView)
{
  float a = 1.0 / max(1.0 + viewDirView.z, 0.001);
  float b = -viewDirView.x * viewDirView.y * a;
  vec3 basisX = vec3(
    1.0 - viewDirView.x * viewDirView.x * a,
    b,
    -viewDirView.x
  );
  vec3 basisY = vec3(
    b,
    1.0 - viewDirView.y * viewDirView.y * a,
    -viewDirView.y
  );
  vec2 uv = vec2(dot(basisX, normalView), dot(basisY, normalView));
  return clamp(uv * 0.496 + 0.5, vec2(0.0), vec2(1.0));
}

vec3 shadeSurface(vec3 normal, vec3 rayDirection)
{
  vec3 normalView = normalize(vec3(
    dot(mathops.viewRow0.xyz, normal),
    dot(mathops.viewRow1.xyz, normal),
    dot(mathops.viewRow2.xyz, normal)
  ));
  vec3 viewDirView = normalize(vec3(
    -dot(mathops.viewRow0.xyz, rayDirection),
    -dot(mathops.viewRow1.xyz, rayDirection),
    -dot(mathops.viewRow2.xyz, rayDirection)
  ));
  vec2 uv = matcapUV(normalView, viewDirView);
  vec3 color = texture(matcapDiffuse, uv).rgb;
  if (showSpecularValue()) {
    color += texture(matcapSpecular, uv).rgb;
  }
  return color;
}

vec3 pruningDebugColor(vec3 worldPoint, vec3 fallbackColor)
{
  if (!pruningEnabledValue()) {
    if (debugModeValue() == 1) {
      return inferno(min(1.0, float(instructionCountValue()) / debugVizMaxValue()));
    }
    return fallbackColor;
  }

  int cellIndex = pruningCellIndex(worldPoint);
  int activeCount = int(loadScalar(pruneCellCounts, cellIndex) + 0.5);
  float cellValue = loadScalar(pruneCellErrors, cellIndex);
  if (debugModeValue() == 1) {
    int primitiveEstimate = (activeCount + 1) / 2;
    return inferno(min(1.0, float(primitiveEstimate) / debugVizMaxValue()));
  }
  if (debugModeValue() == 2) {
    return inferno(min(1.0, abs(cellValue) / max(debugVizMaxValue(), 1.0)));
  }
  return fallbackColor;
}

vec3 stepCountColor(int stepsTaken)
{
  return inferno(min(1.0, float(stepsTaken) / 128.0));
}

float surfaceDepth(vec3 worldPoint)
{
  vec4 clipPoint = viewProjectionMatrix * vec4(worldPoint, 1.0);
  return clamp((clipPoint.z / max(clipPoint.w, 1e-7)) * 0.5 + 0.5, 0.00001, 0.99999);
}

void main()
{
  OutlineId = 0.0;
  HitPosition = vec4(0.0);
  vec3 rayOrigin;
  vec3 rayDirection;
  computeRay(uvInterp, rayOrigin, rayDirection);
  if (primitiveCountValue() <= 0 || instructionCountValue() <= 0) {
    vec3 color = (debugModeValue() == 3) ? vec3(0.0) : backgroundColor(rayDirection);
    FragColor = vec4(pow(color, vec3(gammaValue())), 1.0);
    gl_FragDepth = 0.99999;
    return;
  }

  if (!orthographicViewValue() && evalSceneExact(mathops.cameraPosition.xyz) < 0.0) {
    vec3 color = backgroundColor(rayDirection);
    FragColor = vec4(pow(color, vec3(gammaValue())), 1.0);
    gl_FragDepth = 0.99999;
    return;
  }

  vec3 boundsMin = renderBoundsMinValue();
  vec3 boundsMax = renderBoundsMaxValue();
  float travel = 0.0;
  float travelExit = 0.0;
  if (!bboxIntersect(boundsMin, boundsMax, rayOrigin, rayDirection, travel, travelExit)) {
    vec3 color = (debugModeValue() == 3) ? vec3(0.0) : backgroundColor(rayDirection);
    FragColor = vec4(pow(color, vec3(gammaValue())), 1.0);
    gl_FragDepth = 0.99999;
    return;
  }

  travel += 1e-4;
  float marchDistance = 0.0;

  bool hit = false;
  vec3 worldPoint = rayOrigin;
  int hitCellIndex = -1;
  int maxSteps = maxStepsValue();
  int stepsTaken = 0;
  float travelPrev = travel;
  float distPrev = 1e20;
  bool prevExact = false;
  for (int step = 0; step < maxSteps; step++) {
    if (step >= 4096 || travel > travelExit || marchDistance > maxDistanceValue()) {
      break;
    }
    stepsTaken = step + 1;
    worldPoint = rayOrigin + rayDirection * travel;
    if (any(lessThan(worldPoint, boundsMin)) || any(greaterThanEqual(worldPoint, boundsMax))) {
      hit = false;
      break;
    }
    bool exactSample;
    bool nearField;
    int activeCount;
    float cellValue;
    float dist = evalSceneTrace(worldPoint, exactSample, nearField, activeCount, cellValue);
    float absDist = abs(dist);
    float adaptiveEpsilon = surfaceEpsilonValue() * (1.0 + travel * 0.001);
    bool signChange = exactSample && prevExact && (distPrev < 1e19) && (distPrev > 0.0) && (dist < 0.0);
    if (exactSample && (absDist < adaptiveEpsilon || signChange)) {
      hit = true;
      float denom = distPrev - dist;
      if (denom > 1e-8) {
        float alpha = distPrev / denom;
        if (alpha > 0.0 && alpha < 1.0 && distPrev < 1e19) {
          worldPoint = rayOrigin + rayDirection * mix(travelPrev, travel, alpha);
        }
        else {
          float cosTheta = denom / max(travel - travelPrev, 1e-8);
          float nearFactor = clamp(5.0 / max(travel, 0.1), 0.0, 1.0);
          float minCos = mix(0.9, 0.25, nearFactor);
          worldPoint += rayDirection * dist / clamp(cosTheta, minCos, 1.0);
        }
      }
      else {
        worldPoint += rayDirection * dist;
      }

      {
        float epsSnap = surfaceEpsilonValue() * 0.5;
        float d0 = evalSceneExact(worldPoint);
        float d1 = evalSceneExact(worldPoint + rayDirection * epsSnap);
        float cosEst = clamp((d0 - d1) / epsSnap, 0.1, 1.0);
        worldPoint += rayDirection * d0 / cosEst;
      }

      if (pruningEnabledValue()) {
        hitCellIndex = pruningCellIndex(worldPoint);
      }
      break;
    }
    travelPrev = travel;
    distPrev = dist;
    prevExact = exactSample;
    float stepDistance = max(absDist, surfaceEpsilonValue() * 0.5);
    travel += stepDistance;
    marchDistance += stepDistance;
  }

  if (debugModeValue() == 3) {
    FragColor = vec4(pow(stepCountColor(stepsTaken), vec3(gammaValue())), 1.0);
    gl_FragDepth = 0.99999;
    return;
  }

  if (!hit) {
    FragColor = vec4(pow(backgroundColor(rayDirection), vec3(gammaValue())), 1.0);
    gl_FragDepth = 0.99999;
    return;
  }

  OutlineId = surfaceOutlineId(worldPoint);
  HitPosition = vec4(worldPoint, 1.0);
  vec3 color = disableSurfaceShadingValue() ? shadeSurface(vec3(0.0, 0.0, 1.0), rayDirection) : backgroundColor(rayDirection);
  if (debugModeValue() != 0) {
    color = pruningDebugColor(worldPoint, vec3(0.0));
    HitPosition = vec4(0.0);
    FragColor = vec4(pow(color, vec3(gammaValue())), 1.0);
    gl_FragDepth = surfaceDepth(worldPoint);
    return;
  }
  FragColor = vec4(pow(color, vec3(gammaValue())), 1.0);
  gl_FragDepth = surfaceDepth(worldPoint);
}
"""
