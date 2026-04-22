from .. import runtime


LOCAL_GROUP_SIZE = (4, 4, 4)


COMPUTE_SOURCE = """\
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#define PRIMITIVE_STRIDE %d
#define STACK_DEPTH %d
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



vec4 loadSceneRow(int row)
{
  return texelFetch(sceneData, ivec2(0, row), 0);
}

vec4 primitiveExtra0(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return loadSceneRow(baseRow + 5);
}

vec4 primitiveExtra1(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return loadSceneRow(baseRow + 6);
}

vec4 primitiveExtra2(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return loadSceneRow(baseRow + 7);
}

#define STORE_FLOAT_IMAGE(textureImage, index, value) imageStore(textureImage, linearCoord(imageSize(textureImage), index), vec4(value, 0.0, 0.0, 0.0))
#define LOAD_UINT_IMAGE(textureImage, index) imageLoad(textureImage, linearCoord(imageSize(textureImage), index)).x
#define STORE_UINT_IMAGE(textureImage, index, value) imageStore(textureImage, linearCoord(imageSize(textureImage), index), uvec4(value, 0u, 0u, 0u))
#define COUNTER_ATOMIC_ADD(index, value) imageAtomicAdd(countersImg, linearCoord(imageSize(countersImg), index), value)

float activeNodeEncode(int instructionIndex, bool positiveSign)
{
  float encoded = float(instructionIndex + 1);
  return positiveSign ? encoded : -encoded;
}

int activeNodeIndex(float encoded)
{
  return int(abs(encoded) + 0.5) - 1;
}

bool activeNodeSign(float encoded)
{
  return encoded >= 0.0;
}

float parentEncode(int parentIndex)
{
  return float(parentIndex + 1);
}

int parentDecode(float encoded)
{
  return int(encoded + 0.5) - 1;
}

uint tmpEncode(int state, bool activeGlobal, bool inactiveAncestors, bool positiveSign, int parentIndex)
{
  uint bits = uint(state & 3);
  if (activeGlobal) {
    bits |= (1u << 2);
  }
  if (inactiveAncestors) {
    bits |= (1u << 3);
  }
  if (positiveSign) {
    bits |= (1u << 4);
  }
  bits |= uint(parentIndex + 1) << 5;
  return bits;
}

int tmpStateGet(uint bits)
{
  return int(bits & 3u);
}

uint tmpStateSet(uint bits, int state)
{
  return (bits & ~3u) | uint(state & 3);
}

bool tmpActiveGlobalGet(uint bits)
{
  return ((bits >> 2) & 1u) != 0u;
}

uint tmpActiveGlobalSet(uint bits, bool value)
{
  bits &= ~(1u << 2);
  if (value) {
    bits |= (1u << 2);
  }
  return bits;
}

bool tmpInactiveAncestorsGet(uint bits)
{
  return ((bits >> 3) & 1u) != 0u;
}

uint tmpInactiveAncestorsSet(uint bits, bool value)
{
  bits &= ~(1u << 3);
  if (value) {
    bits |= (1u << 3);
  }
  return bits;
}

bool tmpSignGet(uint bits)
{
  return ((bits >> 4) & 1u) != 0u;
}

uint tmpSignSet(uint bits, bool value)
{
  bits &= ~(1u << 4);
  if (value) {
    bits |= (1u << 4);
  }
  return bits;
}

int tmpParentGet(uint bits)
{
  return int(bits >> 5) - 1;
}

uint tmpParentSet(uint bits, int parentIndex)
{
  bits &= 31u;
  bits |= uint(parentIndex + 1) << 5;
  return bits;
}

vec3 primitiveScale(int primitiveIndex)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  return max(loadSceneRow(baseRow + 4).xyz, vec3(1e-6));
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

vec3 toLocalPoint(int primitiveIndex, vec3 worldPoint, out float distanceScale)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 row0 = loadSceneRow(baseRow + 1);
  vec4 row1 = loadSceneRow(baseRow + 2);
  vec4 row2 = loadSceneRow(baseRow + 3);
  int packedWarpInfo = int(loadSceneRow(baseRow + 4).w + 0.5);
  int warpBase = primitiveCount * PRIMITIVE_STRIDE;
  int warpOffset = packedWarpInfo / MIRROR_WARP_PACK_SCALE;
  int warpCount = packedWarpInfo - warpOffset * MIRROR_WARP_PACK_SCALE;
  vec3 warpedPoint = worldPoint;
  distanceScale = 1.0;
  for (int index = 0; index < warpCount; index++) {
    int warpRow = warpBase + warpOffset + index * WARP_ROWS_PER_ENTRY;
    vec4 warp0 = loadSceneRow(warpRow);
    vec4 warp1 = loadSceneRow(warpRow + 1);
    vec4 warp2 = loadSceneRow(warpRow + 2);
    vec4 warp3 = loadSceneRow(warpRow + 3);
    vec4 warp4 = loadSceneRow(warpRow + 4);
    vec4 warp5 = loadSceneRow(warpRow + 5);
    vec4 warp6 = loadSceneRow(warpRow + 6);
    vec4 warp7 = loadSceneRow(warpRow + 7);
    vec4 warp8 = loadSceneRow(warpRow + 8);
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
  vec2 a = loadPolygonPoint(pointOffset + pointCount - 1);
  for (int i = 0; i < pointCount; i++) {
    vec2 b = loadPolygonPoint(pointOffset + i);
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
    a = b;
  }
  return (winding != 0) ? -d : d;
}

float sdQuadraticBezier2D(vec2 pos, vec2 A, vec2 B, vec2 C)
{
  vec2 a = B - A;
  vec2 b = A - 2.0*B + C;
  vec2 c = a * 2.0;
  vec2 d = A - pos;
  float bb = dot(b,b);
  if (bb < 1e-10) {
    vec2 e = C - A;
    float t = clamp(dot(d, e) / max(dot(e,e), 1e-12), 0.0, 1.0);
    return length(d + e*t);
  }
  float kk = 1.0/bb;
  float kx = kk * dot(a,b);
  float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
  float kz = kk * dot(d,a);
  float res = 1e20;
  float p = ky - kx*kx;
  float p3 = p*p*p;
  float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
  float h = q*q + 4.0*p3;
  if (h >= 0.0) {
    h = sqrt(h);
    vec2 x = (vec2(h,-h)-q)/2.0;
    vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
    float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
    vec2 w = d+(c+b*t)*t;
    res = dot(w,w);
  } else {
    float z = sqrt(-p);
    float v = acos(clamp(q/(p*z*2.0), -1.0, 1.0)) / 3.0;
    float m = cos(v);
    float n = sin(v)*1.732050808;
    vec3 t = clamp(vec3(m+m,-n-m,n-m)*z-kx, 0.0, 1.0);
    vec2 w0 = d+(c+b*t.x)*t.x;
    vec2 w1 = d+(c+b*t.y)*t.y;
    res = min(dot(w0,w0), dot(w1,w1));
  }
  return sqrt(res);
}

vec2 sdBezierQuadEndPoint(int pointOffset, int segCount, int i)
{
  int next = (i + 1 < segCount) ? (i + 1) : 0;
  return loadPolygonRow(pointOffset + next).xy;
}

float sdBezierPolygon2D(vec2 p, int pointOffset, int segCount)
{
  if (segCount < 3) {
    return 1e20;
  }
  float d = 1e20;
  int winding = 0;
  for (int i = 0; i < segCount; i++) {
    vec4 row = loadPolygonRow(pointOffset + i);
    vec2 start = row.xy;
    vec2 ctrl = row.zw;
    vec2 endPt = sdBezierQuadEndPoint(pointOffset, segCount, i);
    d = min(d, sdQuadraticBezier2D(p, start, ctrl, endPt));
    vec2 edge = endPt - start;
    vec2 rel = p - start;
    float cv = edge.x * rel.y - edge.y * rel.x;
    if (start.y <= p.y && endPt.y > p.y && cv > 0.0) {
      winding++;
    }
    if (start.y > p.y && endPt.y <= p.y && cv < 0.0) {
      winding--;
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

float evalPrimitive(int primitiveIndex, vec3 worldPoint)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 meta = loadSceneRow(baseRow);
  vec4 extra0 = primitiveExtra0(primitiveIndex);
  vec4 extra1 = primitiveExtra1(primitiveIndex);
  vec4 extra2 = primitiveExtra2(primitiveIndex);
  float distanceScale;
  vec3 localPoint = toLocalPoint(primitiveIndex, worldPoint, distanceScale);
  vec3 scale = primitiveScale(primitiveIndex);
  float minScale = max(min(scale.x, min(scale.y, scale.z)), 1e-6);
  int primitiveType = int(meta.x + 0.5);
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
  return (distanceValue - postBevel) * distanceScale;
}

float kernel(float x, float k)
{
  if (k <= 0.0) {
    return 0.0;
  }
  float m = max(0.0, k - x);
  return m * m * 0.25 / k;
}

float opSign(int opKind)
{
  return (opKind == 1) ? 1.0 : -1.0;
}

float evalSignedBinary(int opKind, float leftValue, float rightValue, float blend)
{
  float s = opSign(opKind);
  return s * (min(s * leftValue, s * rightValue) - kernel(abs(leftValue - rightValue), blend));
}

int getCellIndex(ivec3 cell, int currentGridSize)
{
  uint x = uint(cell.x) & 0x000003ffu;
  uint y = uint(cell.y) & 0x000003ffu;
  uint z = uint(cell.z) & 0x000003ffu;

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

int getParentCellIndex(int cellIndex, int currentGridSize)
{
  return cellIndex / 64;
}

float loadInputActiveNode(int parentOffset, int index)
{
  if (firstLevel != 0) {
    return loadScalar(activeNodesInit, index);
  }
  return loadScalar(activeNodesIn, parentOffset + index);
}

int loadInputParent(int parentOffset, int index)
{
  if (firstLevel != 0) {
    return parentDecode(loadScalar(parentsInit, index));
  }
  return parentDecode(loadScalar(parentsIn, parentOffset + index));
}

void flagOverflow()
{
  COUNTER_ATOMIC_ADD(statusCounterIndex, 1u);
}

void computePruning(vec3 cellCenter, vec3 cellSize, int cellIndex)
{
  const int NODESTATE_INACTIVE = 0;
  const int NODESTATE_SKIPPED = 1;
  const int NODESTATE_ACTIVE = 2;

  float stackDistances[STACK_DEPTH];
  int stackNodeIndices[STACK_DEPTH];
  int stackPointer = 0;
  float cellRadius = length(cellSize) * 0.5;

  int parentCellIndex = 0;
  int parentOffset = 0;
  int numNodes = totalNumNodes;
  if (firstLevel == 0) {
    parentCellIndex = getParentCellIndex(cellIndex, currentGridSize);
    parentOffset = int(loadScalar(parentCellOffsetsTex, parentCellIndex) + 0.5);
    numNodes = int(loadScalar(parentCellCountsTex, parentCellIndex) + 0.5);
  }

  if (numNodes <= 0) {
    STORE_FLOAT_IMAGE(childCellOffsetsImg, cellIndex, 0.0);
    STORE_FLOAT_IMAGE(cellCountsImg, cellIndex, 0.0);
    STORE_FLOAT_IMAGE(cellValueOutImg, cellIndex, firstLevel != 0 ? 0.0 : loadScalar(cellValueInTex, parentCellIndex));
    return;
  }

  if (numNodes == 1) {
    int cellOffset = int(COUNTER_ATOMIC_ADD(activeCounterIndex, 1u));
    if ((cellOffset + 1) > activeCapacity) {
      flagOverflow();
      return;
    }
    STORE_FLOAT_IMAGE(childCellOffsetsImg, cellIndex, float(cellOffset));
    STORE_FLOAT_IMAGE(cellCountsImg, cellIndex, 1.0);
    STORE_FLOAT_IMAGE(cellValueOutImg, cellIndex, firstLevel != 0 ? 0.0 : loadScalar(cellValueInTex, parentCellIndex));
    STORE_FLOAT_IMAGE(parentsOut, cellOffset, parentEncode(-1));
    STORE_FLOAT_IMAGE(activeNodesOut, cellOffset, loadInputActiveNode(parentOffset, 0));
    return;
  }

  int tmpOffset = -1;
  if (subgroupElect()) {
    tmpOffset = int(COUNTER_ATOMIC_ADD(tmpCounterIndex, uint(int(gl_SubgroupSize) * numNodes)));
  }
  tmpOffset = subgroupBroadcastFirst(tmpOffset);
  if ((tmpOffset + int(gl_SubgroupSize) * numNodes) > tmpCapacity) {
    flagOverflow();
    return;
  }

  for (int index = 0; index < numNodes; index++) {
    float activeNode = loadInputActiveNode(parentOffset, index);
    int instructionIndex = activeNodeIndex(activeNode);
    int instructionRow = (primitiveCount * PRIMITIVE_STRIDE) + warpRowCount + instructionIndex;
    vec4 instruction = loadSceneRow(instructionRow);
    int kind = int(instruction.x + 0.5);
    float distanceValue;
    if (kind == 0) {
      distanceValue = evalPrimitive(int(instruction.y + 0.5), cellCenter);
      STORE_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * index + int(gl_SubgroupInvocationID), tmpEncode(NODESTATE_ACTIVE, false, false, true, -1));
    }
    else {
      float leftValue = stackDistances[stackPointer - 2];
      float rightValue = stackDistances[stackPointer - 1];
      int leftIndex = stackNodeIndices[stackPointer - 2];
      int rightIndex = stackNodeIndices[stackPointer - 1];
      stackPointer -= 2;

      float blend = max(instruction.w, 0.0);
      distanceValue = evalSignedBinary(kind, leftValue, rightValue, blend);
      int currentState = NODESTATE_ACTIVE;
      if (kind != 2 && abs(leftValue - rightValue) > (2.0 * cellRadius + blend)) {
        currentState = NODESTATE_SKIPPED;
        float signedLeft = opSign(kind) * leftValue;
        float signedRight = opSign(kind) * rightValue;
        int inactiveIndex = (signedLeft < signedRight) ? rightIndex : leftIndex;
        uint inactiveBits = LOAD_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * inactiveIndex + int(gl_SubgroupInvocationID));
        inactiveBits = tmpStateSet(inactiveBits, NODESTATE_INACTIVE);
        STORE_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * inactiveIndex + int(gl_SubgroupInvocationID), inactiveBits);
      }
      STORE_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * index + int(gl_SubgroupInvocationID), tmpEncode(currentState, false, false, true, -1));
    }

    distanceValue *= activeNodeSign(activeNode) ? 1.0 : -1.0;
    stackDistances[stackPointer] = distanceValue;
    stackNodeIndices[stackPointer] = index;
    stackPointer += 1;
  }

  float rootDistance = stackDistances[0];
  if (abs(rootDistance) > (2.0 * cellRadius)) {
    STORE_FLOAT_IMAGE(childCellOffsetsImg, cellIndex, 0.0);
    STORE_FLOAT_IMAGE(cellCountsImg, cellIndex, 0.0);
    STORE_FLOAT_IMAGE(cellValueOutImg, cellIndex, sign(rootDistance) * (abs(rootDistance) - cellRadius));
    return;
  }

  int cellActiveCount = 0;
  for (int index = numNodes - 1; index >= 0; index--) {
    uint tmpBits = LOAD_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * index + int(gl_SubgroupInvocationID));
    if (tmpStateGet(tmpBits) == NODESTATE_INACTIVE) {
      tmpBits = tmpActiveGlobalSet(tmpBits, false);
      tmpBits = tmpInactiveAncestorsSet(tmpBits, true);
      STORE_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * index + int(gl_SubgroupInvocationID), tmpBits);
      continue;
    }

    int parentIndex = loadInputParent(parentOffset, index);
    uint parentBits = 0u;
    bool hasParent = parentIndex >= 0;
    if (hasParent) {
      parentBits = LOAD_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * parentIndex + int(gl_SubgroupInvocationID));
    }

    bool inactiveAncestors = hasParent ? tmpInactiveAncestorsGet(parentBits) : false;
    bool activeGlobal = (tmpStateGet(tmpBits) == NODESTATE_ACTIVE) && !inactiveAncestors;
    if (activeGlobal) {
      cellActiveCount += 1;
    }

    float oldActiveNode = loadInputActiveNode(parentOffset, index);
    int nodeSign = activeNodeSign(oldActiveNode) ? 1 : -1;
    int newParentIndex = parentIndex;
    if (hasParent && tmpStateGet(parentBits) == NODESTATE_SKIPPED) {
      nodeSign *= tmpSignGet(parentBits) ? 1 : -1;
      newParentIndex = tmpParentGet(parentBits);
    }

    tmpBits = tmpActiveGlobalSet(tmpBits, activeGlobal);
    tmpBits = tmpInactiveAncestorsSet(tmpBits, inactiveAncestors);
    tmpBits = tmpParentSet(tmpBits, newParentIndex);
    tmpBits = tmpSignSet(tmpBits, nodeSign > 0);
    STORE_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * index + int(gl_SubgroupInvocationID), tmpBits);
  }

  int cellOffset = int(COUNTER_ATOMIC_ADD(activeCounterIndex, uint(cellActiveCount)));
  if ((cellOffset + cellActiveCount) > activeCapacity) {
    flagOverflow();
    return;
  }

  int outIndex = cellActiveCount - 1;
  for (int index = numNodes - 1; index >= 0; index--) {
    uint tmpBits = LOAD_UINT_IMAGE(tmpImg, tmpOffset + int(gl_SubgroupSize) * index + int(gl_SubgroupInvocationID));
    if (!tmpActiveGlobalGet(tmpBits)) {
      continue;
    }

    float oldActiveNode = loadInputActiveNode(parentOffset, index);
    STORE_FLOAT_IMAGE(activeNodesOut, cellOffset + outIndex, activeNodeEncode(activeNodeIndex(oldActiveNode), tmpSignGet(tmpBits)));
    STORE_UINT_IMAGE(oldToNewImg, tmpOffset + int(gl_SubgroupSize) * index + int(gl_SubgroupInvocationID), uint(outIndex + 1));

    int newParentOld = tmpParentGet(tmpBits);
    int newParent = -1;
    if (newParentOld >= 0) {
      newParent = int(LOAD_UINT_IMAGE(oldToNewImg, tmpOffset + int(gl_SubgroupSize) * newParentOld + int(gl_SubgroupInvocationID))) - 1;
    }
    STORE_FLOAT_IMAGE(parentsOut, cellOffset + outIndex, parentEncode(newParent));
    outIndex -= 1;
  }

  STORE_FLOAT_IMAGE(childCellOffsetsImg, cellIndex, float(cellOffset));
  STORE_FLOAT_IMAGE(cellCountsImg, cellIndex, float(cellActiveCount));
  STORE_FLOAT_IMAGE(cellValueOutImg, cellIndex, 0.0);
}

void main()
{
  if (any(greaterThanEqual(gl_GlobalInvocationID.xyz, uvec3(currentGridSize)))) {
    return;
  }
  ivec3 cell = ivec3(gl_GlobalInvocationID.xyz);
  int cellIndex = getCellIndex(cell, currentGridSize);
  vec3 cellSize = (aabbMax.xyz - aabbMin.xyz) / float(currentGridSize);
  vec3 cellCenter = aabbMin.xyz + cellSize * (vec3(cell) + 0.5);
  computePruning(cellCenter, cellSize, cellIndex);
}
""" % (runtime.PRIMITIVE_TEXELS, runtime.MAX_STACK)
