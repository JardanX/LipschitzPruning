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
#define WARP_ROWS_PER_ENTRY 4
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

vec3 worldToArrayLocal(vec3 worldPoint, vec3 origin, vec4 rotation)
{
  return rotateByQuaternion(vec4(-rotation.xyz, rotation.w), worldPoint - origin);
}

vec3 arrayLocalToWorld(vec3 localPoint, vec3 origin, vec4 rotation)
{
  return origin + rotateByQuaternion(rotation, localPoint);
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
  bool mirrored = fract(abs(id) * 0.5) > 0.25;
  bool odd = (count %% 2) != 0;
  bool atDefect = odd && (abs(angleRel) > WARP_PI - sector * 0.5);
  if (atDefect) {
    local = abs(local);
  }
  else if (mirrored) {
    local = -local;
  }
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

vec3 toLocalPoint(int primitiveIndex, vec3 worldPoint)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 row0 = loadRow(baseRow + 1);
  vec4 row1 = loadRow(baseRow + 2);
  vec4 row2 = loadRow(baseRow + 3);
  int packedWarpInfo = int(loadRow(baseRow + 4).w + 0.5);
  int warpBase = primitiveCountValue() * PRIMITIVE_STRIDE;
  int warpOffset = packedWarpInfo / MIRROR_WARP_PACK_SCALE;
  int warpCount = packedWarpInfo - warpOffset * MIRROR_WARP_PACK_SCALE;
  vec3 warpedPoint = worldPoint;
  for (int index = 0; index < warpCount; index++) {
    int warpRow = warpBase + warpOffset + index * WARP_ROWS_PER_ENTRY;
    vec4 warp0 = loadRow(warpRow);
    vec4 warp1 = loadRow(warpRow + 1);
    vec4 warp2 = loadRow(warpRow + 2);
    vec4 warp3 = loadRow(warpRow + 3);
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
      vec3 primitiveCenter = primitiveLocalToWorld(row0, row1, row2, vec3(0.0));
      vec3 origin = warp2.xyz;
      vec4 rotation = unpackWarpRotation(warp3.xyz);
      vec3 arrayPoint = worldToArrayLocal(warpedPoint, origin, rotation);
      arrayPoint.x = foldFiniteGridAxis(arrayPoint.x, primitiveCenter.x, primitiveCenter.x, spacing.x, counts.x, blend);
      arrayPoint.y = foldFiniteGridAxis(arrayPoint.y, primitiveCenter.y, primitiveCenter.y, spacing.y, counts.y, blend);
      arrayPoint.z = foldFiniteGridAxis(arrayPoint.z, primitiveCenter.z, primitiveCenter.z, spacing.z, counts.z, blend);
      warpedPoint = arrayLocalToWorld(arrayPoint, origin, rotation);
      continue;
    }
    if (warpKind == WARP_KIND_RADIAL) {
      int count = int(warp0.y + 0.5);
      float blend = max(warp0.z, 0.0);
      float radius = max(warp0.w, 0.0);
      vec3 primitiveCenter = primitiveLocalToWorld(row0, row1, row2, vec3(0.0));
      vec3 repeatOrigin = warp1.xyz;
      vec3 fieldOrigin = warp2.xyz;
      vec4 rotation = unpackWarpRotation(warp3.xyz);
      vec3 arrayPoint = worldToArrayLocal(warpedPoint, fieldOrigin, rotation);
      arrayPoint = applyRadialArray(arrayPoint, repeatOrigin, primitiveCenter, radius, count, blend);
      warpedPoint = arrayLocalToWorld(arrayPoint, fieldOrigin, rotation);
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

float sdCylinder(vec3 p, float r, float halfHeight)
{
  vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, halfHeight);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}

float sdTorus(vec3 p, vec2 t)
{
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
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

float evalPrimitive(int primitiveIndex, vec3 worldPoint)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 meta = loadRow(baseRow);
  vec3 localPoint = toLocalPoint(primitiveIndex, worldPoint);
  vec3 scale = primitiveScale(primitiveIndex);
  float minScale = max(min(scale.x, min(scale.y, scale.z)), 1e-6);
  int primitiveType = int(meta.x + 0.5);
  if (primitiveType == 0) {
    return sdEllipsoid(localPoint, vec3(meta.y) * scale);
  }
  if (primitiveType == 1) {
    return sdBox(localPoint, meta.yzw * scale);
  }
  if (primitiveType == 2) {
    return sdCylinder(localPoint / scale, meta.y, meta.z) * minScale;
  }
  if (primitiveType == 3) {
    return sdTorus(localPoint / scale, vec2(meta.y, meta.z)) * minScale;
  }
  return 1e20;
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

float evalSceneForNormal(vec3 worldPoint)
{
  if (pruningEnabledValue()) {
    vec3 boundsMin = pruningBoundsMinValue();
    vec3 boundsMax = pruningBoundsMaxValue();
    if (all(greaterThanEqual(worldPoint, boundsMin)) && all(lessThan(worldPoint, boundsMax))) {
      int cellIndex = pruningCellIndex(worldPoint);
      bool nearField;
      return evalActiveSceneForCell(worldPoint, cellIndex, nearField);
    }
  }
  int instructionBase = instructionBaseValue();
  return evalInstructionRange(worldPoint, instructionBase, instructionCountValue());
}

vec3 estimateTetrahedronNormal(vec3 worldPoint, vec3 rayDirection)
{
  float h = max(surfaceEpsilonValue() * 2.0, 0.0005);
  const vec2 k = vec2(1.0, -1.0);
  vec3 normal =
    k.xyy * evalSceneForNormal(worldPoint + k.xyy * h) +
    k.yyx * evalSceneForNormal(worldPoint + k.yyx * h) +
    k.yxy * evalSceneForNormal(worldPoint + k.yxy * h) +
    k.xxx * evalSceneForNormal(worldPoint + k.xxx * h);
  float lenSq = dot(normal, normal);
  if (!(lenSq > 1e-12)) {
    return -rayDirection;
  }
  normal *= inversesqrt(lenSq);
  if (dot(normal, -rayDirection) < 0.0) {
    normal = -normal;
  }
  return normal;
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
  vec3 rayOrigin;
  vec3 rayDirection;
  computeRay(uvInterp, rayOrigin, rayDirection);
  if (primitiveCountValue() <= 0 || instructionCountValue() <= 0) {
    vec3 color = (debugModeValue() == 3) ? vec3(0.0) : backgroundColor(rayDirection);
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
    float dist = evalScene(worldPoint);
    if (dist < surfaceEpsilonValue()) {
      hit = true;
      if (pruningEnabledValue()) {
        hitCellIndex = pruningCellIndex(worldPoint);
      }
      break;
    }
    float stepDistance = abs(dist);
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
  vec3 normal = disableSurfaceShadingValue() ? vec3(0.0, 0.0, 1.0) : estimateTetrahedronNormal(worldPoint, rayDirection);
  vec3 color = shadeSurface(normal, rayDirection);
  if (debugModeValue() != 0) {
    color = pruningDebugColor(worldPoint, color);
    FragColor = vec4(pow(color, vec3(gammaValue())), 1.0);
    gl_FragDepth = surfaceDepth(worldPoint);
    return;
  }
  FragColor = vec4(pow(color, vec3(gammaValue())), 1.0);
  gl_FragDepth = surfaceDepth(worldPoint);
}
"""
