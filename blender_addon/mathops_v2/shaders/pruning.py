from .. import runtime


LOCAL_GROUP_SIZE = (4, 4, 4)


COMPUTE_SOURCE = """\
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#define PRIMITIVE_STRIDE %d
#define STACK_DEPTH %d
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

ivec2 linearCoord(ivec2 size, int index)
{
  int width = max(size.x, 1);
  return ivec2(index %% width, index / width);
}

float loadScalar(sampler2D textureSampler, int index)
{
  return texelFetch(textureSampler, linearCoord(textureSize(textureSampler, 0), index), 0).x;
}

vec4 loadSceneRow(int row)
{
  return texelFetch(sceneData, ivec2(0, row), 0);
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
  vec4 row0 = loadSceneRow(baseRow + 1);
  vec4 row1 = loadSceneRow(baseRow + 2);
  vec4 row2 = loadSceneRow(baseRow + 3);
  int packedWarpInfo = int(loadSceneRow(baseRow + 4).w + 0.5);
  int warpBase = primitiveCount * PRIMITIVE_STRIDE;
  int warpOffset = packedWarpInfo / MIRROR_WARP_PACK_SCALE;
  int warpCount = packedWarpInfo - warpOffset * MIRROR_WARP_PACK_SCALE;
  vec3 warpedPoint = worldPoint;
  for (int index = 0; index < warpCount; index++) {
    int warpRow = warpBase + warpOffset + index * WARP_ROWS_PER_ENTRY;
    vec4 warp0 = loadSceneRow(warpRow);
    vec4 warp1 = loadSceneRow(warpRow + 1);
    vec4 warp2 = loadSceneRow(warpRow + 2);
    vec4 warp3 = loadSceneRow(warpRow + 3);
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

float evalPrimitive(int primitiveIndex, vec3 worldPoint)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 meta = loadSceneRow(baseRow);
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
