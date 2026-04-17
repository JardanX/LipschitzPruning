from .. import runtime


UNIFORMS_SOURCE = """
struct MathOPSV2ViewParams {
  vec4 cameraPosition;
  vec4 lightDirectionSurfaceEpsilon;
  vec4 distanceAndCounts;
  vec4 pruningBoundsMinGridSize;
  vec4 pruningBoundsMaxEnabled;
  vec4 debugModePadding;
};
"""


VERTEX_SOURCE = """
void main()
{
  uvInterp = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}
"""


FRAGMENT_SOURCE = """\
#define PRIMITIVE_STRIDE %d
#define MAX_STACK %d

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
  return mathops.lightDirectionSurfaceEpsilon.w;
}

float maxDistanceValue()
{
  return mathops.distanceAndCounts.x;
}

int maxStepsValue()
{
  return int(mathops.distanceAndCounts.y + 0.5);
}

int primitiveCountValue()
{
  return int(mathops.distanceAndCounts.z + 0.5);
}

int instructionCountValue()
{
  return int(mathops.distanceAndCounts.w + 0.5);
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

int debugModeValue()
{
  return int(mathops.debugModePadding.x + 0.5);
}

float debugVizMaxValue()
{
  return max(mathops.debugModePadding.y, 1.0);
}

vec3 toLocalPoint(int primitiveIndex, vec3 worldPoint)
{
  int baseRow = primitiveIndex * PRIMITIVE_STRIDE;
  vec4 world = vec4(worldPoint, 1.0);
  vec4 row0 = loadRow(baseRow + 1);
  vec4 row1 = loadRow(baseRow + 2);
  vec4 row2 = loadRow(baseRow + 3);
  return vec3(dot(row0, world), dot(row1, world), dot(row2, world));
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
  vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, halfHeight);
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
  return mix(b, -a, h) + k * h * (1.0 - h);
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
  int primitiveType = int(meta.x + 0.5);
  if (primitiveType == 0) {
    return sdEllipsoid(localPoint, vec3(meta.y) * scale);
  }
  if (primitiveType == 1) {
    return sdBox(localPoint, meta.yzw * scale);
  }
  if (primitiveType == 2) {
    float radialScale = max(0.5 * (scale.x + scale.z), 1e-6);
    return sdCylinder(localPoint, meta.y * radialScale, meta.z * scale.y);
  }
  if (primitiveType == 3) {
    float radialScale = max(0.5 * (scale.x + scale.z), 1e-6);
    float minorScale = max(min(radialScale, scale.y), 1e-6);
    return sdTorus(localPoint, vec2(meta.y * radialScale, meta.z * minorScale));
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

bool bboxIntersect(vec3 boxMin, vec3 boxMax, vec3 rayOrigin, vec3 rayDirection, out float travel)
{
  vec3 tbot = (boxMin - rayOrigin) / rayDirection;
  vec3 ttop = (boxMax - rayOrigin) / rayDirection;
  vec3 tmin = min(ttop, tbot);
  vec3 tmax = max(ttop, tbot);
  vec2 t0v = max(tmin.xx, tmin.yz);
  float t0 = max(t0v.x, t0v.y);
  vec2 t1v = min(tmax.xx, tmax.yz);
  float t1 = min(t1v.x, t1v.y);
  travel = max(t0, 0.0);
  return t1 > max(t0, 0.0);
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
  int instructionBase = primitiveCountValue() * PRIMITIVE_STRIDE;
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
  int instructionBase = primitiveCountValue() * PRIMITIVE_STRIDE;
  return evalInstructionRange(worldPoint, instructionBase, instructionCountValue());
}

vec3 estimateNormal(vec3 worldPoint)
{
  float eps = max(surfaceEpsilonValue() * 2.0, 0.0005);
  return normalize(vec3(
    evalScene(worldPoint + vec3(eps, 0.0, 0.0)) - evalScene(worldPoint - vec3(eps, 0.0, 0.0)),
    evalScene(worldPoint + vec3(0.0, eps, 0.0)) - evalScene(worldPoint - vec3(0.0, eps, 0.0)),
    evalScene(worldPoint + vec3(0.0, 0.0, eps)) - evalScene(worldPoint - vec3(0.0, 0.0, eps))
  ));
}

vec3 backgroundColor(vec3 rayDirection)
{
  float t = clamp(0.5 * rayDirection.y + 0.5, 0.0, 1.0);
  return mix(vec3(0.02, 0.03, 0.05), vec3(0.12, 0.14, 0.18), t);
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

vec3 computeRayDirection(vec2 uv)
{
  vec2 ndc = uv * 2.0 - 1.0;
  vec4 nearPoint = invViewProjectionMatrix * vec4(ndc, -1.0, 1.0);
  vec4 farPoint = invViewProjectionMatrix * vec4(ndc, 1.0, 1.0);
  nearPoint.xyz /= nearPoint.w;
  farPoint.xyz /= farPoint.w;
  return normalize(farPoint.xyz - nearPoint.xyz);
}

void main()
{
  vec3 rayOrigin = mathops.cameraPosition.xyz;
  vec3 rayDirection = computeRayDirection(uvInterp);
  if (primitiveCountValue() <= 0 || instructionCountValue() <= 0) {
    FragColor = vec4(backgroundColor(rayDirection), 1.0);
    return;
  }

  vec3 boundsMin = pruningBoundsMinValue();
  vec3 boundsMax = pruningBoundsMaxValue();
  float travel = 0.0;
  if (!bboxIntersect(boundsMin, boundsMax, rayOrigin, rayDirection, travel)) {
    FragColor = vec4(backgroundColor(rayDirection), 1.0);
    return;
  }
  travel += 1e-4;
  bool hit = false;
  vec3 worldPoint = rayOrigin;
  int maxSteps = maxStepsValue();
  for (int step = 0; step < maxSteps; step++) {
    if (step >= 4096) {
      break;
    }
    worldPoint = rayOrigin + rayDirection * travel;
    if (any(lessThan(worldPoint, boundsMin)) || any(greaterThanEqual(worldPoint, boundsMax))) {
      hit = false;
      break;
    }
    float dist = evalScene(worldPoint);
    if (dist < surfaceEpsilonValue()) {
      hit = true;
      break;
    }
    travel += abs(dist);
    if (travel > maxDistanceValue()) {
      break;
    }
  }

  if (!hit) {
    FragColor = vec4(backgroundColor(rayDirection), 1.0);
    return;
  }

  vec3 normal = estimateNormal(worldPoint);
  vec3 light = normalize(mathops.lightDirectionSurfaceEpsilon.xyz);
  float diffuse = max(dot(normal, light), 0.0);
  float rim = pow(max(1.0 - max(dot(normal, -rayDirection), 0.0), 0.0), 2.0);
  vec3 baseColor = vec3(0.78, 0.82, 0.90);
  vec3 color = (0.12 + diffuse) * baseColor + rim * vec3(0.12, 0.18, 0.28);
  if (debugModeValue() != 0) {
    color = pruningDebugColor(worldPoint, color);
    FragColor = vec4(color, 1.0);
    return;
  }
  float fog = 1.0 - exp(-0.02 * travel);
  color = mix(color, backgroundColor(rayDirection), clamp(fog, 0.0, 1.0));
  FragColor = vec4(color, 1.0);
}
""" % (runtime.PRIMITIVE_TEXELS, runtime.MAX_STACK)
