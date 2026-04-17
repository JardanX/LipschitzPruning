from .. import runtime


UNIFORMS_SOURCE = """
struct MathOPSV2ViewParams {
  vec4 cameraPosition;
  vec4 lightDirectionSurfaceEpsilon;
  vec4 distanceAndCounts;
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

float sdSphere(vec3 p, float r)
{
  return length(p) - r;
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

float evalScene(vec3 worldPoint)
{
  float stack[MAX_STACK];
  int stackPointer = 0;
  int instructionBase = primitiveCountValue() * PRIMITIVE_STRIDE;
  int instructionCount = instructionCountValue();
  for (int index = 0; index < instructionCount; index++) {
    vec4 instruction = loadRow(instructionBase + index);
    int kind = int(instruction.x + 0.5);
    if (kind == 0) {
      if (stackPointer >= MAX_STACK) {
        return 1e20;
      }
      stack[stackPointer] = evalPrimitive(int(instruction.y + 0.5), worldPoint);
      stackPointer++;
      continue;
    }
    if (stackPointer < 2) {
      return 1e20;
    }
    float rhs = stack[stackPointer - 1];
    float lhs = stack[stackPointer - 2];
    stackPointer -= 2;
    stack[stackPointer] = applyOp(kind, lhs, rhs, max(instruction.w, 0.0));
    stackPointer++;
  }
  if (stackPointer == 0) {
    return 1e20;
  }
  return stack[stackPointer - 1];
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

  float travel = 0.0;
  bool hit = false;
  vec3 worldPoint = rayOrigin;
  int maxSteps = maxStepsValue();
  for (int step = 0; step < maxSteps; step++) {
    if (step >= 4096) {
      break;
    }
    worldPoint = rayOrigin + rayDirection * travel;
    float dist = evalScene(worldPoint);
    if (dist < surfaceEpsilonValue()) {
      hit = true;
      break;
    }
    travel += dist;
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
  float fog = 1.0 - exp(-0.02 * travel);
  color = mix(color, backgroundColor(rayDirection), clamp(fog, 0.0, 1.0));
  FragColor = vec4(color, 1.0);
}
""" % (runtime.PRIMITIVE_TEXELS, runtime.MAX_STACK)
