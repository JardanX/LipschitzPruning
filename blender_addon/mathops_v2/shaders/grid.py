UNIFORMS_SOURCE = """
struct MathOPSV2GridParams {
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  mat4 viewMatrix;
  mat4 projectionMatrix;
  vec4 cameraPosition;
  vec4 resolution;
  vec4 gridColor;
  vec4 gridEmphasisColor;
  vec4 xAxisColor;
  vec4 yAxisColor;
  vec4 zAxisColor;
  float maxDistance;
  float gridScale;
  float gridLevelFrac;
  float isAxisAligned;
  int subdivisions;
  int isOrthographic;
  int gridType;
  int gridFlag;
  vec4 _pad3;
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
#define SHOW_GRID 1
#define SHOW_AXES 2
#define AXIS_X 4
#define AXIS_Y 8
#define AXIS_Z 16
#define PLANE_XY 32
#define PLANE_XZ 64
#define PLANE_YZ 128
#define flag_test(flag, val) ((flag & val) != 0)
#define saturate(x) clamp(x, 0.0, 1.0)
#define pow3f(x) ((x) * (x) * (x))
#define M_1_SQRTPI 0.5641895835477563
#define DISC_RADIUS (M_1_SQRTPI * 1.05)
#define HALF_PI 1.57079632679
#define CLEAR_DEPTH_THRESHOLD 0.99998

float rayPlaneIntersect(vec3 ro, vec3 rd, vec3 pn, float pd)
{
  float d = dot(rd, pn);
  if (abs(d) < 1e-6) {
    return -1.0;
  }
  float t = -(dot(ro, pn) + pd) / d;
  return t > 0.0 ? t : -1.0;
}

float getGridIntensity(vec2 gp, float ss, float dU, float dV)
{
  vec2 uv = fract(gp / ss);
  vec2 uvD = vec2(dU, dV) / ss;
  vec2 lw = uvD * 1.5;
  vec2 dw = clamp(lw, uvD, vec2(0.5));
  vec2 la = uvD * 1.5;
  vec2 guv = 1.0 - abs(uv * 2.0 - 1.0);
  vec2 ln = smoothstep(dw + la, dw - la, guv) * saturate(lw / dw);
  ln = mix(ln, lw, saturate(uvD * 2.0 - 1.0));
  float raw = clamp(mix(ln.x, 1.0, ln.y), 0.0, 1.0);
  float spFade = 1.0 - smoothstep(0.5, 2.0, max(uvD.x, uvD.y));
  return raw * spFade;
}

float getDistanceFade(float d, bool p)
{
  return p ? 1.0 - smoothstep(50.0, 100.0, d) : 1.0;
}

float getRadialSpokeIntensity(vec2 gp, int sub)
{
  float dist = length(gp);
  if (dist < 0.0001) {
    return 0.0;
  }
  vec2 ap = abs(gp);
  float qa = atan(ap.y, ap.x);
  int spq = max(1, sub);
  float ss = HALF_PI / float(spq + 1);
  float ni = clamp(floor(qa / ss + 0.5), 1.0, float(spq));
  float ad = abs(qa - ni * ss);
  float ag = max(length(vec2(dFdx(qa), dFdy(qa))), 0.000001);
  return 1.0 - smoothstep(0.0, 1.0, ad / ag);
}

void getRadialGridIntensities(vec2 gp, int sub, float rs, float dU, float dV, float cd, bool isP,
                              out float maj, out float mnr, out float spk)
{
  float dist = length(gp);
  if (dist < 0.0001) {
    maj = 0.0;
    mnr = 0.0;
    spk = 0.0;
    return;
  }
  float av = (dU + dV) * 0.5;
  float la = 1.5;
  float cb = isP ? mix(1.0, 0.2, smoothstep(50.0, 150.0, cd)) : 1.0;
  float msf = smoothstep(1.5, 4.0, rs / av);
  float nmi = floor(dist / rs + 0.5);
  maj = smoothstep(la, 0.0, abs(dist - nmi * rs) / av) * msf * cb;
  mnr = 0.0;
  if (sub > 1) {
    float ms = rs / float(sub);
    float mnrsf = smoothstep(1.5, 4.0, ms / av);
    float nni = floor(dist / ms + 0.5);
    mnr = smoothstep(la, 0.0, abs(dist - nni * ms) / av) * mnrsf * cb;
    if (mod(nni, float(sub)) < 0.5) {
      mnr = 0.0;
    }
  }
  float sas = HALF_PI / float(sub + 1);
  float ada = av / dist;
  float ssf = smoothstep(1.5, 4.0, sas / max(ada, 0.000001));
  spk = getRadialSpokeIntensity(gp, sub) * ssf;
}

float getAxisLine(float d, float dv)
{
  return smoothstep(1.5, 0.0, abs(d) / max(dv, 1e-7));
}

float rayAxisDistance(vec3 ro, vec3 rd, int ax, out float ct)
{
  float dn;
  float nm;
  if (ax == 0) {
    dn = rd.y * rd.y + rd.z * rd.z;
    nm = -(ro.y * rd.y + ro.z * rd.z);
  }
  else if (ax == 1) {
    dn = rd.x * rd.x + rd.z * rd.z;
    nm = -(ro.x * rd.x + ro.z * rd.z);
  }
  else {
    dn = rd.x * rd.x + rd.y * rd.y;
    nm = -(ro.x * rd.x + ro.y * rd.y);
  }
  if (dn < 1e-10) {
    ct = 0.0;
    return ax == 0 ? length(ro.yz) : ax == 1 ? length(ro.xz) : length(ro.xy);
  }
  ct = nm / dn;
  if (ct < 0.0) {
    ct = 0.0;
    return ax == 0 ? length(ro.yz) : ax == 1 ? length(ro.xz) : length(ro.xy);
  }
  vec3 cp = ro + rd * ct;
  return ax == 0 ? length(cp.yz) : ax == 1 ? length(cp.xz) : length(cp.xy);
}

vec3 getPlaneNormal(int f)
{
  return flag_test(f, PLANE_XZ) ? vec3(0.0, 1.0, 0.0) : flag_test(f, PLANE_YZ) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
}

vec2 getPlaneCoords(vec3 point, int f)
{
  return flag_test(f, PLANE_XZ) ? point.xz : flag_test(f, PLANE_YZ) ? point.yz : point.xy;
}

void main()
{
  vec2 ndc = uvInterp * 2.0 - 1.0;
  vec3 cameraPos = params.cameraPosition.xyz;
  bool isPerspective = params.isOrthographic == 0;
  int gridFlag = int(params.gridFlag);

  vec3 rayOrigin;
  vec3 rayDir;
  if (isPerspective) {
    rayOrigin = cameraPos;
    vec4 targetView = params.projectionMatrixInverse * vec4(ndc, 1.0, 1.0);
    targetView /= targetView.w;
    rayDir = normalize((params.viewMatrixInverse * targetView).xyz - rayOrigin);
  }
  else {
    vec3 camRight = vec3(params.viewMatrixInverse[0]);
    vec3 camUp = vec3(params.viewMatrixInverse[1]);
    vec3 camForward = -vec3(params.viewMatrixInverse[2]);
    float halfWidth = params.projectionMatrixInverse[0][0];
    float halfHeight = params.projectionMatrixInverse[1][1];
    rayDir = normalize(camForward);
    rayOrigin = cameraPos + camRight * ndc.x * halfWidth + camUp * ndc.y * halfHeight - rayDir * params.maxDistance * 4.0;
  }

  int planeFlag = gridFlag & (PLANE_XY | PLANE_XZ | PLANE_YZ);
  if (planeFlag == 0) {
    planeFlag = PLANE_XY;
  }
  vec3 planeNormal = getPlaneNormal(planeFlag);
  float planeHit = rayPlaneIntersect(rayOrigin, rayDir, planeNormal, 0.0);
  bool hasHit = planeHit >= 0.0;
  vec3 worldPoint = hasHit ? (rayOrigin + rayDir * planeHit) : (rayOrigin + rayDir * 100.0);

  vec2 gridPos;
  vec3 gridWorldPoint = worldPoint;
  if (isPerspective) {
    gridPos = getPlaneCoords(worldPoint, planeFlag);
  }
  else {
    vec3 camRight = vec3(params.viewMatrixInverse[0]);
    vec3 camUp = vec3(params.viewMatrixInverse[1]);
    vec3 camForward = -vec3(params.viewMatrixInverse[2]);
    float halfWidth = params.projectionMatrixInverse[0][0];
    float halfHeight = params.projectionMatrixInverse[1][1];
    vec3 orthoOrigin = cameraPos + camRight * ndc.x * halfWidth + camUp * ndc.y * halfHeight;
    float dn = dot(camForward, planeNormal);
    if (abs(dn) > 1e-6) {
      vec3 orthoPlanePoint = orthoOrigin + camForward * (-dot(orthoOrigin, planeNormal) / dn);
      gridPos = getPlaneCoords(orthoPlanePoint, planeFlag);
      gridWorldPoint = orthoPlanePoint;
    }
    else {
      gridPos = getPlaneCoords(worldPoint, planeFlag);
    }
  }

  float gridDepth = 1.0;
  bool occluded = false;
  if (hasHit) {
    float sdfDepth = texture(sdfDepthTex, uvInterp).r;
    bool axisAligned = params.isAxisAligned > 0.5;
    if (isPerspective) {
      vec4 viewPoint = params.viewMatrix * vec4(worldPoint, 1.0);
      vec4 clipPoint = params.projectionMatrix * viewPoint;
      gridDepth = clamp((clipPoint.z / clipPoint.w) * 0.5 + 0.5, 0.0, 1.0);
      float sdfNdc = sdfDepth * 2.0 - 1.0;
      float sdfZ = params.projectionMatrixInverse[2][2] * sdfNdc + params.projectionMatrixInverse[3][2];
      float sdfW = params.projectionMatrixInverse[2][3] * sdfNdc + params.projectionMatrixInverse[3][3];
      occluded = (-sdfZ / max(sdfW, 1e-7) < -viewPoint.z + 0.05 && sdfDepth < CLEAR_DEPTH_THRESHOLD);
    }
    else {
      vec4 viewPoint = params.viewMatrix * vec4(gridWorldPoint, 1.0);
      vec4 clipPoint = params.projectionMatrix * viewPoint;
      gridDepth = clamp(clipPoint.z * 0.5 + 0.5, 0.0, 1.0);
      if (axisAligned) {
        occluded = sdfDepth < CLEAR_DEPTH_THRESHOLD;
      }
      else {
        float sdfNdc = sdfDepth * 2.0 - 1.0;
        float sdfZ = params.projectionMatrixInverse[2][2] * sdfNdc + params.projectionMatrixInverse[3][2];
        float sdfW = params.projectionMatrixInverse[2][3] * sdfNdc + params.projectionMatrixInverse[3][3];
        occluded = (-sdfZ / max(sdfW, 1e-7) < -viewPoint.z + 0.05 && sdfDepth < CLEAR_DEPTH_THRESHOLD);
      }
    }
  }

  vec2 gridDx = dFdx(gridPos);
  vec2 gridDy = dFdy(gridPos);
  float gridDU = length(vec2(gridDx.x, gridDy.x));
  float gridDV = length(vec2(gridDx.y, gridDy.y));
  vec3 pointDx = dFdx(worldPoint);
  vec3 pointDy = dFdy(worldPoint);
  float distX = length(vec2(pointDx.x, pointDy.x));
  float distY = length(vec2(pointDx.y, pointDy.y));
  float distZ = length(vec2(pointDx.z, pointDy.z));
  if (!isPerspective) {
    float halfWidth = params.projectionMatrixInverse[0][0];
    float halfHeight = params.projectionMatrixInverse[1][1];
    vec2 res = params.resolution.xy;
    float worldPixelX = (2.0 * halfWidth) / res.x;
    float worldPixelY = (2.0 * halfHeight) / res.y;
    vec3 camRight = normalize(vec3(params.viewMatrixInverse[0]));
    vec3 camUp = normalize(vec3(params.viewMatrixInverse[1]));
    vec3 camForward = -normalize(vec3(params.viewMatrixInverse[2]));
    float forwardDot = dot(camForward, planeNormal);
    float safeDot = sign(forwardDot) * max(abs(forwardDot), 0.05);
    vec3 planeRight = camRight - camForward * (dot(camRight, planeNormal) / safeDot);
    vec3 planeUp = camUp - camForward * (dot(camUp, planeNormal) / safeDot);
    vec3 worldDx = planeRight * worldPixelX;
    vec3 worldDy = planeUp * worldPixelY;
    if (flag_test(planeFlag, PLANE_XY)) {
      gridDU = length(vec2(worldDx.x, worldDy.x));
      gridDV = length(vec2(worldDx.y, worldDy.y));
      distX = gridDU;
      distY = gridDV;
      distZ = length(vec2(worldDx.z, worldDy.z));
    }
    else if (flag_test(planeFlag, PLANE_XZ)) {
      gridDU = length(vec2(worldDx.x, worldDy.x));
      gridDV = length(vec2(worldDx.z, worldDy.z));
      distX = gridDU;
      distZ = gridDV;
      distY = length(vec2(worldDx.y, worldDy.y));
    }
    else {
      gridDU = length(vec2(worldDx.y, worldDy.y));
      gridDV = length(vec2(worldDx.z, worldDy.z));
      distY = gridDU;
      distZ = gridDV;
      distX = length(vec2(worldDx.x, worldDy.x));
    }
  }

  gridDU = max(gridDU, 1e-6);
  gridDV = max(gridDV, 1e-6);
  distX = max(distX, 1e-6);
  distY = max(distY, 1e-6);
  distZ = max(distZ, 1e-6);

  int subdivisions = max(1, params.subdivisions);
  float clipEnd = params.maxDistance;
  float gridFade = 1.0;
  if (hasHit) {
    vec3 viewDir = isPerspective ? normalize(cameraPos - worldPoint) : -vec3(params.viewMatrix[0][2], params.viewMatrix[1][2], params.viewMatrix[2][2]);
    float planeDot = flag_test(planeFlag, PLANE_XY) ? abs(viewDir.z) : flag_test(planeFlag, PLANE_XZ) ? abs(viewDir.y) : abs(viewDir.x);
    gridFade *= 1.0 - pow3f(1.0 - planeDot);
    if (isPerspective) {
      float viewDistance = length(cameraPos - worldPoint);
      gridFade *= smoothstep(min(clipEnd * 0.4, 400.0), 0.0, viewDistance);
      gridFade *= 1.0 - smoothstep(0.5 * clipEnd, clipEnd, planeHit);
    }
  }

  float silhouetteFade = 1.0;
  if (hasHit && !occluded && texture(sdfDepthTex, uvInterp).r > CLEAR_DEPTH_THRESHOLD) {
    vec2 texelSize = 1.0 / params.resolution.xy;
    float neighbors = 0.0;
    if (texture(sdfDepthTex, uvInterp + vec2(texelSize.x, 0.0)).r < CLEAR_DEPTH_THRESHOLD) {
      neighbors += 1.0;
    }
    if (texture(sdfDepthTex, uvInterp - vec2(texelSize.x, 0.0)).r < CLEAR_DEPTH_THRESHOLD) {
      neighbors += 1.0;
    }
    if (texture(sdfDepthTex, uvInterp + vec2(0.0, texelSize.y)).r < CLEAR_DEPTH_THRESHOLD) {
      neighbors += 1.0;
    }
    if (texture(sdfDepthTex, uvInterp - vec2(0.0, texelSize.y)).r < CLEAR_DEPTH_THRESHOLD) {
      neighbors += 1.0;
    }
    silhouetteFade = 1.0 - neighbors * 0.25;
  }
  gridFade *= silhouetteFade;

  vec4 outColor = vec4(0.0);
  float axisMask = 1.0;
  if (flag_test(gridFlag, SHOW_AXES)) {
    if (flag_test(planeFlag, PLANE_XY)) {
      if (flag_test(gridFlag, AXIS_X)) {
        axisMask *= smoothstep(0.0, 3.0, abs(worldPoint.y) / max(gridDV, 0.0001));
      }
      if (flag_test(gridFlag, AXIS_Y)) {
        axisMask *= smoothstep(0.0, 3.0, abs(worldPoint.x) / max(gridDU, 0.0001));
      }
    }
    else if (flag_test(planeFlag, PLANE_XZ)) {
      if (flag_test(gridFlag, AXIS_X)) {
        axisMask *= smoothstep(0.0, 3.0, abs(worldPoint.z) / max(gridDV, 0.0001));
      }
      if (flag_test(gridFlag, AXIS_Z)) {
        axisMask *= smoothstep(0.0, 3.0, abs(worldPoint.x) / max(gridDU, 0.0001));
      }
    }
    else if (flag_test(planeFlag, PLANE_YZ)) {
      if (flag_test(gridFlag, AXIS_Y)) {
        axisMask *= smoothstep(0.0, 3.0, abs(worldPoint.z) / max(gridDV, 0.0001));
      }
      if (flag_test(gridFlag, AXIS_Z)) {
        axisMask *= smoothstep(0.0, 3.0, abs(worldPoint.y) / max(gridDU, 0.0001));
      }
    }
  }

  if (hasHit && !occluded && flag_test(gridFlag, SHOW_GRID)) {
    float intensityAlpha = isPerspective ? 0.85 : 1.0;
    float levelFrac = params.gridLevelFrac;
    if (params.gridType == 1) {
      float radialScale = params.gridScale * 8.0;
      float cameraDistance = flag_test(planeFlag, PLANE_XY) ? length(cameraPos.xy) : flag_test(planeFlag, PLANE_XZ) ? length(cameraPos.xz) : length(cameraPos.yz);
      float major;
      float minor;
      float spokes;
      getRadialGridIntensities(gridPos, subdivisions, radialScale, gridDU, gridDV, cameraDistance, isPerspective, major, minor, spokes);
      if (spokes > 0.001) {
        float alpha = spokes * axisMask * params.gridColor.a * intensityAlpha * gridFade;
        outColor.rgb = mix(outColor.rgb, params.gridColor.rgb, alpha);
        outColor.a = max(outColor.a, alpha);
      }
      if (subdivisions > 1 && minor > 0.001) {
        float alpha = minor * axisMask * params.gridColor.a * intensityAlpha * saturate(1.0 - levelFrac) * gridFade;
        outColor.rgb = mix(outColor.rgb, params.gridColor.rgb, alpha);
        outColor.a = max(outColor.a, alpha);
      }
      if (major > 0.001) {
        float emphasis = saturate(1.0 - levelFrac);
        vec3 lineColor = mix(params.gridColor.rgb, params.gridEmphasisColor.rgb, emphasis);
        float lineAlpha = mix(params.gridColor.a, params.gridEmphasisColor.a, emphasis);
        float alpha = major * axisMask * lineAlpha * intensityAlpha * gridFade;
        outColor.rgb = mix(outColor.rgb, lineColor, alpha);
        outColor.a = max(outColor.a, alpha);
      }
    }
    else {
      float minorScale = params.gridScale / float(subdivisions);
      float distanceFade = length(cameraPos - worldPoint);
      if (subdivisions > 1) {
        float minor = getGridIntensity(gridPos, minorScale, gridDU, gridDV) * getDistanceFade(distanceFade, isPerspective) * axisMask;
        float minorAlpha = saturate(1.0 - levelFrac);
        if (minor > 0.001 && minorAlpha > 0.001) {
          float alpha = minor * params.gridColor.a * intensityAlpha * minorAlpha * gridFade;
          outColor.rgb = mix(outColor.rgb, params.gridColor.rgb, alpha);
          outColor.a = max(outColor.a, alpha);
        }
      }
      float major = getGridIntensity(gridPos, params.gridScale, gridDU, gridDV) * axisMask;
      if (major > 0.001) {
        float emphasis = saturate(1.0 - levelFrac);
        vec3 lineColor = mix(params.gridColor.rgb, params.gridEmphasisColor.rgb, emphasis);
        float lineAlpha = mix(params.gridColor.a, params.gridEmphasisColor.a, emphasis);
        float alpha = major * lineAlpha * intensityAlpha * gridFade;
        outColor.rgb = mix(outColor.rgb, lineColor, alpha);
        outColor.a = max(outColor.a, alpha);
      }
    }
  }

  if (hasHit && !occluded && flag_test(gridFlag, SHOW_AXES)) {
    float axisFade = gridFade;
    if (flag_test(planeFlag, PLANE_XY)) {
      if (flag_test(gridFlag, AXIS_X)) {
        float value = getAxisLine(gridWorldPoint.y, distY);
        if (value > 0.001) {
          float alpha = value * params.xAxisColor.a * axisFade;
          outColor.rgb = mix(outColor.rgb, params.xAxisColor.rgb, alpha);
          outColor.a = max(outColor.a, alpha);
        }
      }
      if (flag_test(gridFlag, AXIS_Y)) {
        float value = getAxisLine(gridWorldPoint.x, distX);
        if (value > 0.001) {
          float alpha = value * params.yAxisColor.a * axisFade;
          outColor.rgb = mix(outColor.rgb, params.yAxisColor.rgb, alpha);
          outColor.a = max(outColor.a, alpha);
        }
      }
    }
    else if (flag_test(planeFlag, PLANE_XZ)) {
      if (flag_test(gridFlag, AXIS_X)) {
        float value = getAxisLine(gridWorldPoint.z, distZ);
        if (value > 0.001) {
          float alpha = value * params.xAxisColor.a * axisFade;
          outColor.rgb = mix(outColor.rgb, params.xAxisColor.rgb, alpha);
          outColor.a = max(outColor.a, alpha);
        }
      }
      if (flag_test(gridFlag, AXIS_Z)) {
        float value = getAxisLine(gridWorldPoint.x, distX);
        if (value > 0.001) {
          float alpha = value * params.zAxisColor.a * axisFade;
          outColor.rgb = mix(outColor.rgb, params.zAxisColor.rgb, alpha);
          outColor.a = max(outColor.a, alpha);
        }
      }
    }
    else if (flag_test(planeFlag, PLANE_YZ)) {
      if (flag_test(gridFlag, AXIS_Y)) {
        float value = getAxisLine(gridWorldPoint.z, distZ);
        if (value > 0.001) {
          float alpha = value * params.yAxisColor.a * axisFade;
          outColor.rgb = mix(outColor.rgb, params.yAxisColor.rgb, alpha);
          outColor.a = max(outColor.a, alpha);
        }
      }
      if (flag_test(gridFlag, AXIS_Z)) {
        float value = getAxisLine(gridWorldPoint.y, distY);
        if (value > 0.001) {
          float alpha = value * params.zAxisColor.a * axisFade;
          outColor.rgb = mix(outColor.rgb, params.zAxisColor.rgb, alpha);
          outColor.a = max(outColor.a, alpha);
        }
      }
    }
  }

  int axisIndex = -1;
  vec4 axisColor = vec4(0.0);
  if (flag_test(planeFlag, PLANE_XY) && flag_test(gridFlag, AXIS_Z)) {
    axisIndex = 2;
    axisColor = params.zAxisColor;
  }
  else if (flag_test(planeFlag, PLANE_XZ) && flag_test(gridFlag, AXIS_Y)) {
    axisIndex = 1;
    axisColor = params.yAxisColor;
  }
  else if (flag_test(planeFlag, PLANE_YZ) && flag_test(gridFlag, AXIS_X)) {
    axisIndex = 0;
    axisColor = params.xAxisColor;
  }

  if (axisIndex >= 0 && flag_test(gridFlag, SHOW_AXES)) {
    vec3 axisDir = axisIndex == 0 ? vec3(1.0, 0.0, 0.0) : axisIndex == 1 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec4 axisOriginView = params.viewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    float axisOriginDepth = -axisOriginView.z;
    if (axisOriginDepth > 1e-5) {
      float axisExtent = max(1.0, min(clipEnd * 0.25, axisOriginDepth * 0.5));
      vec4 axisClipA = params.projectionMatrix * (params.viewMatrix * vec4(-axisDir * axisExtent, 1.0));
      vec4 axisClipB = params.projectionMatrix * (params.viewMatrix * vec4(axisDir * axisExtent, 1.0));
      if (abs(axisClipA.w) > 1e-7 && abs(axisClipB.w) > 1e-7) {
        vec2 screenSize = params.resolution.xy;
        vec2 axisScreenA = (axisClipA.xy / axisClipA.w) * 0.5 + 0.5;
        vec2 axisScreenB = (axisClipB.xy / axisClipB.w) * 0.5 + 0.5;
        vec2 axisLine = axisScreenB - axisScreenA;
        float axisLineLenSq = dot(axisLine, axisLine);
        if (axisLineLenSq > 1e-12) {
          float axisLineT = dot(uvInterp - axisScreenA, axisLine) / axisLineLenSq;
          vec2 closestAxisUv = axisScreenA + axisLine * axisLineT;
          float screenDistance = length((uvInterp - closestAxisUv) * screenSize);
          float axisIntensity = smoothstep(0.9, 0.0, screenDistance);
          if (axisIntensity > 0.001) {
            float closestT;
            rayAxisDistance(rayOrigin, rayDir, axisIndex, closestT);
            if (closestT > 0.0) {
              vec3 axisPoint = rayOrigin + rayDir * closestT;
              vec4 axisViewPoint = params.viewMatrix * vec4(axisPoint, 1.0);
              vec4 axisClip = params.projectionMatrix * axisViewPoint;
              float axisDepth = clamp((axisClip.z / axisClip.w) * 0.5 + 0.5, 0.0, 1.0);
              float sdfDepth = texture(sdfDepthTex, uvInterp).r;
              float axisViewDepth = -axisViewPoint.z;
              float sdfNdc = sdfDepth * 2.0 - 1.0;
              float sdfZ = params.projectionMatrixInverse[2][2] * sdfNdc + params.projectionMatrixInverse[3][2];
              float sdfW = params.projectionMatrixInverse[2][3] * sdfNdc + params.projectionMatrixInverse[3][3];
              bool axisOccluded = (-sdfZ / max(sdfW, 1e-7) < axisViewDepth - 0.05 && sdfDepth < CLEAR_DEPTH_THRESHOLD);
              if (!axisOccluded) {
                float fade = isPerspective ? 1.0 - smoothstep(0.5 * clipEnd, clipEnd, axisOriginDepth) : 1.0;
                vec3 cameraForward = normalize(-vec3(params.viewMatrixInverse[2]));
                float directionFade = axisIndex == 0 ? abs(cameraForward.x) : axisIndex == 1 ? abs(cameraForward.y) : abs(cameraForward.z);
                float alpha = axisIntensity * fade * (1.0 - smoothstep(0.995, 1.0, directionFade)) * axisColor.a;
                if (alpha > 0.001) {
                  outColor.rgb = mix(outColor.rgb, axisColor.rgb, alpha);
                  outColor.a = max(outColor.a, alpha);
                  gridDepth = min(gridDepth, max(0.0, axisDepth - 1e-6));
                }
              }
            }
          }
        }
      }
    }
  }

  if (outColor.a < 0.001) {
    discard;
  }
  FragColor = vec4(outColor.rgb, outColor.a);
  gl_FragDepth = clamp(gridDepth, 0.00001, 0.99999);
}
"""
