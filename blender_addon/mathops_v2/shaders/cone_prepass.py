from . import raymarch


LOCAL_GROUP_SIZE = (8, 8, 1)


COMPUTE_SOURCE = raymarch.COMMON_SOURCE + """\
const int CONE_TILE_SIZE = 8;

float coneDistanceEstimate(vec3 worldPoint, out bool activeCell)
{
  int cellIndex = pruningCellIndex(worldPoint);
  int activeCount = int(loadScalar(pruneCellCounts, cellIndex) + 0.5);
  activeCell = activeCount > 0;
  if (activeCell) {
    bool nearField;
    return evalActiveSceneForCell(worldPoint, cellIndex, nearField);
  }
  return loadScalar(pruneCellErrors, cellIndex);
}

float orthographicTileSpan(vec2 uv, vec2 screenSize, vec3 rayOrigin)
{
  vec2 uvOffsetX = clamp(uv + vec2(float(CONE_TILE_SIZE) / max(screenSize.x, 1.0), 0.0), vec2(0.0), vec2(1.0));
  vec2 uvOffsetY = clamp(uv + vec2(0.0, float(CONE_TILE_SIZE) / max(screenSize.y, 1.0)), vec2(0.0), vec2(1.0));
  vec3 rayOriginX;
  vec3 rayDirectionX;
  vec3 rayOriginY;
  vec3 rayDirectionY;
  computeRay(uvOffsetX, rayOriginX, rayDirectionX);
  computeRay(uvOffsetY, rayOriginY, rayDirectionY);
  return max(length(rayOriginX - rayOrigin), length(rayOriginY - rayOrigin));
}

void main()
{
  ivec2 tileCoord = ivec2(gl_GlobalInvocationID.xy);
  ivec2 tileDims = imageSize(coneTileHints);
  if (tileCoord.x >= tileDims.x || tileCoord.y >= tileDims.y) {
    return;
  }

  if (primitiveCountValue() <= 0 || instructionCountValue() <= 0 || !pruningEnabledValue()) {
    imageStore(coneTileHints, tileCoord, vec4(0.0, 0.0, 0.0, -1.0));
    return;
  }

  vec2 screenSize = max(screenSizePx, vec2(1.0));
  vec2 tileCenter = min((vec2(tileCoord) + 0.5) * float(CONE_TILE_SIZE), screenSize - vec2(0.5));
  vec2 uv = tileCenter / screenSize;

  vec3 rayOrigin;
  vec3 rayDirection;
  computeRay(uv, rayOrigin, rayDirection);

  vec3 boundsMin = pruningBoundsMinValue();
  vec3 boundsMax = pruningBoundsMaxValue();
  float tEnter = 0.0;
  float tExit = 0.0;
  if (!bboxIntersect(boundsMin, boundsMax, rayOrigin, rayDirection, tEnter, tExit)) {
    imageStore(coneTileHints, tileCoord, vec4(0.0, 0.0, 0.0, -1.0));
    return;
  }

  float coneEpsilon = max(surfaceEpsilonValue() * 32.0, 1e-4);
  float tanHalfTile = float(CONE_TILE_SIZE) / max(screenSize.y, 1.0);
  float orthoTileWorld = orthographicViewValue() ? orthographicTileSpan(uv, screenSize, rayOrigin) : 0.0;
  float orthoConeRadius = orthoTileWorld * coneAperture;

  float t = tEnter;
  float tSafe = tEnter;
  float dSafe = 0.0;
  float coneRSafe = 0.0;
  int maxConeSteps = max(coneSteps, 1);
  for (int step = 0; step < maxConeSteps; step++) {
    if (step >= 512 || t > tExit) {
      break;
    }
    vec3 pos = rayOrigin + rayDirection * t;
    bool activeCell = false;
    float dist = coneDistanceEstimate(pos, activeCell);
    float coneRadius = orthographicViewValue() ? orthoConeRadius : t * tanHalfTile * coneAperture;
    if (dist <= coneRadius) {
      float margin = orthographicViewValue() ? orthoTileWorld * 3.0 : coneRSafe * 3.0;
      float tSkip = max(tSafe + max(dSafe - margin, 0.0), tEnter);
      imageStore(coneTileHints, tileCoord, vec4(rayOrigin + rayDirection * tSkip, tSkip));
      return;
    }
    float stepDistance = activeCell ? max(dist, coneEpsilon) : max(dist - coneRadius, coneEpsilon);
    tSafe = t;
    dSafe = dist;
    coneRSafe = coneRadius;
    t += stepDistance;
  }

  if (t >= tExit && tSafe > tEnter) {
    imageStore(coneTileHints, tileCoord, vec4(0.0, 0.0, 0.0, -1.0));
    return;
  }

  if (tSafe > tEnter) {
    float margin = orthographicViewValue() ? orthoTileWorld * 3.0 : coneRSafe * 3.0;
    float tSkip = max(tSafe + max(dSafe - margin, 0.0), tEnter);
    imageStore(coneTileHints, tileCoord, vec4(rayOrigin + rayDirection * tSkip, tSkip));
    return;
  }

  imageStore(coneTileHints, tileCoord, vec4(rayOrigin + rayDirection * tEnter, 0.0));
}
"""
