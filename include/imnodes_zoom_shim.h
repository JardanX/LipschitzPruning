#ifndef LIPSCHITZ_IMNODES_ZOOM_SHIM_H
#define LIPSCHITZ_IMNODES_ZOOM_SHIM_H

struct ImVec2;

namespace ImNodes {

float EditorContextGetZoom();
void EditorContextSetZoom(float zoom, const ImVec2& focus);

}

#endif
