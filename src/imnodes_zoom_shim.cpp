#include "imnodes_zoom_shim.h"

namespace ImNodes {

namespace {

float g_editor_zoom = 1.0f;

}

float EditorContextGetZoom() {
    return g_editor_zoom;
}

void EditorContextSetZoom(float zoom, const ImVec2&) {
    if (zoom < 0.2f) zoom = 0.2f;
    if (zoom > 5.0f) zoom = 5.0f;
    g_editor_zoom = zoom;
}

}
