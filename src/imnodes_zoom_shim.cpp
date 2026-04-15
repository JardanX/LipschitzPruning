#include "imnodes_zoom_shim.h"

#include "imnodes_internal.h"

namespace ImNodes {

float EditorContextGetZoom() {
    if (::GImNodes == nullptr || ::GImNodes->EditorCtx == nullptr) {
        return 1.0f;
    }
    return ::GImNodes->EditorCtx->Zoom;
}

void EditorContextSetZoom(float zoom, const ImVec2& focus) {
    if (zoom < 0.2f) zoom = 0.2f;
    if (zoom > 5.0f) zoom = 5.0f;
    if (::GImNodes == nullptr || ::GImNodes->EditorCtx == nullptr) {
        return;
    }

    ImNodesEditorContext& editor = *::GImNodes->EditorCtx;
    float old_zoom = editor.Zoom;
    if (old_zoom <= 0.0f) {
        old_zoom = 1.0f;
    }

    ImVec2 focus_editor = focus - ::GImNodes->CanvasOriginScreenSpace;
    ImVec2 focus_grid = (focus_editor - editor.Panning) / old_zoom;
    editor.Zoom = zoom;
    editor.Panning = focus_editor - focus_grid * editor.Zoom;
}

}
