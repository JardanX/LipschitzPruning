if(NOT DEFINED IMNODES_DIR)
    message(FATAL_ERROR "IMNODES_DIR is required")
endif()

set(imnodes_internal_h "${IMNODES_DIR}/imnodes_internal.h")
set(imnodes_cpp "${IMNODES_DIR}/imnodes.cpp")

function(replace_if_needed file_path old_text new_text marker)
    file(READ "${file_path}" file_text)
    string(FIND "${file_text}" "${marker}" marker_pos)
    if(marker_pos GREATER -1)
        return()
    endif()

    string(FIND "${file_text}" "${old_text}" old_pos)
    if(old_pos EQUAL -1)
        message(FATAL_ERROR "Patch anchor not found in ${file_path}: ${marker}")
    endif()

    string(REPLACE "${old_text}" "${new_text}" file_text "${file_text}")
    file(WRITE "${file_path}" "${file_text}")
endfunction()

replace_if_needed(
    "${imnodes_internal_h}"
    "    ImVec2 Panning;\n    ImVec2 AutoPanningDelta;"
    "    ImVec2 Panning;\n    float  Zoom;\n    ImVec2 AutoPanningDelta;"
    "float  Zoom;")

replace_if_needed(
    "${imnodes_internal_h}"
    "        : Nodes(), Pins(), Links(), Panning(0.f, 0.f), SelectedNodeIndices(), SelectedLinkIndices(),"
    "        : Nodes(), Pins(), Links(), Panning(0.f, 0.f), Zoom(1.0f), SelectedNodeIndices(), SelectedLinkIndices(),"
    "Zoom(1.0f)")

replace_if_needed(
    "${imnodes_cpp}"
    "    return v - GImNodes->CanvasOriginScreenSpace - editor.Panning;"
    "    return (v - GImNodes->CanvasOriginScreenSpace - editor.Panning) / editor.Zoom;"
    "    return (v - GImNodes->CanvasOriginScreenSpace - editor.Panning) / editor.Zoom;")

replace_if_needed(
    "${imnodes_cpp}"
    "    return v + GImNodes->CanvasOriginScreenSpace + editor.Panning;"
    "    return v * editor.Zoom + GImNodes->CanvasOriginScreenSpace + editor.Panning;"
    "v * editor.Zoom + GImNodes->CanvasOriginScreenSpace")

replace_if_needed(
    "${imnodes_cpp}"
    "    return v + editor.Panning;"
    "    return v * editor.Zoom + editor.Panning;"
    "v * editor.Zoom + editor.Panning")

replace_if_needed(
    "${imnodes_cpp}"
    "    return v - editor.Panning;"
    "    return (v - editor.Panning) / editor.Zoom;"
    "(v - editor.Panning) / editor.Zoom")

replace_if_needed(
    "${imnodes_cpp}"
    "    const ImVec2 offset = editor.Panning;\n    ImU32        line_color = GImNodes->Style.Colors[ImNodesCol_GridLine];"
    "    const ImVec2 offset = editor.Panning;\n    const float  spacing = GImNodes->Style.GridSpacing * editor.Zoom;\n    ImU32        line_color = GImNodes->Style.Colors[ImNodesCol_GridLine];"
    "const float  spacing = GImNodes->Style.GridSpacing * editor.Zoom;")

replace_if_needed(
    "${imnodes_cpp}"
    "    for (float x = fmodf(offset.x, GImNodes->Style.GridSpacing); x < canvas_size.x;\n         x += GImNodes->Style.GridSpacing)"
    "    for (float x = fmodf(offset.x, spacing); x < canvas_size.x; x += spacing)"
    "for (float x = fmodf(offset.x, spacing)")

replace_if_needed(
    "${imnodes_cpp}"
    "    for (float y = fmodf(offset.y, GImNodes->Style.GridSpacing); y < canvas_size.y;\n         y += GImNodes->Style.GridSpacing)"
    "    for (float y = fmodf(offset.y, spacing); y < canvas_size.y; y += spacing)"
    "for (float y = fmodf(offset.y, spacing)")

replace_if_needed(
    "${imnodes_cpp}"
    "    editor.PrimaryNodeOffset =\n        ref_origin + GImNodes->CanvasOriginScreenSpace + editor.Panning - GImNodes->MousePos;"
    "    editor.PrimaryNodeOffset = GridSpaceToScreenSpace(editor, ref_origin) - GImNodes->MousePos;"
    "GridSpaceToScreenSpace(editor, ref_origin)")

replace_if_needed(
    "${imnodes_cpp}"
    "        const ImVec2 origin = SnapOriginToGrid(\n            GImNodes->MousePos - GImNodes->CanvasOriginScreenSpace - editor.Panning +\n            editor.PrimaryNodeOffset);"
    "        const ImVec2 origin = SnapOriginToGrid(\n            ScreenSpaceToGridSpace(editor, GImNodes->MousePos + editor.PrimaryNodeOffset));"
    "ScreenSpaceToGridSpace(editor, GImNodes->MousePos + editor.PrimaryNodeOffset)")

replace_if_needed(
    "${imnodes_cpp}"
    "                node.Origin = origin + node_rel + editor.AutoPanningDelta;"
    "                node.Origin = origin + node_rel + editor.AutoPanningDelta / editor.Zoom;"
    "editor.AutoPanningDelta / editor.Zoom")

replace_if_needed(
    "${imnodes_cpp}"
    "    ImGui::SetCursorPos(node.Origin + editor.Panning);"
    "    ImGui::SetCursorPos(GridSpaceToEditorSpace(editor, node.Origin));"
    "GridSpaceToEditorSpace(editor, node.Origin)")

replace_if_needed(
    "${imnodes_cpp}"
    "        editor.Panning = ImFloor(center - target);"
    "        editor.Panning = ImFloor(center - target * editor.Zoom);"
    "center - target * editor.Zoom")

replace_if_needed(
    "${imnodes_cpp}"
    "    editor.Panning.x = -node.Origin.x;\n    editor.Panning.y = -node.Origin.y;"
    "    editor.Panning.x = -node.Origin.x * editor.Zoom;\n    editor.Panning.y = -node.Origin.y * editor.Zoom;"
    "editor.Panning.x = -node.Origin.x * editor.Zoom")

replace_if_needed(
    "${imnodes_cpp}"
    "    editor.GridContentBounds.Add(node.Origin + node.Rect.GetSize());"
    "    editor.GridContentBounds.Add(node.Origin + node.Rect.GetSize() / editor.Zoom);"
    "node.Rect.GetSize() / editor.Zoom")

replace_if_needed(
    "${imnodes_cpp}"
    "    return node.Rect.GetSize();"
    "    return node.Rect.GetSize() / editor.Zoom;"
    "return node.Rect.GetSize() / editor.Zoom")
