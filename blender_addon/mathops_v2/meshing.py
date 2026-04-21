import importlib


_NATIVE_MODULE = None
_NATIVE_MODULE_RESOLVED = False


def _native_module():
    global _NATIVE_MODULE
    global _NATIVE_MODULE_RESOLVED
    if _NATIVE_MODULE_RESOLVED:
        return _NATIVE_MODULE

    _NATIVE_MODULE_RESOLVED = True
    root_package = __package__.split(".", 1)[0]
    module_names = (
        f"{root_package}.mathops_meshing_native",
        "mathops_meshing_native",
    )
    for module_name in module_names:
        try:
            _NATIVE_MODULE = importlib.import_module(module_name)
            return _NATIVE_MODULE
        except Exception:
            continue
    return None


def extract_dual_contour_mesh(compiled, resolution):
    native_module = _native_module()
    if native_module is None or not hasattr(native_module, "extract_dual_contour_mesh"):
        raise RuntimeError("MathOPS CPU meshing module is not available; rebuild the addon native modules")

    bounds_min, bounds_max = compiled.get("render_bounds") or compiled.get("scene_bounds") or ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    return native_module.extract_dual_contour_mesh(
        compiled.get("primitive_rows", ()),
        compiled.get("polygon_rows", ()),
        compiled.get("warp_rows", ()),
        compiled.get("instruction_rows", ()),
        bounds_min,
        bounds_max,
        int(resolution),
    )


def extract_iso_simplex_mesh(compiled, resolution):
    native_module = _native_module()
    if native_module is None or not hasattr(native_module, "extract_iso_simplex_mesh"):
        raise RuntimeError("MathOPS CPU meshing module is not available; rebuild the addon native modules")

    bounds_min, bounds_max = compiled.get("render_bounds") or compiled.get("scene_bounds") or ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    return native_module.extract_iso_simplex_mesh(
        compiled.get("primitive_rows", ()),
        compiled.get("polygon_rows", ()),
        compiled.get("warp_rows", ()),
        compiled.get("instruction_rows", ()),
        bounds_min,
        bounds_max,
        int(resolution),
    )
