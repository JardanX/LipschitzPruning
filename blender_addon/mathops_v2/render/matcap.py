import os
import subprocess
import sys
import tempfile

import bpy
import gpu
import numpy as np


_MATCAP_TEXTURES = {}
_WHITE_TEXTURE = None
_BLACK_TEXTURE = None

_EXR_CORE = (
    "import os, OpenEXR, Imath, numpy as np\n"
    "def _read_rgba(exr, channels, height, width, prefix):\n"
    " rgba = np.zeros((height, width, 4), dtype=np.float32)\n"
    " for index, channel_name in enumerate('RGBA'):\n"
    "  source = f'{prefix}.{channel_name}' if prefix else channel_name\n"
    "  if source in channels:\n"
    "   rgba[:, :, index] = np.frombuffer(exr.channel(source, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape((height, width))\n"
    "  elif channel_name == 'A':\n"
    "   rgba[:, :, index] = 1.0\n"
    " return np.flip(rgba, 0).copy()\n"
    "def _open_exr(path):\n"
    " exr = OpenEXR.InputFile(path)\n"
    " header = exr.header()\n"
    " data_window = header['dataWindow']\n"
    " width = data_window.max.x - data_window.min.x + 1\n"
    " height = data_window.max.y - data_window.min.y + 1\n"
    " return exr, list(header['channels'].keys()), width, height\n"
    "def _extract(path, output_dir):\n"
    " exr, channels, width, height = _open_exr(path)\n"
    " has_diffuse = any(channel.startswith('diffuse.') for channel in channels)\n"
    " has_specular = any(channel.startswith('specular.') for channel in channels)\n"
    " _read_rgba(exr, channels, height, width, 'diffuse' if has_diffuse else '').tofile(os.path.join(output_dir, 'diffuse.bin'))\n"
    " if has_diffuse and has_specular:\n"
    "  _read_rgba(exr, channels, height, width, 'specular').tofile(os.path.join(output_dir, 'specular.bin'))\n"
    " print(f'{width},{height},{1 if has_diffuse and has_specular else 0}')\n"
)


def _free_texture(texture):
    if texture is None:
        return
    try:
        texture.free()
    except Exception:
        pass


def clear_cache():
    global _WHITE_TEXTURE, _BLACK_TEXTURE

    released = set()
    for diffuse_texture, specular_texture in _MATCAP_TEXTURES.values():
        for texture in (diffuse_texture, specular_texture):
            texture_id = id(texture)
            if texture_id in released:
                continue
            released.add(texture_id)
            _free_texture(texture)
    _MATCAP_TEXTURES.clear()

    for texture_name in ("_WHITE_TEXTURE", "_BLACK_TEXTURE"):
        texture = globals()[texture_name]
        texture_id = id(texture)
        if texture is not None and texture_id not in released:
            _free_texture(texture)
        globals()[texture_name] = None


def _configure_texture(texture):
    if texture is None:
        return None
    try:
        texture.filter_mode = "LINEAR"
        texture.extension = "EXTEND"
    except Exception:
        pass
    return texture


def _solid_texture(color):
    buffer = gpu.types.Buffer("FLOAT", 4, color)
    return _configure_texture(
        gpu.types.GPUTexture((1, 1), format="RGBA32F", data=buffer)
    )


def _white_texture():
    global _WHITE_TEXTURE
    if _WHITE_TEXTURE is None:
        _WHITE_TEXTURE = _solid_texture((1.0, 1.0, 1.0, 1.0))
    return _WHITE_TEXTURE


def _black_texture():
    global _BLACK_TEXTURE
    if _BLACK_TEXTURE is None:
        _BLACK_TEXTURE = _solid_texture((0.0, 0.0, 0.0, 1.0))
    return _BLACK_TEXTURE


def _matcap_studio_lights(context):
    if context is None:
        return []
    try:
        return [sl for sl in context.preferences.studio_lights if sl.type == "MATCAP"]
    except Exception:
        return []


def _studio_light_identifier(studio_light):
    return (
        os.path.basename(studio_light.path) if studio_light.path else studio_light.name
    )


def _find_studio_light(context, name):
    if not name or name == "NONE":
        return None
    for studio_light in _matcap_studio_lights(context):
        if _studio_light_identifier(studio_light) == name:
            return studio_light
        if studio_light.name == name:
            return studio_light
        if studio_light.path and studio_light.path.endswith(name):
            return studio_light
    return None


def get_matcaps_enum(_self, context):
    studio_lights = _matcap_studio_lights(context)
    if not studio_lights:
        return [("NONE", "None", "No matcaps available", 0, 0)]

    items = []
    for index, studio_light in enumerate(studio_lights):
        items.append(
            (
                _studio_light_identifier(studio_light),
                studio_light.name,
                f"Matcap: {studio_light.name}",
                bpy.types.UILayout.icon(studio_light),
                index,
            )
        )
    return items


def _space_shading(context):
    space = getattr(context, "space_data", None)
    return getattr(space, "shading", None)


def selected_matcap_name(context, preferred_name=""):
    studio_light = _find_studio_light(context, preferred_name)
    if studio_light is not None:
        return _studio_light_identifier(studio_light)

    shading = _space_shading(context)
    if shading is not None:
        studio_light = _find_studio_light(context, getattr(shading, "studio_light", ""))
        if studio_light is not None:
            return _studio_light_identifier(studio_light)

    studio_lights = _matcap_studio_lights(context)
    if studio_lights:
        return _studio_light_identifier(studio_lights[0])
    return ""


def sync_viewport_matcap(context, preferred_name):
    matcap_name = selected_matcap_name(context, preferred_name)
    if not matcap_name:
        return

    try:
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for space in area.spaces:
                    if space.type != "VIEW_3D":
                        continue
                    try:
                        if hasattr(space.shading, "studio_light"):
                            space.shading.studio_light = matcap_name
                    except Exception:
                        pass
                area.tag_redraw()
    except Exception:
        pass


def _texture_from_array(width, height, values):
    data = np.ascontiguousarray(values, dtype=np.float32)
    buffer = gpu.types.Buffer("FLOAT", len(data), data)
    return _configure_texture(
        gpu.types.GPUTexture((width, height), format="RGBA32F", data=buffer)
    )


def _load_exr_textures(path):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script = _EXR_CORE + (f"_extract({path!r}, {tmp_dir!r})\n")
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None

            parts = result.stdout.strip().split(",")
            if len(parts) < 3:
                return None

            width, height, has_specular = int(parts[0]), int(parts[1]), int(parts[2])
            diffuse_path = os.path.join(tmp_dir, "diffuse.bin")
            if not os.path.exists(diffuse_path):
                return None

            diffuse = np.fromfile(diffuse_path, dtype=np.float32)
            specular = None
            if has_specular:
                specular_path = os.path.join(tmp_dir, "specular.bin")
                if os.path.exists(specular_path):
                    specular = np.fromfile(specular_path, dtype=np.float32)

            diffuse_texture = _texture_from_array(width, height, diffuse)
            specular_texture = (
                _texture_from_array(width, height, specular)
                if specular is not None
                else _black_texture()
            )
            return diffuse_texture, specular_texture
    except Exception:
        return None


def _load_standard_texture(path):
    try:
        image = bpy.data.images.load(path, check_existing=True)
        return _configure_texture(gpu.texture.from_image(image)), _black_texture()
    except Exception:
        return None


def get_matcap_textures(context, preferred_name=""):
    matcap_name = selected_matcap_name(context, preferred_name)
    if not matcap_name:
        return _white_texture(), _black_texture()

    cached = _MATCAP_TEXTURES.get(matcap_name)
    if cached is not None:
        return cached

    studio_light = _find_studio_light(context, matcap_name)
    if studio_light is None or not studio_light.path:
        return _white_texture(), _black_texture()

    path = bpy.path.abspath(studio_light.path)
    textures = None
    if path.lower().endswith(".exr"):
        textures = _load_exr_textures(path)
    if textures is None:
        textures = _load_standard_texture(path)
    if textures is None:
        textures = (_white_texture(), _black_texture())

    _MATCAP_TEXTURES[matcap_name] = textures
    return textures
