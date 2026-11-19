"""Blender Action Library for Blender AI Assistant.
Executed on Blender's main thread via bpy.app.timers.
"""
import math
try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

from action_validator import SUPPORTED_ACTIONS


def get_object(name: str):
    obj = bpy.data.objects.get(name)
    if not obj:
        raise ValueError(f"Object '{name}' not found in Blender scene.")
    return obj


def compute_world_bounds(obj):
    """Calculate the world-space axis-aligned bounding box for an object."""
    bpy.context.view_layer.update()
    mw = obj.matrix_world
    pts = [mw @ mathutils.Vector(corner) for corner in obj.bound_box]

    min_x = min(p.x for p in pts)
    max_x = max(p.x for p in pts)
    min_y = min(p.y for p in pts)
    max_y = max(p.y for p in pts)
    min_z = min(p.z for p in pts)
    max_z = max(p.z for p in pts)

    center = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0]
    dims = [max_x - min_x, max_y - min_y, max_z - min_z]

    return {
        "min": [round(min_x, 4), round(min_y, 4), round(min_z, 4)],
        "max": [round(max_x, 4), round(max_y, 4), round(max_z, 4)],
        "center": [round(c, 4) for c in center],
        "dimensions": [round(d, 4) for d in dims],
        "corners": [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in pts]
    }


def build_object_response(obj, action_name: str, message: str) -> dict:
    """Standardized success payload for primitive creation and basic transformations."""
    bounds = compute_world_bounds(obj)
    return {
        "status": "success",
        "action": action_name,
        "object": obj.name,
        "message": message,
        "location": [round(v, 4) for v in obj.location],
        "rotation": [round(math.degrees(v), 4) for v in obj.rotation_euler],
        "scale": [round(v, 4) for v in obj.scale],
        "dimensions": [round(v, 4) for v in obj.dimensions],
        "bounds": {
            "min": bounds["min"],
            "max": bounds["max"],
            "center": bounds["center"]
        }
    }


def action_create_cube(params: dict) -> dict:
    name = params.get("name", "Cube")
    location = [float(v) for v in params.get("location", [0, 0, 0])]
    size = float(params.get("size", 2.0))

    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object or bpy.context.object
    obj.name = name
    actual_name = obj.name

    if "dimensions" in params:
        dims = [float(v) for v in params["dimensions"]]
        obj.dimensions = tuple(dims)

    if "color" in params:
        action_set_material({
            "name": actual_name,
            "material_name": f"{actual_name}_Material",
            "color": params["color"]
        })

    msg = f"Created cube '{actual_name}' at {location}"
    return build_object_response(obj, "create_cube", msg)


def action_create_cylinder(params: dict) -> dict:
    name = params.get("name", "Cylinder")
    location = [float(v) for v in params.get("location", [0, 0, 0])]
    radius = float(params.get("radius", 1.0))
    height = float(params.get("height", params.get("depth", 2.0)))
    vertices = int(params.get("vertices", 32))

    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=height,
        vertices=vertices,
        location=location
    )
    obj = bpy.context.active_object or bpy.context.object
    obj.name = name
    actual_name = obj.name

    if "color" in params:
        action_set_material({
            "name": actual_name,
            "material_name": f"{actual_name}_Material",
            "color": params["color"]
        })

    msg = f"Created cylinder '{actual_name}' at {location} (radius={radius}, height={height}, vertices={vertices})"
    return build_object_response(obj, "create_cylinder", msg)


def action_create_sphere(params: dict) -> dict:
    name = params.get("name", "Sphere")
    location = [float(v) for v in params.get("location", [0, 0, 0])]
    radius = float(params.get("radius", 1.0))
    segments = int(params.get("segments", 32))
    ring_count = int(params.get("ring_count", 16))

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=segments,
        ring_count=ring_count,
        location=location
    )
    obj = bpy.context.active_object or bpy.context.object
    obj.name = name
    actual_name = obj.name

    if "color" in params:
        action_set_material({
            "name": actual_name,
            "material_name": f"{actual_name}_Material",
            "color": params["color"]
        })

    action_name = params.get("action", "create_sphere")
    msg = f"Created sphere '{actual_name}' at {location} (radius={radius}, segments={segments}, ring_count={ring_count})"
    return build_object_response(obj, action_name, msg)


def action_create_plane(params: dict) -> dict:
    name = params.get("name", "Plane")
    location = [float(v) for v in params.get("location", [0, 0, 0])]
    size = float(params.get("size", 2.0))

    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    obj = bpy.context.active_object or bpy.context.object
    obj.name = name
    actual_name = obj.name

    if "dimensions" in params:
        dims = params["dimensions"]
        obj.dimensions = (float(dims[0]), float(dims[1]), 0.0)

    if "color" in params:
        action_set_material({
            "name": actual_name,
            "material_name": f"{actual_name}_Material",
            "color": params["color"]
        })

    msg = f"Created plane '{actual_name}' at {location} (size={size})"
    return build_object_response(obj, "create_plane", msg)


def action_create_cone(params: dict) -> dict:
    name = params.get("name", "Cone")
    location = [float(v) for v in params.get("location", [0, 0, 0])]
    radius = float(params.get("radius", params.get("radius1", 1.0)))
    radius_top = float(params.get("radius_top", params.get("radius2", 0.0)))
    height = float(params.get("height", params.get("depth", 2.0)))
    vertices = int(params.get("vertices", 32))

    bpy.ops.mesh.primitive_cone_add(
        radius1=radius,
        radius2=radius_top,
        depth=height,
        vertices=vertices,
        location=location
    )
    obj = bpy.context.active_object or bpy.context.object
    obj.name = name
    actual_name = obj.name

    if "color" in params:
        action_set_material({
            "name": actual_name,
            "material_name": f"{actual_name}_Material",
            "color": params["color"]
        })

    msg = f"Created cone '{actual_name}' at {location} (radius={radius}, radius_top={radius_top}, height={height}, vertices={vertices})"
    return build_object_response(obj, "create_cone", msg)


def action_create_torus(params: dict) -> dict:
    name = params.get("name", "Torus")
    location = [float(v) for v in params.get("location", [0, 0, 0])]
    major_radius = float(params.get("major_radius", 1.0))
    minor_radius = float(params.get("minor_radius", 0.25))
    major_segments = int(params.get("major_segments", 48))
    minor_segments = int(params.get("minor_segments", 12))

    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=major_segments,
        minor_segments=minor_segments,
        location=location
    )
    obj = bpy.context.active_object or bpy.context.object
    obj.name = name
    actual_name = obj.name

    if "color" in params:
        action_set_material({
            "name": actual_name,
            "material_name": f"{actual_name}_Material",
            "color": params["color"]
        })

    msg = f"Created torus '{actual_name}' at {location} (major_radius={major_radius}, minor_radius={minor_radius}, major_segments={major_segments}, minor_segments={minor_segments})"
    return build_object_response(obj, "create_torus", msg)


def action_set_location(params: dict) -> dict:
    obj = get_object(params["name"])
    loc = [float(v) for v in params["location"]]
    obj.location = tuple(loc)
    msg = f"Set location of '{obj.name}' to {loc}"
    return build_object_response(obj, "set_location", msg)


def action_set_rotation(params: dict) -> dict:
    obj = get_object(params["name"])
    rot_deg = [float(v) for v in params["rotation"]]
    obj.rotation_euler = tuple(math.radians(v) for v in rot_deg)
    msg = f"Set rotation of '{obj.name}' to {rot_deg} degrees"
    return build_object_response(obj, "set_rotation", msg)


def action_set_scale(params: dict) -> dict:
    obj = get_object(params["name"])
    scale = [float(v) for v in params["scale"]]
    obj.scale = tuple(scale)
    msg = f"Set scale of '{obj.name}' to {scale}"
    return build_object_response(obj, "set_scale", msg)


def action_set_dimensions(params: dict) -> dict:
    obj = get_object(params["name"])
    dims = [float(v) for v in params["dimensions"]]

    # If any dimension is zero, ensure the object is actually planar along that axis
    if hasattr(obj, "data") and hasattr(obj.data, "vertices") and len(obj.data.vertices) > 0:
        axis_attrs = ("x", "y", "z")
        for d, axis in zip(dims, axis_attrs):
            if d == 0.0:
                coords = [getattr(v.co, axis) for v in obj.data.vertices]
                local_span = max(coords) - min(coords)
                if local_span > 1e-5:
                    raise ValueError(f"Cannot set zero dimension on non-planar axis '{axis.upper()}' of object '{obj.name}'.")

    obj.dimensions = tuple(dims)
    msg = f"Set dimensions of '{obj.name}' to {dims}"
    return build_object_response(obj, "set_dimensions", msg)


def _restore_selection_and_active(prev_selected: list, prev_active, target_obj=None):
    """Safely restore original selection and active object, ignoring deleted objects."""
    try:
        # If target was not originally selected, ensure it is deselected
        if target_obj and target_obj not in prev_selected:
            try:
                if hasattr(bpy.data, "objects") and target_obj.name in bpy.data.objects:
                    target_obj.select_set(False)
            except Exception:
                pass

        # Restore original selection set (ignoring any objects that no longer exist)
        for o in prev_selected:
            try:
                if o and hasattr(bpy.data, "objects") and o.name in bpy.data.objects:
                    o.select_set(True)
            except Exception:
                pass

        # Restore original active object (including None)
        if prev_active:
            try:
                if hasattr(bpy.data, "objects") and prev_active.name in bpy.data.objects:
                    bpy.context.view_layer.objects.active = prev_active
            except Exception:
                pass
        else:
            try:
                bpy.context.view_layer.objects.active = None
            except Exception:
                pass
    except Exception:
        pass


def action_apply_transform(params: dict) -> dict:
    obj = get_object(params["name"])
    apply_loc = bool(params.get("location", False))
    apply_rot = bool(params.get("rotation", False))
    apply_scale = bool(params.get("scale", True))

    if not (apply_loc or apply_rot or apply_scale):
        raise ValueError(f"apply_transform on '{obj.name}' requires at least one of 'location', 'rotation', or 'scale' to be True.")

    prev_active = getattr(bpy.context.view_layer.objects, "active", None)
    prev_selected = list(getattr(bpy.context, "selected_objects", []))

    try:
        # Deselect other objects and ensure target is selected and active
        for o in prev_selected:
            if o != obj:
                o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        ensure_object_mode()

        if hasattr(bpy.ops.object, "transform_apply"):
            if not bpy.ops.object.transform_apply.poll():
                raise RuntimeError(f"Cannot apply transform on '{obj.name}': context poll failed.")
            bpy.ops.object.transform_apply(
                location=apply_loc,
                rotation=apply_rot,
                scale=apply_scale
            )
        else:
            raise RuntimeError("bpy.ops.object.transform_apply operator not available.")

        bpy.context.view_layer.update()
    finally:
        _restore_selection_and_active(prev_selected, prev_active, obj)

    msg = f"Applied transform on '{obj.name}' (location={apply_loc}, rotation={apply_rot}, scale={apply_scale})"
    return build_object_response(obj, "apply_transform", msg)


def action_rename_object(params: dict) -> dict:
    obj = get_object(params["name"])
    new_name = str(params["new_name"]).strip()
    if not new_name:
        raise ValueError("Parameter 'new_name' cannot be empty")
    old_name = obj.name
    obj.name = new_name
    actual_name = obj.name
    return {"status": "success", "action": "rename_object", "message": f"Renamed '{old_name}' to '{actual_name}'", "object": actual_name}


def action_delete_object(params: dict) -> dict:
    obj = get_object(params["name"])
    name = obj.name
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    return {"status": "success", "message": f"Deleted object '{name}'", "object": name}


def action_clear_scene(params: dict = None) -> dict:
    deleted_count = 0
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
            deleted_count += 1
    bpy.context.view_layer.update()
    return {"status": "success", "message": f"Cleared {deleted_count} mesh objects from scene"}


def action_add_bevel(params: dict) -> dict:
    obj = get_object(params["name"])
    if obj.type != 'MESH':
        raise ValueError(f"Object '{obj.name}' is not a mesh; cannot add bevel.")

    mod_name_req = str(params.get("modifier_name", "Bevel")).strip() or "Bevel"
    width = float(params.get("width", 0.05))
    segments = int(params.get("segments", 3))

    limit_method_raw = params.get("limit_method", "ANGLE")
    limit_method = str(limit_method_raw).strip().upper() if isinstance(limit_method_raw, str) else "ANGLE"
    if limit_method not in {"ANGLE", "NONE"}:
        limit_method = "ANGLE"

    angle_limit = float(params.get("angle_limit", 30.0))
    harden_normals = params.get("harden_normals", True)
    if not isinstance(harden_normals, bool):
        harden_normals = True

    mod = obj.modifiers.new(name=mod_name_req, type='BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = limit_method
    if limit_method == "ANGLE":
        mod.angle_limit = math.radians(angle_limit)
    else:
        try:
            mod.angle_limit = math.radians(angle_limit)
        except Exception:
            pass
    mod.harden_normals = harden_normals
    actual_mod_name = mod.name

    bpy.context.view_layer.update()
    return {
        "status": "success",
        "action": "add_bevel",
        "message": f"Added Bevel modifier '{actual_mod_name}' to '{obj.name}' (width={width}, segments={segments}, limit_method={limit_method})",
        "object": obj.name,
        "modifier_name": actual_mod_name,
        "width": width,
        "segments": segments,
        "limit_method": limit_method,
        "angle_limit": angle_limit,
        "harden_normals": harden_normals
    }


def action_set_shade_smooth(params: dict) -> dict:
    obj = get_object(params["name"])
    if obj.type != 'MESH':
        raise ValueError(f"Object '{obj.name}' is not a mesh; cannot set smooth shading.")

    smooth = params.get("smooth", True)
    if not isinstance(smooth, bool):
        smooth = True

    auto_smooth = params.get("auto_smooth", False)
    if not isinstance(auto_smooth, bool):
        auto_smooth = False

    angle = float(params.get("angle", 30.0))

    prev_active = getattr(bpy.context.view_layer.objects, "active", None)
    prev_selected = list(getattr(bpy.context, "selected_objects", []))

    try:
        # Deselect other objects and ensure target is selected and active
        for o in prev_selected:
            if o != obj:
                o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        ensure_object_mode()

        if not smooth:
            if hasattr(bpy.ops.object, "shade_flat"):
                if not bpy.ops.object.shade_flat.poll():
                    raise RuntimeError(f"Cannot apply shade_flat on '{obj.name}': context poll failed.")
                bpy.ops.object.shade_flat()
            msg = f"Applied flat shading to '{obj.name}'"
        elif auto_smooth:
            if hasattr(bpy.ops.object, "shade_smooth_by_angle"):
                if not bpy.ops.object.shade_smooth_by_angle.poll():
                    raise RuntimeError(f"Cannot apply shade_smooth_by_angle on '{obj.name}': context poll failed.")
                bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))
            elif hasattr(bpy.ops.object, "shade_smooth"):
                if not bpy.ops.object.shade_smooth.poll():
                    raise RuntimeError(f"Cannot apply shade_smooth on '{obj.name}': context poll failed.")
                bpy.ops.object.shade_smooth()
            msg = f"Applied smooth shading by angle ({angle}°) to '{obj.name}'"
        else:
            if hasattr(bpy.ops.object, "shade_smooth"):
                if not bpy.ops.object.shade_smooth.poll():
                    raise RuntimeError(f"Cannot apply shade_smooth on '{obj.name}': context poll failed.")
                bpy.ops.object.shade_smooth()
            msg = f"Applied smooth shading to '{obj.name}'"

        bpy.context.view_layer.update()
    finally:
        _restore_selection_and_active(prev_selected, prev_active, obj)

    return {
        "status": "success",
        "action": "set_shade_smooth",
        "message": msg,
        "object": obj.name,
        "smooth": smooth,
        "auto_smooth": auto_smooth,
        "angle": angle
    }


def action_apply_modifier(params: dict) -> dict:
    obj = get_object(params["name"])
    mod_name = str(params.get("modifier_name", "Bevel")).strip()
    if not mod_name:
        raise ValueError("Parameter 'modifier_name' cannot be empty")
    if mod_name not in obj.modifiers:
        raise ValueError(f"Modifier '{mod_name}' not found on object '{obj.name}'")

    prev_active = getattr(bpy.context.view_layer.objects, "active", None)
    prev_selected = list(getattr(bpy.context, "selected_objects", []))

    try:
        # Deselect other objects and ensure target is selected and active
        for o in prev_selected:
            if o != obj:
                o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Ensure object mode for modifier application
        ensure_object_mode()

        if hasattr(bpy.ops.object, "modifier_apply"):
            if not bpy.ops.object.modifier_apply.poll():
                raise RuntimeError(f"Cannot apply modifier '{mod_name}' on '{obj.name}': context poll failed.")
            bpy.ops.object.modifier_apply(modifier=mod_name)
        else:
            raise RuntimeError("bpy.ops.object.modifier_apply operator not available.")

        bpy.context.view_layer.update()
    finally:
        _restore_selection_and_active(prev_selected, prev_active, obj)

    return {
        "status": "success",
        "action": "apply_modifier",
        "message": f"Applied modifier '{mod_name}' on '{obj.name}'",
        "object": obj.name,
        "modifier_name": mod_name
    }


def action_set_material(params: dict) -> dict:
    obj = get_object(params["name"])
    if obj.type != 'MESH':
        raise ValueError(f"Object '{obj.name}' is not a mesh; cannot assign material.")

    mat_name = params.get("material_name", f"{obj.name}_Material")
    color = [float(v) for v in params.get("color", [0.8, 0.8, 0.8, 1.0])]
    metallic = float(params.get("metallic", 0.0))
    roughness = float(params.get("roughness", 0.5))
    alpha = float(params.get("alpha", 1.0))

    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)

    if hasattr(mat, "use_nodes"):
        try:
            mat.use_nodes = True
        except Exception:
            pass
    mat.diffuse_color = tuple(color)

    if mat.node_tree:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = tuple(color)
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = metallic
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = roughness
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = alpha

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return {
        "status": "success",
        "message": f"Assigned material '{mat_name}' to '{obj.name}' (color={color}, metallic={metallic}, roughness={roughness}, alpha={alpha})",
        "object": obj.name,
        "material": mat_name
    }


def action_create_light(params: dict) -> dict:
    name = params.get("name", "Light")
    light_type = params.get("light_type", "POINT").upper()
    if light_type not in {"POINT", "SUN", "SPOT", "AREA"}:
        raise ValueError(f"Invalid light type: {light_type}. Must be POINT, SUN, SPOT, or AREA")

    location = [float(v) for v in params.get("location", [4, -4, 5])]
    energy = float(params.get("energy", 1000.0))
    color = [float(v) for v in params.get("color", [1.0, 1.0, 1.0])][:3]

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    light_data.color = tuple(color)

    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = tuple(location)
    bpy.context.view_layer.update()

    return {"status": "success", "message": f"Created {light_type} light '{name}' at {location} (energy={energy})", "object": name}


def action_create_camera(params: dict) -> dict:
    name = params.get("name", "Camera")
    location = [float(v) for v in params.get("location", [0, -5, 3])]
    rot_deg = [float(v) for v in params.get("rotation", [60, 0, 0])]

    cam_data = bpy.data.cameras.new(name=name)
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = tuple(location)
    cam_obj.rotation_euler = tuple(math.radians(v) for v in rot_deg)
    bpy.context.view_layer.update()

    return {"status": "success", "message": f"Created camera '{name}' at {location}", "object": name}


def action_set_background_color(params: dict) -> dict:
    color = [float(v) for v in params.get("color", [0.05, 0.05, 0.05, 1.0])]
    strength = float(params.get("strength", 1.0))

    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = tuple(color)
        bg.inputs["Strength"].default_value = strength

    return {"status": "success", "message": f"Set world background color to {color} (strength={strength})"}


def action_get_scene_info(params: dict = None) -> dict:
    bpy.context.view_layer.update()
    objects_info = []
    for obj in bpy.context.scene.objects:
        mats = [m.name for m in obj.data.materials if m] if hasattr(obj.data, "materials") else []
        objects_info.append({
            "name": obj.name,
            "type": obj.type,
            "location": [round(v, 3) for v in obj.location],
            "rotation_deg": [round(math.degrees(v), 2) for v in obj.rotation_euler],
            "dimensions": [round(v, 3) for v in obj.dimensions],
            "materials": mats
        })

    return {
        "status": "success",
        "message": f"Retrieved scene information ({len(objects_info)} objects)",
        "objects": objects_info,
        "object_count": len(objects_info)
    }


# ---------------------------------------------------------------------------
# Object-Relative Positioning & Assembly Handlers
# ---------------------------------------------------------------------------

def action_get_object_bounds(params: dict) -> dict:
    obj = get_object(params["name"])
    bounds = compute_world_bounds(obj)
    return {
        "status": "success",
        "message": f"Computed world-space bounds for '{obj.name}'",
        "object": obj.name,
        "min": bounds["min"],
        "max": bounds["max"],
        "center": bounds["center"],
        "dimensions": bounds["dimensions"],
        "corners": bounds["corners"]
    }


def action_place_relative(params: dict) -> dict:
    obj = get_object(params["name"])
    ref = get_object(params["reference_object"])

    if obj == ref:
        raise ValueError(f"Cannot position object '{obj.name}' relative to itself.")

    rel = str(params.get("relationship", "above")).lower()
    gap = float(params.get("gap", 0.0))
    overlap = float(params.get("overlap", 0.0))
    net_offset = gap - overlap

    obj_bounds = compute_world_bounds(obj)
    ref_bounds = compute_world_bounds(ref)

    delta_x = 0.0
    delta_y = 0.0
    delta_z = 0.0

    if rel == "above":
        # Bottom of obj touches top of ref (+ net_offset)
        delta_z = (ref_bounds["max"][2] + net_offset) - obj_bounds["min"][2]
        if params.get("align_xy", True):
            delta_x = ref_bounds["center"][0] - obj_bounds["center"][0]
            delta_y = ref_bounds["center"][1] - obj_bounds["center"][1]

    elif rel == "below":
        # Top of obj touches bottom of ref (- net_offset)
        delta_z = (ref_bounds["min"][2] - net_offset) - obj_bounds["max"][2]
        if params.get("align_xy", True):
            delta_x = ref_bounds["center"][0] - obj_bounds["center"][0]
            delta_y = ref_bounds["center"][1] - obj_bounds["center"][1]

    elif rel == "beside":
        direction = str(params.get("direction", "+X")).upper()
        if direction in {"+X", "RIGHT"}:
            delta_x = (ref_bounds["max"][0] + net_offset) - obj_bounds["min"][0]
        elif direction in {"-X", "LEFT"}:
            delta_x = (ref_bounds["min"][0] - net_offset) - obj_bounds["max"][0]
        elif direction in {"+Y", "FRONT"}:
            delta_y = (ref_bounds["max"][1] + net_offset) - obj_bounds["min"][1]
        elif direction in {"-Y", "BACK"}:
            delta_y = (ref_bounds["min"][1] - net_offset) - obj_bounds["max"][1]
        else:
            raise ValueError(f"Invalid beside direction: '{direction}'. Must be +X, -X, +Y, or -Y")

        if params.get("align_z", False):
            delta_z = ref_bounds["center"][2] - obj_bounds["center"][2]
        if params.get("align_y", False) and direction in {"+X", "-X", "RIGHT", "LEFT"}:
            delta_y = ref_bounds["center"][1] - obj_bounds["center"][1]
        if params.get("align_x", False) and direction in {"+Y", "-Y", "FRONT", "BACK"}:
            delta_x = ref_bounds["center"][0] - obj_bounds["center"][0]

    elif rel in {"centred_on", "centered_on"}:
        delta_x = ref_bounds["center"][0] - obj_bounds["center"][0]
        delta_y = ref_bounds["center"][1] - obj_bounds["center"][1]
        if params.get("align_z", True):
            delta_z = ref_bounds["center"][2] - obj_bounds["center"][2]

    elif rel in {"inside", "around"}:
        delta_x = ref_bounds["center"][0] - obj_bounds["center"][0]
        delta_y = ref_bounds["center"][1] - obj_bounds["center"][1]
        delta_z = ref_bounds["center"][2] - obj_bounds["center"][2]
    else:
        raise ValueError(f"Unsupported relationship: '{rel}'")

    # Preserve all other properties: only update translation
    obj.location.x += delta_x
    obj.location.y += delta_y
    obj.location.z += delta_z
    bpy.context.view_layer.update()

    new_bounds = compute_world_bounds(obj)
    return {
        "status": "success",
        "message": (
            f"Positioned '{obj.name}' {rel} '{ref.name}' "
            f"(offset delta=[{delta_x:.4f}, {delta_y:.4f}, {delta_z:.4f}])"
        ),
        "object": obj.name,
        "reference_object": ref.name,
        "relationship": rel,
        "new_location": [round(v, 4) for v in obj.location],
        "bounds": new_bounds
    }


def action_align_objects(params: dict) -> dict:
    obj = get_object(params["name"])
    ref = get_object(params["reference_object"])

    if obj == ref:
        raise ValueError(f"Cannot align object '{obj.name}' to itself.")

    raw_axes = params.get("axes", ["X", "Y"])
    if isinstance(raw_axes, str):
        axes = [c.upper() for c in raw_axes if c.upper() in {"X", "Y", "Z"}]
    else:
        axes = [str(a).upper() for a in raw_axes]

    alignment = str(params.get("alignment", "center")).lower()
    axis_map = {"X": 0, "Y": 1, "Z": 2}

    obj_bounds = compute_world_bounds(obj)
    ref_bounds = compute_world_bounds(ref)

    deltas = [0.0, 0.0, 0.0]
    for axis in axes:
        idx = axis_map[axis]
        if alignment in {"center", "centre"}:
            deltas[idx] = ref_bounds["center"][idx] - obj_bounds["center"][idx]
        elif alignment == "min":
            deltas[idx] = ref_bounds["min"][idx] - obj_bounds["min"][idx]
        elif alignment == "max":
            deltas[idx] = ref_bounds["max"][idx] - obj_bounds["max"][idx]
        else:
            raise ValueError(f"Unsupported alignment: '{alignment}'")

    obj.location.x += deltas[0]
    obj.location.y += deltas[1]
    obj.location.z += deltas[2]
    bpy.context.view_layer.update()

    new_bounds = compute_world_bounds(obj)
    return {
        "status": "success",
        "message": f"Aligned '{obj.name}' with '{ref.name}' along {axes} ({alignment})",
        "object": obj.name,
        "reference_object": ref.name,
        "axes": axes,
        "alignment": alignment,
        "new_location": [round(v, 4) for v in obj.location],
        "bounds": new_bounds
    }


def action_set_relative_scale(params: dict) -> dict:
    obj = get_object(params["name"])
    ref = get_object(params["reference_object"])

    factor = float(params.get("factor", 1.0))
    match_dim = str(params.get("match_dimension", "all")).lower()

    obj_bounds = compute_world_bounds(obj)
    ref_bounds = compute_world_bounds(ref)

    if match_dim in {"width", "x"}:
        ratio = (ref_bounds["dimensions"][0] * factor) / max(obj_bounds["dimensions"][0], 1e-6)
    elif match_dim in {"depth", "y"}:
        ratio = (ref_bounds["dimensions"][1] * factor) / max(obj_bounds["dimensions"][1], 1e-6)
    elif match_dim in {"height", "z"}:
        ratio = (ref_bounds["dimensions"][2] * factor) / max(obj_bounds["dimensions"][2], 1e-6)
    elif match_dim == "max":
        ratio = (max(ref_bounds["dimensions"]) * factor) / max(max(obj_bounds["dimensions"]), 1e-6)
    else:  # "all"
        ratio = factor

    obj.scale.x *= ratio
    obj.scale.y *= ratio
    obj.scale.z *= ratio
    bpy.context.view_layer.update()

    new_bounds = compute_world_bounds(obj)
    return {
        "status": "success",
        "message": f"Scaled '{obj.name}' relative to '{ref.name}' by ratio {ratio:.4f} (match={match_dim})",
        "object": obj.name,
        "reference_object": ref.name,
        "new_dimensions": new_bounds["dimensions"],
        "bounds": new_bounds
    }


def action_verify_assembly(params: dict) -> dict:
    name_a = params.get("object_a", params.get("name"))
    name_b = params.get("object_b", params.get("reference_object"))

    obj_a = get_object(name_a)
    obj_b = get_object(name_b)

    if obj_a == obj_b:
        raise ValueError("Cannot verify assembly of an object with itself.")

    tol = float(params.get("tolerance", 0.01))
    expected_rel = params.get("expected_relationship")

    b_a = compute_world_bounds(obj_a)
    b_b = compute_world_bounds(obj_b)

    # 1. Center alignment offsets
    offset_x = round(abs(b_a["center"][0] - b_b["center"][0]), 4)
    offset_y = round(abs(b_a["center"][1] - b_b["center"][1]), 4)
    offset_z = round(abs(b_a["center"][2] - b_b["center"][2]), 4)
    aligned_xy = (offset_x <= tol and offset_y <= tol)

    # 2. Facing boundary gaps
    # Gap along Z if A is above B (min_a - max_b)
    gap_z_above = round(b_a["min"][2] - b_b["max"][2], 4)
    # Gap along Z if B is above A (min_b - max_a)
    gap_z_below = round(b_b["min"][2] - b_a["max"][2], 4)

    # 3. 3D Axis-Aligned Bounding Box (AABB) overlap evaluation
    overlap_x = max(0.0, min(b_a["max"][0], b_b["max"][0]) - max(b_a["min"][0], b_b["min"][0]))
    overlap_y = max(0.0, min(b_a["max"][1], b_b["max"][1]) - max(b_a["min"][1], b_b["min"][1]))
    overlap_z = max(0.0, min(b_a["max"][2], b_b["max"][2]) - max(b_a["min"][2], b_b["min"][2]))
    has_aabb_overlap = bool(overlap_x > tol and overlap_y > tol and overlap_z > tol)

    # 4. Physical contact evaluation along vertical axis
    touching_above = abs(gap_z_above) <= tol
    touching_below = abs(gap_z_below) <= tol
    is_touching = touching_above or touching_below

    # 5. Relationship check
    relationship_verified = True
    rel_message = "Assembly verified"

    if expected_rel:
        exp = str(expected_rel).lower()
        if exp == "above":
            if not touching_above and gap_z_above > tol:
                relationship_verified = False
                rel_message = f"Excessive gap along Z ({gap_z_above:.4f} > {tol})"
            elif not touching_above and gap_z_above < -tol:
                relationship_verified = False
                rel_message = f"Bounding-box penetration along Z ({gap_z_above:.4f} < -{tol})"
            elif not aligned_xy:
                relationship_verified = False
                rel_message = f"Center XY offset exceeds tolerance (dx={offset_x}, dy={offset_y})"
            else:
                rel_message = f"'{obj_a.name}' is directly above '{obj_b.name}' and touching (gap={gap_z_above:.4f})"
        elif exp == "below":
            if not touching_below and gap_z_below > tol:
                relationship_verified = False
                rel_message = f"Excessive gap along Z ({gap_z_below:.4f} > {tol})"
            elif not touching_below and gap_z_below < -tol:
                relationship_verified = False
                rel_message = f"Bounding-box penetration along Z ({gap_z_below:.4f} < -{tol})"
            elif not aligned_xy:
                relationship_verified = False
                rel_message = f"Center XY offset exceeds tolerance (dx={offset_x}, dy={offset_y})"
            else:
                rel_message = f"'{obj_a.name}' is directly below '{obj_b.name}' and touching (gap={gap_z_below:.4f})"

    return {
        "status": "success",
        "verified": relationship_verified,
        "message": rel_message,
        "object_a": obj_a.name,
        "object_b": obj_b.name,
        "tolerance": tol,
        "aligned_xy": aligned_xy,
        "center_offsets": {"dx": offset_x, "dy": offset_y, "dz": offset_z},
        "vertical_gap": {
            "a_above_b": gap_z_above,
            "b_above_a": gap_z_below,
            "is_touching": is_touching
        },
        "bounding_box_overlap": has_aabb_overlap,
        "collision_detection_method": "Axis-Aligned Bounding Box (AABB) in world space (manifold mesh collision not evaluated)",
        "bounds_a": {"min": b_a["min"], "max": b_a["max"], "center": b_a["center"], "dimensions": b_a["dimensions"]},
        "bounds_b": {"min": b_b["min"], "max": b_b["max"], "center": b_b["center"], "dimensions": b_b["dimensions"]}
    }


# Action Dispatch Table
ACTION_DISPATCH = {
    "create_cube": action_create_cube,
    "create_cylinder": action_create_cylinder,
    "create_sphere": action_create_sphere,
    "create_uv_sphere": action_create_sphere,
    "create_plane": action_create_plane,
    "create_cone": action_create_cone,
    "create_torus": action_create_torus,
    "set_location": action_set_location,
    "set_rotation": action_set_rotation,
    "set_scale": action_set_scale,
    "set_dimensions": action_set_dimensions,
    "apply_transform": action_apply_transform,
    "rename_object": action_rename_object,
    "delete_object": action_delete_object,
    "clear_scene": action_clear_scene,
    "add_bevel": action_add_bevel,
    "set_shade_smooth": action_set_shade_smooth,
    "apply_modifier": action_apply_modifier,
    "set_material": action_set_material,
    "create_light": action_create_light,
    "create_camera": action_create_camera,
    "set_background_color": action_set_background_color,
    "get_scene_info": action_get_scene_info,
    # Object-Relative Positioning & Assembly
    "get_object_bounds": action_get_object_bounds,
    "place_relative": action_place_relative,
    "align_objects": action_align_objects,
    "set_relative_scale": action_set_relative_scale,
    "verify_assembly": action_verify_assembly,
}


# Actions requiring Blender to be in Object mode
OBJECT_MODE_REQUIRED_ACTIONS = {
    "create_cube",
    "create_cylinder",
    "create_sphere",
    "create_uv_sphere",
    "create_plane",
    "create_cone",
    "create_torus",
    "set_location",
    "set_rotation",
    "set_scale",
    "set_dimensions",
    "apply_transform",
    "delete_object",
    "clear_scene",
    "add_bevel",
    "set_shade_smooth",
    "apply_modifier",
    "create_light",
    "create_camera",
}


def ensure_object_mode():
    """Ensure Blender is in OBJECT mode if currently in another mode."""
    try:
        if hasattr(bpy.context, "mode") and bpy.context.mode != 'OBJECT':
            if hasattr(bpy.ops.object, "mode_set") and bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass


def execute_action(action: dict) -> dict:
    """Execute a single validated action on Blender's main thread."""
    act_type = action.get("action")
    if act_type not in ACTION_DISPATCH:
        return {"status": "error", "error": f"Unsupported action: '{act_type}'"}

    if act_type in OBJECT_MODE_REQUIRED_ACTIONS:
        ensure_object_mode()

    handler = ACTION_DISPATCH[act_type]
    try:
        result = handler(action)
        bpy.context.view_layer.update()
        return result
    except Exception as exc:
        return {"status": "error", "error": f"Action '{act_type}' failed: {str(exc)}"}


def execute_action_list(actions: list) -> dict:
    """Execute multiple sequential actions and return detailed itemized results."""
    results = []
    has_error = False

    for idx, act in enumerate(actions):
        res = execute_action(act)
        res["index"] = idx
        res["action"] = act.get("action", "unknown")
        results.append(res)
        if res.get("status") == "error":
            has_error = True

    overall_status = "error" if (has_error and len(results) == 1) else ("partial_success" if has_error else "success")
    return {
        "status": overall_status,
        "total_actions": len(actions),
        "results": results
    }
