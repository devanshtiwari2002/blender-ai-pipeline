"""Shared Action Validator for Blender AI Assistant.
Runs in standard Python (venv) as well as inside Blender (no bpy dependency).
"""
import math

SUPPORTED_ACTIONS = {
    # Creation
    "create_cube",
    "create_cylinder",
    "create_sphere",
    "create_uv_sphere",
    "create_plane",
    "create_cone",
    "create_torus",
    # Modification
    "set_location",
    "set_rotation",
    "set_scale",
    "set_dimensions",
    "apply_transform",
    "rename_object",
    "delete_object",
    "clear_scene",
    # Geometry
    "add_bevel",
    "set_shade_smooth",
    "apply_modifier",
    # Materials
    "set_material",
    # Scene
    "create_light",
    "create_camera",
    "set_background_color",
    # Scene Awareness
    "get_scene_info",
    # Object-Relative Positioning & Assembly
    "get_object_bounds",
    "place_relative",
    "align_objects",
    "set_relative_scale",
    "verify_assembly",
}

MAX_ACTIONS_PER_REQUEST = 500

VALID_RELATIONSHIPS = {"above", "below", "beside", "inside", "centred_on", "centered_on", "around"}
VALID_BESIDE_DIRECTIONS = {"+X", "-X", "+Y", "-Y", "RIGHT", "LEFT", "FRONT", "BACK"}
VALID_ALIGNMENTS = {"center", "centre", "min", "max"}

# Allowed keys for the 7 primitive creation actions (strict unknown parameter checking)
PRIMITIVE_ALLOWED_KEYS = {
    "create_cube": {"action", "name", "location", "color", "size", "dimensions", "width", "depth", "height"},
    "create_cylinder": {"action", "name", "location", "color", "radius", "diameter", "height", "depth", "vertices"},
    "create_sphere": {"action", "name", "location", "color", "radius", "diameter", "segments", "ring_count"},
    "create_uv_sphere": {"action", "name", "location", "color", "radius", "diameter", "segments", "ring_count"},
    "create_plane": {"action", "name", "location", "color", "size", "dimensions", "width", "depth"},
    "create_cone": {"action", "name", "location", "color", "radius", "radius1", "radius_top", "radius2", "height", "depth", "vertices"},
    "create_torus": {"action", "name", "location", "color", "major_radius", "minor_radius", "major_segments", "minor_segments"},
}

# Allowed keys for modifier actions
MODIFIER_ALLOWED_KEYS = {
    "add_bevel": {"action", "name", "width", "segments", "modifier_name", "limit_method", "angle_limit", "harden_normals"},
    "apply_modifier": {"action", "name", "modifier_name"},
    "set_shade_smooth": {"action", "name", "smooth", "auto_smooth", "angle"},
}


def _is_number(v):
    """Check if value is a valid finite number, strictly rejecting booleans."""
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    if math.isnan(v) or math.isinf(v):
        return False
    return True


def _is_positive_number(v):
    return _is_number(v) and v > 0


def _is_non_negative_number(v):
    return _is_number(v) and v >= 0


def _is_integer(v):
    """Check if value is an actual integer, strictly rejecting booleans and floats."""
    if isinstance(v, bool):
        return False
    return isinstance(v, int)


def _validate_vec3(val, field_name="vector"):
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        return False, f"Field '{field_name}' must be a list/tuple of 3 numbers [x, y, z]"
    for x in val:
        if not _is_number(x):
            return False, f"All elements in '{field_name}' must be finite numbers (got {x!r})"
    return True, None


def _validate_color(val):
    if not isinstance(val, (list, tuple)) or len(val) not in (3, 4):
        return False, "Color must be a list of 3 or 4 RGBA floats [r, g, b, (a)]"
    for c in val:
        if not _is_number(c):
            return False, f"All color channels must be numeric (got {c!r})"
        if not (0.0 <= c <= 1.0):
            return False, f"Color channel {c} out of range [0.0, 1.0]"
    return True, None


def validate_single_action(action: dict):
    """Validate and normalize a single action dictionary against strict schema."""
    if not isinstance(action, dict):
        return False, "Action must be a JSON dictionary object"

    act_type = action.get("action")
    if not act_type:
        return False, "Missing required field: 'action'"

    if act_type not in SUPPORTED_ACTIONS:
        return False, f"Unsupported action: '{act_type}'. Allowed: {sorted(list(SUPPORTED_ACTIONS))}"

    # -------------------------------------------------------------------------
    # Primitive Creation Actions
    # -------------------------------------------------------------------------
    if act_type in PRIMITIVE_ALLOWED_KEYS:
        # 1. Strict unknown parameter rejection for primitives
        allowed = PRIMITIVE_ALLOWED_KEYS[act_type]
        for k in action.keys():
            if k not in allowed:
                return False, f"Unrecognized parameter '{k}' for action '{act_type}'. Allowed: {sorted(list(allowed))}"

        # 2. Common location and color validation
        if "location" in action:
            ok, err = _validate_vec3(action["location"], "location")
            if not ok:
                return False, err
        if "color" in action:
            ok, err = _validate_color(action["color"])
            if not ok:
                return False, err

        # 3. Primitive-specific dimension and quality validation & alias normalization
        if act_type == "create_cube":
            if "size" in action:
                if not _is_positive_number(action["size"]):
                    return False, "Parameter 'size' must be a positive number"
                action["size"] = float(action["size"])

            has_dims = "dimensions" in action
            has_w = "width" in action
            has_d = "depth" in action
            has_h = "height" in action

            for dim_k, dim_val in [("width", action.get("width")), ("depth", action.get("depth")), ("height", action.get("height"))]:
                if dim_val is not None and not _is_positive_number(dim_val):
                    return False, f"Parameter '{dim_k}' must be a positive number"

            if has_dims:
                dims = action["dimensions"]
                if not isinstance(dims, (list, tuple)) or len(dims) != 3:
                    return False, "Parameter 'dimensions' must be a list/tuple of 3 positive numbers [width, depth, height]"
                for i, d in enumerate(dims):
                    if not _is_positive_number(d):
                        return False, f"Element [{i}] of 'dimensions' must be a positive number"
                action["dimensions"] = [float(d) for d in dims]

                # Conflict checking if width/depth/height also passed
                if has_w and abs(float(action["width"]) - action["dimensions"][0]) > 1e-5:
                    return False, f"Conflicting parameters 'width' ({action['width']}) and 'dimensions[0]' ({action['dimensions'][0]})"
                if has_d and abs(float(action["depth"]) - action["dimensions"][1]) > 1e-5:
                    return False, f"Conflicting parameters 'depth' ({action['depth']}) and 'dimensions[1]' ({action['dimensions'][1]})"
                if has_h and abs(float(action["height"]) - action["dimensions"][2]) > 1e-5:
                    return False, f"Conflicting parameters 'height' ({action['height']}) and 'dimensions[2]' ({action['dimensions'][2]})"
                action.pop("width", None)
                action.pop("depth", None)
                action.pop("height", None)
            elif has_w or has_d or has_h:
                base = action.get("size", 2.0)
                w = float(action.pop("width", base))
                d = float(action.pop("depth", base))
                h = float(action.pop("height", base))
                action["dimensions"] = [w, d, h]

        elif act_type == "create_cylinder":
            if "vertices" in action:
                v = action["vertices"]
                if not _is_integer(v) or not (3 <= v <= 256):
                    return False, "Parameter 'vertices' must be an integer between 3 and 256"

            has_r = "radius" in action
            has_dia = "diameter" in action
            if has_r and not _is_positive_number(action["radius"]):
                return False, "Parameter 'radius' must be a positive number"
            if has_dia and not _is_positive_number(action["diameter"]):
                return False, "Parameter 'diameter' must be a positive number"

            if has_r and has_dia:
                r = float(action["radius"])
                dia = float(action["diameter"])
                if abs(r - (dia / 2.0)) > 1e-5:
                    return False, f"Conflicting parameters 'radius' ({r}) and 'diameter' ({dia})"
                action.pop("diameter")
                action["radius"] = r
            elif has_dia:
                action["radius"] = float(action.pop("diameter")) / 2.0
            elif has_r:
                action["radius"] = float(action["radius"])

            has_h = "height" in action
            has_dep = "depth" in action
            if has_h and not _is_positive_number(action["height"]):
                return False, "Parameter 'height' must be a positive number"
            if has_dep and not _is_positive_number(action["depth"]):
                return False, "Parameter 'depth' must be a positive number"

            if has_h and has_dep:
                h = float(action["height"])
                dep = float(action["depth"])
                if abs(h - dep) > 1e-5:
                    return False, f"Conflicting parameters 'height' ({h}) and 'depth' ({dep})"
                action.pop("depth")
                action["height"] = h
            elif has_dep:
                action["height"] = float(action.pop("depth"))
            elif has_h:
                action["height"] = float(action["height"])

        elif act_type in {"create_sphere", "create_uv_sphere"}:
            if "segments" in action:
                s = action["segments"]
                if not _is_integer(s) or not (3 <= s <= 256):
                    return False, "Parameter 'segments' must be an integer between 3 and 256"
            if "ring_count" in action:
                rc = action["ring_count"]
                if not _is_integer(rc) or not (3 <= rc <= 256):
                    return False, "Parameter 'ring_count' must be an integer between 3 and 256"

            has_r = "radius" in action
            has_dia = "diameter" in action
            if has_r and not _is_positive_number(action["radius"]):
                return False, "Parameter 'radius' must be a positive number"
            if has_dia and not _is_positive_number(action["diameter"]):
                return False, "Parameter 'diameter' must be a positive number"

            if has_r and has_dia:
                r = float(action["radius"])
                dia = float(action["diameter"])
                if abs(r - (dia / 2.0)) > 1e-5:
                    return False, f"Conflicting parameters 'radius' ({r}) and 'diameter' ({dia})"
                action.pop("diameter")
                action["radius"] = r
            elif has_dia:
                action["radius"] = float(action.pop("diameter")) / 2.0
            elif has_r:
                action["radius"] = float(action["radius"])

        elif act_type == "create_plane":
            if "size" in action:
                if not _is_positive_number(action["size"]):
                    return False, "Parameter 'size' must be a positive number"
                action["size"] = float(action["size"])

            has_dims = "dimensions" in action
            has_w = "width" in action
            has_d = "depth" in action

            if has_w and not _is_positive_number(action["width"]):
                return False, "Parameter 'width' must be a positive number"
            if has_d and not _is_positive_number(action["depth"]):
                return False, "Parameter 'depth' must be a positive number"

            if has_dims:
                dims = action["dimensions"]
                if not isinstance(dims, (list, tuple)) or len(dims) != 2:
                    return False, "Parameter 'dimensions' must be a list/tuple of 2 positive numbers [width, depth]"
                for i, d in enumerate(dims):
                    if not _is_positive_number(d):
                        return False, f"Element [{i}] of 'dimensions' must be a positive number"
                action["dimensions"] = [float(d) for d in dims]

                if has_w and abs(float(action["width"]) - action["dimensions"][0]) > 1e-5:
                    return False, f"Conflicting parameters 'width' ({action['width']}) and 'dimensions[0]' ({action['dimensions'][0]})"
                if has_d and abs(float(action["depth"]) - action["dimensions"][1]) > 1e-5:
                    return False, f"Conflicting parameters 'depth' ({action['depth']}) and 'dimensions[1]' ({action['dimensions'][1]})"
                action.pop("width", None)
                action.pop("depth", None)
            elif has_w or has_d:
                base = action.get("size", 2.0)
                w = float(action.pop("width", base))
                d = float(action.pop("depth", base))
                action["dimensions"] = [w, d]

        elif act_type == "create_cone":
            if "vertices" in action:
                v = action["vertices"]
                if not _is_integer(v) or not (3 <= v <= 256):
                    return False, "Parameter 'vertices' must be an integer between 3 and 256"

            # radius vs radius1
            has_r = "radius" in action
            has_r1 = "radius1" in action
            if has_r and not _is_positive_number(action["radius"]):
                return False, "Parameter 'radius' must be a positive number"
            if has_r1 and not _is_positive_number(action["radius1"]):
                return False, "Parameter 'radius1' must be a positive number"

            if has_r and has_r1:
                r = float(action["radius"])
                r1 = float(action["radius1"])
                if abs(r - r1) > 1e-5:
                    return False, f"Conflicting parameters 'radius' ({r}) and 'radius1' ({r1})"
                action.pop("radius1")
                action["radius"] = r
            elif has_r1:
                action["radius"] = float(action.pop("radius1"))
            elif has_r:
                action["radius"] = float(action["radius"])

            # radius_top vs radius2 (allow 0.0 for pointed cone)
            has_rt = "radius_top" in action
            has_r2 = "radius2" in action
            if has_rt and not _is_non_negative_number(action["radius_top"]):
                return False, "Parameter 'radius_top' must be a non-negative number"
            if has_r2 and not _is_non_negative_number(action["radius2"]):
                return False, "Parameter 'radius2' must be a non-negative number"

            if has_rt and has_r2:
                rt = float(action["radius_top"])
                r2 = float(action["radius2"])
                if abs(rt - r2) > 1e-5:
                    return False, f"Conflicting parameters 'radius_top' ({rt}) and 'radius2' ({r2})"
                action.pop("radius2")
                action["radius_top"] = rt
            elif has_r2:
                action["radius_top"] = float(action.pop("radius2"))
            elif has_rt:
                action["radius_top"] = float(action["radius_top"])

            # height vs depth
            has_h = "height" in action
            has_dep = "depth" in action
            if has_h and not _is_positive_number(action["height"]):
                return False, "Parameter 'height' must be a positive number"
            if has_dep and not _is_positive_number(action["depth"]):
                return False, "Parameter 'depth' must be a positive number"

            if has_h and has_dep:
                h = float(action["height"])
                dep = float(action["depth"])
                if abs(h - dep) > 1e-5:
                    return False, f"Conflicting parameters 'height' ({h}) and 'depth' ({dep})"
                action.pop("depth")
                action["height"] = h
            elif has_dep:
                action["height"] = float(action.pop("depth"))
            elif has_h:
                action["height"] = float(action["height"])

        elif act_type == "create_torus":
            if "major_segments" in action:
                ms = action["major_segments"]
                if not _is_integer(ms) or not (3 <= ms <= 256):
                    return False, "Parameter 'major_segments' must be an integer between 3 and 256"
            if "minor_segments" in action:
                mis = action["minor_segments"]
                if not _is_integer(mis) or not (3 <= mis <= 128):
                    return False, "Parameter 'minor_segments' must be an integer between 3 and 128"

            if "major_radius" in action:
                if not _is_positive_number(action["major_radius"]):
                    return False, "Parameter 'major_radius' must be a positive number"
                action["major_radius"] = float(action["major_radius"])
            if "minor_radius" in action:
                if not _is_positive_number(action["minor_radius"]):
                    return False, "Parameter 'minor_radius' must be a positive number"
                action["minor_radius"] = float(action["minor_radius"])

            maj = float(action.get("major_radius", 1.0))
            min_r = float(action.get("minor_radius", 0.25))
            if maj <= min_r:
                return False, f"Parameter 'major_radius' ({maj}) must be greater than 'minor_radius' ({min_r})"

    # Transformations
    elif act_type == "set_location":
        if "name" not in action or not str(action["name"]).strip():
            return False, "set_location requires a valid 'name' attribute"
        ok, err = _validate_vec3(action.get("location"), "location")
        if not ok:
            return False, err

    elif act_type == "set_rotation":
        if "name" not in action or not str(action["name"]).strip():
            return False, "set_rotation requires a valid 'name' attribute"
        ok, err = _validate_vec3(action.get("rotation"), "rotation")
        if not ok:
            return False, err

    elif act_type == "set_scale":
        if "name" not in action or not str(action["name"]).strip():
            return False, "set_scale requires a valid 'name' attribute"
        ok, err = _validate_vec3(action.get("scale"), "scale")
        if not ok:
            return False, err
        for s in action["scale"]:
            if not _is_positive_number(s):
                return False, f"All scale components in 'set_scale' must be strictly positive numbers (got {s!r})"

    elif act_type == "set_dimensions":
        if "name" not in action or not str(action["name"]).strip():
            return False, "set_dimensions requires a valid 'name' attribute"
        ok, err = _validate_vec3(action.get("dimensions"), "dimensions")
        if not ok:
            return False, err
        dims = action["dimensions"]
        for d in dims:
            if not _is_non_negative_number(d):
                return False, f"All dimensions in 'set_dimensions' must be non-negative numbers (got {d!r})"
        if sum(1 for d in dims if d == 0) > 1:
            return False, "At most one dimension in 'set_dimensions' may be zero for planar geometry"
        if sum(1 for d in dims if d > 0) < 2:
            return False, "At least two dimensions in 'set_dimensions' must be strictly positive"

    elif act_type == "apply_transform":
        if "name" not in action or not str(action["name"]).strip():
            return False, "apply_transform requires a valid 'name' attribute"
        allowed_keys = {"action", "name", "location", "rotation", "scale"}
        for k in action.keys():
            if k not in allowed_keys:
                return False, f"Unrecognized parameter '{k}' for action 'apply_transform'. Allowed: {sorted(list(allowed_keys))}"
        for flag in ("location", "rotation", "scale"):
            if flag in action and not isinstance(action[flag], bool):
                return False, f"Parameter '{flag}' in 'apply_transform' must be a boolean"
        if action.get("location") is False and action.get("rotation") is False and action.get("scale") is False:
            return False, "apply_transform requires at least one of 'location', 'rotation', or 'scale' to be True"

    elif act_type == "rename_object":
        if not action.get("name") or not action.get("new_name"):
            return False, "rename_object requires 'name' and 'new_name'"

    elif act_type == "delete_object":
        if not action.get("name"):
            return False, "delete_object requires 'name'"

    elif act_type == "clear_scene":
        pass

    # Geometry
    elif act_type in MODIFIER_ALLOWED_KEYS:
        # Strict unknown parameter checking for modifier actions
        allowed = MODIFIER_ALLOWED_KEYS[act_type]
        for k in action.keys():
            if k not in allowed:
                return False, f"Unrecognized parameter '{k}' for action '{act_type}'. Allowed: {sorted(list(allowed))}"

        # Target object name validation
        if "name" not in action or not isinstance(action["name"], str) or not action["name"].strip():
            return False, f"{act_type} requires a valid non-empty 'name' attribute"

        if act_type == "add_bevel":
            if "modifier_name" in action:
                if not isinstance(action["modifier_name"], str) or not action["modifier_name"].strip():
                    return False, "Parameter 'modifier_name' must be a non-empty string"
            if "width" in action:
                if not _is_positive_number(action["width"]):
                    return False, "Bevel 'width' must be a strictly positive finite number"
            if "segments" in action:
                seg = action["segments"]
                if not _is_integer(seg) or not (1 <= seg <= 128):
                    return False, "Bevel 'segments' must be an integer between 1 and 128"
            if "limit_method" in action:
                lm = action["limit_method"]
                if not isinstance(lm, str) or not lm.strip():
                    return False, "Parameter 'limit_method' in 'add_bevel' must be a string ('ANGLE' or 'NONE')"
                lm_norm = lm.strip().upper()
                if lm_norm not in {"ANGLE", "NONE"}:
                    return False, f"Parameter 'limit_method' in 'add_bevel' must be 'ANGLE' or 'NONE', got '{lm}'"
                action["limit_method"] = lm_norm
            if "angle_limit" in action:
                al = action["angle_limit"]
                if not _is_number(al) or not (0.0 < float(al) <= 180.0):
                    return False, "Parameter 'angle_limit' in 'add_bevel' must be a finite number strictly greater than 0 and at most 180 degrees"
            if "harden_normals" in action:
                if not isinstance(action["harden_normals"], bool):
                    return False, "Parameter 'harden_normals' in 'add_bevel' must be a boolean"

        elif act_type == "apply_modifier":
            if "modifier_name" in action:
                if not isinstance(action["modifier_name"], str) or not action["modifier_name"].strip():
                    return False, "Parameter 'modifier_name' must be a non-empty string"

        elif act_type == "set_shade_smooth":
            if "smooth" in action:
                if not isinstance(action["smooth"], bool):
                    return False, "Parameter 'smooth' in 'set_shade_smooth' must be a boolean"
            if "auto_smooth" in action:
                if not isinstance(action["auto_smooth"], bool):
                    return False, "Parameter 'auto_smooth' in 'set_shade_smooth' must be a boolean"
            if "angle" in action:
                ang = action["angle"]
                if not _is_number(ang) or not (0.0 < float(ang) <= 180.0):
                    return False, "Parameter 'angle' in 'set_shade_smooth' must be a finite number strictly greater than 0 and at most 180 degrees"

    # Materials
    elif act_type == "set_material":
        if not action.get("name"):
            return False, "set_material requires target 'name'"
        if "color" in action:
            ok, err = _validate_color(action["color"])
            if not ok:
                return False, err
        for prop in ("metallic", "roughness", "alpha"):
            if prop in action:
                try:
                    val = float(action[prop])
                    if not (0.0 <= val <= 1.0):
                        return False, f"Parameter '{prop}' must be in [0.0, 1.0]"
                except (ValueError, TypeError):
                    return False, f"Parameter '{prop}' must be numeric"

    # Scene
    elif act_type == "create_light":
        if "light_type" in action and str(action["light_type"]).upper() not in {"POINT", "SUN", "SPOT", "AREA"}:
            return False, "light_type must be POINT, SUN, SPOT, or AREA"
        if "location" in action:
            ok, err = _validate_vec3(action["location"], "location")
            if not ok:
                return False, err

    elif act_type == "create_camera":
        if "location" in action:
            ok, err = _validate_vec3(action["location"], "location")
            if not ok:
                return False, err
        if "rotation" in action:
            ok, err = _validate_vec3(action["rotation"], "rotation")
            if not ok:
                return False, err

    elif act_type == "set_background_color":
        if "color" in action:
            ok, err = _validate_color(action["color"])
            if not ok:
                return False, err

    elif act_type == "get_scene_info":
        pass

    # Object-Relative Positioning & Assembly
    elif act_type == "get_object_bounds":
        if not action.get("name") or not str(action["name"]).strip():
            return False, "get_object_bounds requires a valid 'name' attribute"

    elif act_type == "place_relative":
        name = action.get("name")
        ref = action.get("reference_object")
        if not name or not str(name).strip():
            return False, "place_relative requires a valid 'name' attribute"
        if not ref or not str(ref).strip():
            return False, "place_relative requires a valid 'reference_object' attribute"
        if str(name).strip() == str(ref).strip():
            return False, "place_relative cannot position an object relative to itself"

        rel = str(action.get("relationship", "")).lower()
        if rel not in VALID_RELATIONSHIPS:
            return False, f"Invalid relationship '{rel}'. Allowed: {sorted(list(VALID_RELATIONSHIPS))}"

        if rel == "beside":
            direction = str(action.get("direction", "+X")).upper()
            if direction not in VALID_BESIDE_DIRECTIONS:
                return False, f"Invalid beside direction '{direction}'. Allowed: {sorted(list(VALID_BESIDE_DIRECTIONS))}"

        for num_prop in ("gap", "overlap"):
            if num_prop in action:
                try:
                    float(action[num_prop])
                except (ValueError, TypeError):
                    return False, f"Parameter '{num_prop}' must be numeric"

    elif act_type == "align_objects":
        name = action.get("name")
        ref = action.get("reference_object")
        if not name or not str(name).strip():
            return False, "align_objects requires a valid 'name' attribute"
        if not ref or not str(ref).strip():
            return False, "align_objects requires a valid 'reference_object' attribute"
        if str(name).strip() == str(ref).strip():
            return False, "align_objects cannot align an object to itself"

        axes = action.get("axes", ["X", "Y"])
        if isinstance(axes, str):
            axes_list = [c.upper() for c in axes if c.upper() in {"X", "Y", "Z"}]
        elif isinstance(axes, (list, tuple)):
            axes_list = [str(a).upper() for a in axes]
        else:
            return False, "Parameter 'axes' must be a list or string of axes (e.g. ['X', 'Y'])"

        for a in axes_list:
            if a not in {"X", "Y", "Z"}:
                return False, f"Invalid axis '{a}'. Axes must be 'X', 'Y', or 'Z'"

        align = str(action.get("alignment", "center")).lower()
        if align not in VALID_ALIGNMENTS:
            return False, f"Invalid alignment '{align}'. Allowed: {sorted(list(VALID_ALIGNMENTS))}"

    elif act_type == "set_relative_scale":
        name = action.get("name")
        ref = action.get("reference_object")
        if not name or not str(name).strip():
            return False, "set_relative_scale requires a valid 'name' attribute"
        if not ref or not str(ref).strip():
            return False, "set_relative_scale requires a valid 'reference_object' attribute"

        factor = action.get("factor", 1.0)
        try:
            if float(factor) <= 0:
                return False, "Scale factor must be positive"
        except (ValueError, TypeError):
            return False, "Parameter 'factor' must be numeric"

        match_dim = str(action.get("match_dimension", "all")).lower()
        if match_dim not in {"width", "depth", "height", "max", "all", "x", "y", "z"}:
            return False, f"Invalid match_dimension '{match_dim}'"

    elif act_type == "verify_assembly":
        obj_a = action.get("object_a", action.get("name"))
        obj_b = action.get("object_b", action.get("reference_object"))
        if not obj_a or not str(obj_a).strip():
            return False, "verify_assembly requires 'object_a' (or 'name')"
        if not obj_b or not str(obj_b).strip():
            return False, "verify_assembly requires 'object_b' (or 'reference_object')"
        if str(obj_a).strip() == str(obj_b).strip():
            return False, "verify_assembly requires two distinct objects"

        if "tolerance" in action:
            try:
                if float(action["tolerance"]) < 0:
                    return False, "Parameter 'tolerance' must be non-negative"
            except (ValueError, TypeError):
                return False, "Parameter 'tolerance' must be numeric"

    return True, None


def validate_payload(payload):
    """Normalize and validate a payload which may be a single action, list of actions, or {'actions': [...]}."""
    if isinstance(payload, dict):
        if "actions" in payload:
            action_list = payload["actions"]
        elif "action" in payload:
            action_list = [payload]
        else:
            return False, "Payload dictionary must contain 'action' or 'actions' key", None
    elif isinstance(payload, list):
        action_list = payload
    else:
        return False, "Payload must be a dictionary or list of actions", None

    if not isinstance(action_list, list) or len(action_list) == 0:
        return False, "Action list is empty", None

    if len(action_list) > MAX_ACTIONS_PER_REQUEST:
        return False, f"Payload exceeds maximum allowed actions ({len(action_list)} > {MAX_ACTIONS_PER_REQUEST})", None

    for idx, act in enumerate(action_list):
        ok, err = validate_single_action(act)
        if not ok:
            return False, f"Action [{idx}] ({act.get('action', 'unknown')}): {err}", None

    return True, None, action_list
