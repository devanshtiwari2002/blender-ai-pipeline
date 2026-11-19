"""Shared Action Validator for Blender AI Assistant.
Runs in standard Python (venv) as well as inside Blender (no bpy dependency).
"""

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


def _validate_vec3(val, field_name="vector"):
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        return False, f"Field '{field_name}' must be a list/tuple of 3 numbers [x, y, z]"
    try:
        [float(x) for x in val]
    except (ValueError, TypeError):
        return False, f"All elements in '{field_name}' must be numeric"
    return True, None


def _validate_color(val):
    if not isinstance(val, (list, tuple)) or len(val) not in (3, 4):
        return False, "Color must be a list of 3 or 4 RGBA floats [r, g, b, (a)]"
    try:
        for c in val:
            v = float(c)
            if not (0.0 <= v <= 1.0):
                return False, f"Color channel {v} out of range [0.0, 1.0]"
    except (ValueError, TypeError):
        return False, "All color channels must be numeric"
    return True, None


def validate_single_action(action: dict):
    """Validate a single action dictionary against strict schema."""
    if not isinstance(action, dict):
        return False, "Action must be a JSON dictionary object"

    act_type = action.get("action")
    if not act_type:
        return False, "Missing required field: 'action'"

    if act_type not in SUPPORTED_ACTIONS:
        return False, f"Unsupported action: '{act_type}'. Allowed: {sorted(list(SUPPORTED_ACTIONS))}"

    # Object creation actions
    if act_type in {"create_cube", "create_cylinder", "create_sphere", "create_uv_sphere", "create_plane", "create_cone", "create_torus"}:
        if "location" in action:
            ok, err = _validate_vec3(action["location"], "location")
            if not ok:
                return False, err
        if "size" in action:
            try:
                if float(action["size"]) <= 0:
                    return False, "Parameter 'size' must be positive"
            except (ValueError, TypeError):
                return False, "Parameter 'size' must be numeric"
        if "radius" in action or "radius1" in action:
            r = action.get("radius", action.get("radius1"))
            try:
                if float(r) <= 0:
                    return False, "Radius must be positive"
            except (ValueError, TypeError):
                return False, "Radius must be numeric"
        if "depth" in action:
            try:
                if float(action["depth"]) <= 0:
                    return False, "Depth must be positive"
            except (ValueError, TypeError):
                return False, "Depth must be numeric"
        if "color" in action:
            ok, err = _validate_color(action["color"])
            if not ok:
                return False, err

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

    elif act_type in {"set_scale", "set_dimensions"}:
        field = "scale" if act_type == "set_scale" else "dimensions"
        if "name" not in action or not str(action["name"]).strip():
            return False, f"{act_type} requires a valid 'name' attribute"
        ok, err = _validate_vec3(action.get(field), field)
        if not ok:
            return False, err

    elif act_type == "rename_object":
        if not action.get("name") or not action.get("new_name"):
            return False, "rename_object requires 'name' and 'new_name'"

    elif act_type == "delete_object":
        if not action.get("name"):
            return False, "delete_object requires 'name'"

    elif act_type == "clear_scene":
        pass

    # Geometry
    elif act_type == "add_bevel":
        if not action.get("name"):
            return False, "add_bevel requires 'name'"
        if "width" in action:
            try:
                if float(action["width"]) <= 0:
                    return False, "Bevel 'width' must be positive"
            except (ValueError, TypeError):
                return False, "Bevel 'width' must be numeric"

    elif act_type == "set_shade_smooth":
        if not action.get("name"):
            return False, "set_shade_smooth requires 'name'"

    elif act_type == "apply_modifier":
        if not action.get("name"):
            return False, "apply_modifier requires 'name'"

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
