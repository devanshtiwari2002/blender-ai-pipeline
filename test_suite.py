"""Comprehensive Stability & Safety Test Suite for Blender AI Assistant.
Can be executed standalone via standard Python (venv) or inside Blender CLI:
  python test_suite.py
  blender --background --python test_suite.py
"""
import sys
import os
import json
import queue
import threading
import time
import math

# Ensure project root is on sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import action_validator
from blender_server import (
    BlenderAIServer,
    TASK_QUEUED,
    TASK_RUNNING,
    TASK_COMPLETED,
    TASK_CANCELLED,
    TASK_FAILED,
    DEFAULT_CHUNK_SIZE,
)


def test_action_validation():
    print("[RUNNING] test_action_validation...")
    # 1. Test single action
    ok, err, acts = action_validator.validate_payload({"action": "create_cube", "name": "Box", "size": 1.5})
    assert ok, f"Single action failed: {err}"
    assert len(acts) == 1

    # 2. Test list of actions
    payload = [
        {"action": "create_cylinder", "name": "Cyl", "location": [0, 0, 1], "radius": 0.5, "depth": 2.0},
        {"action": "create_plane", "name": "Ground", "size": 10.0, "color": [0.2, 0.4, 0.6, 1.0]},
        {"action": "create_torus", "name": "Ring", "major_radius": 1.5, "minor_radius": 0.3, "color": [0.9, 0.1, 0.2, 1.0]},
        {"action": "set_dimensions", "name": "Cyl", "dimensions": [1, 1, 2]},
        {"action": "add_bevel", "name": "Cyl", "width": 0.05, "segments": 3},
        {"action": "set_shade_smooth", "name": "Cyl", "smooth": True},
        {"action": "set_material", "name": "Cyl", "color": [0.8, 0.2, 0.1, 1.0], "metallic": 0.5, "roughness": 0.2},
        {"action": "get_scene_info"}
    ]
    ok, err, acts = action_validator.validate_payload(payload)
    assert ok, f"Action list failed: {err}"
    assert len(acts) == 8

    # 3. Test all 5 spatial actions
    spatial_payload = [
        {"action": "get_object_bounds", "name": "Cyl"},
        {"action": "place_relative", "name": "Cap", "reference_object": "Cyl", "relationship": "above", "gap": 0.0, "align_xy": True},
        {"action": "place_relative", "name": "Side", "reference_object": "Cyl", "relationship": "beside", "direction": "+X", "gap": 0.2},
        {"action": "align_objects", "name": "Cap", "reference_object": "Cyl", "axes": ["X", "Y"], "alignment": "center"},
        {"action": "set_relative_scale", "name": "Cap", "reference_object": "Cyl", "factor": 0.8, "match_dimension": "width"},
        {"action": "verify_assembly", "object_a": "Cap", "object_b": "Cyl", "expected_relationship": "above", "tolerance": 0.01}
    ]
    ok, err, acts = action_validator.validate_payload(spatial_payload)
    assert ok, f"Spatial actions failed: {err}"
    assert len(acts) == 6

    # 4. Test alias normalization for Group 2
    # Cylinder: depth -> height
    ok, err, acts = action_validator.validate_payload({"action": "create_cylinder", "depth": 4.5})
    assert ok and acts[0]["height"] == 4.5 and "depth" not in acts[0]

    # Cylinder: diameter -> radius
    ok, err, acts = action_validator.validate_payload({"action": "create_cylinder", "diameter": 3.0})
    assert ok and acts[0]["radius"] == 1.5 and "diameter" not in acts[0]

    # Sphere: diameter -> radius
    ok, err, acts = action_validator.validate_payload({"action": "create_sphere", "diameter": 5.0})
    assert ok and acts[0]["radius"] == 2.5 and "diameter" not in acts[0]

    # Cone: radius1 -> radius, radius2 -> radius_top, depth -> height
    ok, err, acts = action_validator.validate_payload({
        "action": "create_cone",
        "radius1": 1.5,
        "radius2": 0.5,
        "depth": 3.0,
        "vertices": 48
    })
    assert ok
    c_act = acts[0]
    assert c_act["radius"] == 1.5 and c_act["radius_top"] == 0.5 and c_act["height"] == 3.0 and c_act["vertices"] == 48

    # Cube: width, depth, height -> dimensions
    ok, err, acts = action_validator.validate_payload({
        "action": "create_cube",
        "width": 1.2,
        "depth": 3.4,
        "height": 5.6
    })
    assert ok and acts[0]["dimensions"] == [1.2, 3.4, 5.6]

    # Plane: width, depth -> dimensions
    ok, err, acts = action_validator.validate_payload({
        "action": "create_plane",
        "width": 8.0,
        "depth": 4.0
    })
    assert ok and acts[0]["dimensions"] == [8.0, 4.0]

    # 5. Group 3A: Planar set_dimensions and apply_transform validation
    ok, err, acts = action_validator.validate_payload({
        "action": "set_dimensions",
        "name": "Ground",
        "dimensions": [4.0, 3.0, 0.0]
    })
    assert ok, f"Planar dimensions failed: {err}"
    assert acts[0]["dimensions"] == [4.0, 3.0, 0.0]

    ok, err, acts = action_validator.validate_payload({
        "action": "apply_transform",
        "name": "Box",
        "scale": True,
        "rotation": False,
        "location": False
    })
    assert ok, f"apply_transform validation failed: {err}"

    # 6. Group 3B-1: Modifier validation (add_bevel with modifier_name, segment boundaries)
    ok, err, acts = action_validator.validate_payload({
        "action": "add_bevel",
        "name": "Box",
        "width": 0.05,
        "segments": 1,
        "modifier_name": "CustomBevel"
    })
    assert ok, f"add_bevel valid validation failed: {err}"

    ok, err, acts = action_validator.validate_payload({
        "action": "add_bevel",
        "name": "Box",
        "segments": 128
    })
    assert ok, f"add_bevel segments=128 failed: {err}"

    ok, err, acts = action_validator.validate_payload({
        "action": "set_shade_smooth",
        "name": "Box",
        "smooth": True
    })
    assert ok, f"set_shade_smooth validation failed: {err}"

    ok, err, acts = action_validator.validate_payload({
        "action": "apply_modifier",
        "name": "Box",
        "modifier_name": "CustomBevel"
    })
    assert ok, f"apply_modifier validation failed: {err}"

    # 7. Group 3B-2: Surface Quality validation (add_bevel limit_method, angle_limit, harden_normals; set_shade_smooth auto_smooth, angle)
    ok, err, acts = action_validator.validate_payload({
        "action": "add_bevel",
        "name": "Box",
        "limit_method": "angle",
        "angle_limit": 45.0,
        "harden_normals": True
    })
    assert ok, f"add_bevel angle validation failed: {err}"
    assert acts[0]["limit_method"] == "ANGLE", "limit_method must be normalized to uppercase"
    assert acts[0]["angle_limit"] == 45.0
    assert acts[0]["harden_normals"] is True

    ok, err, acts = action_validator.validate_payload({
        "action": "add_bevel",
        "name": "Box",
        "limit_method": "  none  ",
        "harden_normals": False
    })
    assert ok, f"add_bevel none validation failed: {err}"
    assert acts[0]["limit_method"] == "NONE", "limit_method must be normalized to uppercase"
    assert acts[0]["harden_normals"] is False

    ok, err, acts = action_validator.validate_payload({
        "action": "set_shade_smooth",
        "name": "Box",
        "smooth": True,
        "auto_smooth": True,
        "angle": 45.0
    })
    assert ok, f"set_shade_smooth auto_smooth validation failed: {err}"
    assert acts[0]["smooth"] is True
    assert acts[0]["auto_smooth"] is True
    assert acts[0]["angle"] == 45.0

    ok, err, acts = action_validator.validate_payload({
        "action": "set_shade_smooth",
        "name": "Box",
        "smooth": False,
        "auto_smooth": False
    })
    assert ok, f"set_shade_smooth flat validation failed: {err}"
    assert acts[0]["smooth"] is False
    assert acts[0]["auto_smooth"] is False

    print("  [PASSED] All action validation tests passed.")


def test_invalid_requests():
    print("[RUNNING] test_invalid_requests...")
    # 1. Unsupported action
    ok, err, _ = action_validator.validate_payload({"action": "unsupported_action_xyz"})
    assert not ok and "Unsupported action" in err

    # 2. Self-referencing in place_relative
    ok, err, _ = action_validator.validate_payload({
        "action": "place_relative",
        "name": "Bottle",
        "reference_object": "Bottle",
        "relationship": "above"
    })
    assert not ok and "relative to itself" in err

    # 3. Invalid relationship
    ok, err, _ = action_validator.validate_payload({
        "action": "place_relative",
        "name": "Cap",
        "reference_object": "Bottle",
        "relationship": "flying_over"
    })
    assert not ok and "Invalid relationship" in err

    # 4. Invalid beside direction
    ok, err, _ = action_validator.validate_payload({
        "action": "place_relative",
        "name": "Cap",
        "reference_object": "Bottle",
        "relationship": "beside",
        "direction": "diagonal"
    })
    assert not ok and "Invalid beside direction" in err

    # 5. Color out of range
    ok, err, _ = action_validator.validate_payload({
        "action": "set_material",
        "name": "Bottle",
        "color": [1.0, 2.5, 0.0, 1.0]
    })
    assert not ok and "out of range" in err

    # 6. Negative size
    ok, err, _ = action_validator.validate_payload({
        "action": "create_cube",
        "name": "Cube",
        "size": -2.0
    })
    assert not ok and "must be a positive number" in err

    # 7. Boolean numeric rejection
    ok, err, _ = action_validator.validate_payload({"action": "create_cube", "size": True})
    assert not ok and "positive number" in err

    ok, err, _ = action_validator.validate_payload({"action": "create_cylinder", "height": False})
    assert not ok and "positive number" in err

    # 8. NaN and infinity rejection
    import math
    ok, err, _ = action_validator.validate_payload({"action": "create_cube", "size": float("nan")})
    assert not ok and "positive number" in err

    ok, err, _ = action_validator.validate_payload({"action": "create_cylinder", "radius": float("inf")})
    assert not ok and "positive number" in err

    # 9. Mesh resolution validation (float, boolean, boundary)
    ok, err, _ = action_validator.validate_payload({"action": "create_cylinder", "vertices": 32.5})
    assert not ok and "must be an integer" in err

    ok, err, _ = action_validator.validate_payload({"action": "create_cylinder", "vertices": True})
    assert not ok and "must be an integer" in err

    ok, err, _ = action_validator.validate_payload({"action": "create_cylinder", "vertices": 2})
    assert not ok and "between 3 and 256" in err

    ok, err, _ = action_validator.validate_payload({"action": "create_cylinder", "vertices": 300})
    assert not ok and "between 3 and 256" in err

    ok, err, _ = action_validator.validate_payload({"action": "create_torus", "minor_segments": 200})
    assert not ok and "between 3 and 128" in err

    # 10. Torus radius constraint (major must be > minor)
    ok, err, _ = action_validator.validate_payload({
        "action": "create_torus",
        "name": "Torus",
        "major_radius": 0.2,
        "minor_radius": 0.5
    })
    assert not ok and "must be greater than" in err

    # 11. Alias conflict rejection
    ok, err, _ = action_validator.validate_payload({
        "action": "create_cylinder",
        "height": 2.0,
        "depth": 3.0
    })
    assert not ok and "Conflicting parameters 'height'" in err

    ok, err, _ = action_validator.validate_payload({
        "action": "create_cylinder",
        "radius": 1.0,
        "diameter": 3.0
    })
    assert not ok and "Conflicting parameters 'radius'" in err

    ok, err, _ = action_validator.validate_payload({
        "action": "create_cone",
        "radius": 1.0,
        "radius1": 2.0
    })
    assert not ok and "Conflicting parameters 'radius'" in err

    ok, err, _ = action_validator.validate_payload({
        "action": "create_cone",
        "radius_top": 0.5,
        "radius2": 1.0
    })
    assert not ok and "Conflicting parameters 'radius_top'" in err

    # 12. Unknown parameter rejection on primitives
    ok, err, _ = action_validator.validate_payload({
        "action": "create_cube",
        "unknown_flag": 123
    })
    assert not ok and "Unrecognized parameter 'unknown_flag'" in err

    ok, err, _ = action_validator.validate_payload({
        "action": "create_cylinder",
        "foo": "bar"
    })
    assert not ok and "Unrecognized parameter 'foo'" in err

    # 13. Malformed dimensions array
    ok, err, _ = action_validator.validate_payload({
        "action": "create_cube",
        "dimensions": [1.0, 2.0]
    })
    assert not ok and "must be a list/tuple of 3 positive numbers" in err

    ok, err, _ = action_validator.validate_payload({
        "action": "create_plane",
        "dimensions": [1.0, 2.0, 3.0]
    })
    assert not ok and "must be a list/tuple of 2 positive numbers" in err

    ok, err, _ = action_validator.validate_payload({
        "action": "create_cube",
        "dimensions": [1.0, -2.0, 3.0]
    })
    assert not ok and "must be a positive number" in err

    # 14. Exceeding max actions limit
    huge_payload = [{"action": "clear_scene"}] * (action_validator.MAX_ACTIONS_PER_REQUEST + 1)
    ok, err, _ = action_validator.validate_payload(huge_payload)
    assert not ok and "exceeds maximum allowed actions" in err

    # 15. Group 3A: Scale validation tests
    # Zero scale
    ok, err, _ = action_validator.validate_payload({"action": "set_scale", "name": "Box", "scale": [1.0, 0.0, 1.0]})
    assert not ok and "strictly positive" in err

    # Negative scale
    ok, err, _ = action_validator.validate_payload({"action": "set_scale", "name": "Box", "scale": [-1.0, 1.0, 1.0]})
    assert not ok and "strictly positive" in err

    # Boolean in scale
    ok, err, _ = action_validator.validate_payload({"action": "set_scale", "name": "Box", "scale": [1.0, True, 1.0]})
    assert not ok and ("strictly positive" in err or "finite numbers" in err)

    # NaN / Inf in scale
    import math
    ok, err, _ = action_validator.validate_payload({"action": "set_scale", "name": "Box", "scale": [float("nan"), 1.0, 1.0]})
    assert not ok and ("strictly positive" in err or "finite numbers" in err)

    ok, err, _ = action_validator.validate_payload({"action": "set_scale", "name": "Box", "scale": [1.0, float("inf"), 1.0]})
    assert not ok and ("strictly positive" in err or "finite numbers" in err)

    # Invalid scale array
    ok, err, _ = action_validator.validate_payload({"action": "set_scale", "name": "Box", "scale": [1.0, 2.0]})
    assert not ok and "must be a list/tuple of 3 numbers" in err

    # 16. Group 3A: Dimensions validation tests
    # All zero dimensions
    ok, err, _ = action_validator.validate_payload({"action": "set_dimensions", "name": "Box", "dimensions": [0.0, 0.0, 0.0]})
    assert not ok and ("At most one dimension" in err or "strictly positive" in err or "non-negative" in err)

    # Multiple zero dimensions (line)
    ok, err, _ = action_validator.validate_payload({"action": "set_dimensions", "name": "Box", "dimensions": [0.0, 0.0, 5.0]})
    assert not ok and "At most one dimension" in err

    # Negative dimensions
    ok, err, _ = action_validator.validate_payload({"action": "set_dimensions", "name": "Box", "dimensions": [4.0, -3.0, 0.0]})
    assert not ok and "non-negative" in err

    # Boolean in dimensions
    ok, err, _ = action_validator.validate_payload({"action": "set_dimensions", "name": "Box", "dimensions": [4.0, True, 0.0]})
    assert not ok and ("non-negative" in err or "finite numbers" in err)

    # NaN in dimensions
    ok, err, _ = action_validator.validate_payload({"action": "set_dimensions", "name": "Box", "dimensions": [float("nan"), 3.0, 0.0]})
    assert not ok and ("non-negative" in err or "finite numbers" in err)

    # 17. Group 3A: apply_transform validation tests
    # Missing name
    ok, err, _ = action_validator.validate_payload({"action": "apply_transform", "scale": True})
    assert not ok and "valid 'name'" in err

    # Non-boolean flags
    ok, err, _ = action_validator.validate_payload({"action": "apply_transform", "name": "Box", "scale": "yes"})
    assert not ok and "must be a boolean" in err

    ok, err, _ = action_validator.validate_payload({"action": "apply_transform", "name": "Box", "location": 1})
    assert not ok and "must be a boolean" in err

    # All false
    ok, err, _ = action_validator.validate_payload({"action": "apply_transform", "name": "Box", "location": False, "rotation": False, "scale": False})
    assert not ok and "at least one of" in err

    # Unrecognized parameter
    ok, err, _ = action_validator.validate_payload({"action": "apply_transform", "name": "Box", "unknown_opt": True})
    assert not ok and "Unrecognized parameter 'unknown_opt'" in err

    # 18. Group 3B-1: Modifier validation tests
    # add_bevel width tests
    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "width": 0.0})
    assert not ok and "strictly positive" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "width": -0.1})
    assert not ok and "strictly positive" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "width": True})
    assert not ok and "strictly positive" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "width": float("nan")})
    assert not ok and "strictly positive" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "width": float("inf")})
    assert not ok and "strictly positive" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "width": "0.1"})
    assert not ok and "strictly positive" in err

    # add_bevel segments tests
    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "segments": 0})
    assert not ok and "between 1 and 128" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "segments": -1})
    assert not ok and "between 1 and 128" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "segments": 3.5})
    assert not ok and "must be an integer" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "segments": True})
    assert not ok and "must be an integer" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "segments": "3"})
    assert not ok and "must be an integer" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "segments": 129})
    assert not ok and "between 1 and 128" in err

    # add_bevel modifier_name tests
    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "modifier_name": ""})
    assert not ok and "non-empty string" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "modifier_name": 123})
    assert not ok and "non-empty string" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "modifier_name": True})
    assert not ok and "non-empty string" in err

    # add_bevel unknown keys and missing name
    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "unknown_key": 1})
    assert not ok and "Unrecognized parameter 'unknown_key'" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "   "})
    assert not ok and "valid non-empty 'name'" in err

    # apply_modifier tests
    ok, err, _ = action_validator.validate_payload({"action": "apply_modifier", "name": "Box", "modifier_name": ""})
    assert not ok and "non-empty string" in err

    ok, err, _ = action_validator.validate_payload({"action": "apply_modifier", "name": "Box", "modifier_name": 123})
    assert not ok and "non-empty string" in err

    ok, err, _ = action_validator.validate_payload({"action": "apply_modifier", "name": "Box", "unknown_key": 1})
    assert not ok and "Unrecognized parameter 'unknown_key'" in err

    ok, err, _ = action_validator.validate_payload({"action": "apply_modifier", "name": ""})
    assert not ok and "valid non-empty 'name'" in err

    # set_shade_smooth tests
    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "smooth": "yes"})
    assert not ok and "must be a boolean" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "smooth": 1})
    assert not ok and "must be a boolean" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "unknown_key": 1})
    assert not ok and "Unrecognized parameter 'unknown_key'" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "   "})
    assert not ok and "valid non-empty 'name'" in err

    # Group 3B-2: Surface Quality invalid requests
    # add_bevel limit_method invalid tests
    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "limit_method": "WEIGHT"})
    assert not ok and "must be 'ANGLE' or 'NONE'" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "limit_method": ""})
    assert not ok and "must be a string" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "limit_method": 123})
    assert not ok and "must be a string" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "limit_method": True})
    assert not ok and "must be a string" in err

    # add_bevel angle_limit invalid tests (0 and 181 boundary, bool, nan, inf, string)
    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "angle_limit": 0})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "angle_limit": -10})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "angle_limit": 181})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "angle_limit": True})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "angle_limit": float("nan")})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "angle_limit": float("inf")})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "angle_limit": "45"})
    assert not ok and "greater than 0 and at most 180" in err

    # add_bevel harden_normals invalid tests (int, string rejection)
    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "harden_normals": 1})
    assert not ok and "must be a boolean" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "harden_normals": 0})
    assert not ok and "must be a boolean" in err

    ok, err, _ = action_validator.validate_payload({"action": "add_bevel", "name": "Box", "harden_normals": "true"})
    assert not ok and "must be a boolean" in err

    # set_shade_smooth auto_smooth invalid tests (int, string rejection)
    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "auto_smooth": 1})
    assert not ok and "must be a boolean" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "auto_smooth": 0})
    assert not ok and "must be a boolean" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "auto_smooth": "true"})
    assert not ok and "must be a boolean" in err

    # set_shade_smooth angle invalid tests (0 and 181 boundary, bool, nan, inf, string)
    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "angle": 0})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "angle": -5})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "angle": 181})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "angle": True})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "angle": float("nan")})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "angle": float("inf")})
    assert not ok and "greater than 0 and at most 180" in err

    ok, err, _ = action_validator.validate_payload({"action": "set_shade_smooth", "name": "Box", "angle": "30"})
    assert not ok and "greater than 0 and at most 180" in err

    print("  [PASSED] All invalid requests correctly rejected.")


def test_task_lifecycle_and_timeout_sync():
    """Verify thread-safe synchronization between socket listener and timer runner."""
    print("[RUNNING] test_task_lifecycle_and_timeout_sync...")

    original_actions_mod = sys.modules.get("blender_actions")
    try:
        # Scenario A: Task times out while still in TASK_QUEUED -> Must transition to TASK_CANCELLED and not execute
        executed_actions_a = []

        def mock_execute_a(act):
            executed_actions_a.append(act["name"])
            return {"status": "success"}

        class MockModuleA:
            execute_action = staticmethod(mock_execute_a)

        server_a = BlenderAIServer.__new__(BlenderAIServer)
        server_a.task_queue = queue.Queue()
        server_a.running = True
        server_a.current_task = None
        server_a.chunk_size = DEFAULT_CHUNK_SIZE

        # Create task A
        task_a = {
            "actions": [{"action": "create_cube", "name": "CancelledCube"}],
            "lock": threading.Lock(),
            "state": TASK_QUEUED,
            "event": threading.Event(),
            "result": None,
            "current_index": 0,
            "results_accumulator": [],
            "has_error": False
        }
        server_a.task_queue.put(task_a)

        # Simulate client socket timeout firing while task is queued
        with task_a["lock"]:
            if task_a["state"] == TASK_QUEUED:
                task_a["state"] = TASK_CANCELLED
                timeout_response_a = {"status": "error", "state": TASK_CANCELLED}

        assert timeout_response_a["state"] == TASK_CANCELLED

        # Simulate process_queue ticking afterwards
        sys.modules["blender_actions"] = MockModuleA
        interval = server_a.process_queue()
        assert interval == 0.1
        assert len(executed_actions_a) == 0, "Cancelled task was executed by runner!"
        assert task_a["state"] == TASK_CANCELLED

        # Scenario B: Task times out after reaching TASK_RUNNING -> Must NOT falsely claim cancelled
        task_b = {
            "actions": [{"action": "create_cube", "name": f"Cube_{i}"} for i in range(5)],
            "lock": threading.Lock(),
            "state": TASK_QUEUED,
            "event": threading.Event(),
            "result": None,
            "current_index": 0,
            "results_accumulator": [],
            "has_error": False
        }
        server_b = BlenderAIServer.__new__(BlenderAIServer)
        server_b.task_queue = queue.Queue()
        server_b.running = True
        server_b.current_task = None
        server_b.chunk_size = 2  # Small chunk size to pause mid-execution

        executed_actions_b = []

        def mock_execute_b(act):
            executed_actions_b.append(act["name"])
            return {"status": "success", "object": act["name"]}

        class MockModuleB:
            execute_action = staticmethod(mock_execute_b)

        sys.modules["blender_actions"] = MockModuleB
        server_b.task_queue.put(task_b)

        # First tick: pops task_b, transitions to TASK_RUNNING, executes first chunk (2 actions)
        ret_interval = server_b.process_queue()
        assert ret_interval == 0.01  # Yields to UI loop
        assert task_b["state"] == TASK_RUNNING
        assert len(executed_actions_b) == 2

        # Simulate socket listener timeout firing while task_b is TASK_RUNNING
        with task_b["lock"]:
            if task_b["state"] == TASK_QUEUED:
                task_b["state"] = TASK_CANCELLED
                timeout_response_b = {"status": "error", "state": TASK_CANCELLED}
            elif task_b["state"] == TASK_RUNNING:
                timeout_response_b = {"status": "timeout", "state": TASK_RUNNING}

        # Listener correctly identifies that the task is RUNNING, not cancelled
        assert timeout_response_b["status"] == "timeout"
        assert timeout_response_b["state"] == TASK_RUNNING
        assert task_b["state"] == TASK_RUNNING

        # Runner continues subsequent chunks
        ret_interval = server_b.process_queue()  # actions 2, 3
        assert ret_interval == 0.01
        ret_interval = server_b.process_queue()  # action 4
        assert ret_interval == 0.1  # Complete

        assert task_b["state"] == TASK_COMPLETED
        assert task_b["event"].is_set()
        assert len(executed_actions_b) == 5
        assert len(task_b["result"]["results"]) == 5

        print("  [PASSED] Task lifecycle & timeout synchronization verified.")
    finally:
        if original_actions_mod is not None:
            sys.modules["blender_actions"] = original_actions_mod
        else:
            sys.modules.pop("blender_actions", None)


def test_chunked_execution():
    """Verify bounded execution chunking and UI yield scheduling."""
    print("[RUNNING] test_chunked_execution...")

    original_actions_mod = sys.modules.get("blender_actions")
    try:
        server = BlenderAIServer.__new__(BlenderAIServer)
        server.task_queue = queue.Queue()
        server.running = True
        server.current_task = None
        server.chunk_size = 15

        total_actions = 35
        executed = []

        def mock_exec(act):
            executed.append(act["index"])
            return {"status": "success", "action": act["action"]}

        class MockActions:
            execute_action = staticmethod(mock_exec)

        sys.modules["blender_actions"] = MockActions

        actions_payload = [{"action": "get_scene_info", "index": i} for i in range(total_actions)]
        task = {
            "actions": actions_payload,
            "lock": threading.Lock(),
            "state": TASK_QUEUED,
            "event": threading.Event(),
            "result": None,
            "current_index": 0,
            "results_accumulator": [],
            "has_error": False
        }
        server.task_queue.put(task)

        # Chunk 1: 0..14 (15 items) -> returns 0.01s continuation
        inv1 = server.process_queue()
        assert inv1 == 0.01, f"Expected 0.01s yield, got {inv1}"
        assert task["current_index"] == 15
        assert len(executed) == 15
        assert not task["event"].is_set()

        # Chunk 2: 15..29 (15 items) -> returns 0.01s continuation
        inv2 = server.process_queue()
        assert inv2 == 0.01, f"Expected 0.01s yield, got {inv2}"
        assert task["current_index"] == 30
        assert len(executed) == 30
        assert not task["event"].is_set()

        # Chunk 3: 30..34 (5 items) -> finishes, returns 0.1s idle poll
        inv3 = server.process_queue()
        assert inv3 == 0.1, f"Expected 0.1s idle poll, got {inv3}"
        assert task["current_index"] == 35
        assert len(executed) == 35
        assert task["event"].is_set()
        assert task["state"] == TASK_COMPLETED
        assert task["result"]["status"] == "success"
        assert task["result"]["total_actions"] == 35
        assert len(task["result"]["results"]) == 35

        print("  [PASSED] Chunked execution and cooperative yielding verified.")
    finally:
        if original_actions_mod is not None:
            sys.modules["blender_actions"] = original_actions_mod
        else:
            sys.modules.pop("blender_actions", None)


def test_blender_live_integration():
    """Executed only when running within Blender environment."""
    try:
        import bpy
        import blender_actions
    except ImportError:
        print("[SKIPPED] Blender live integration test (run with 'blender --background --python test_suite.py')")
        return

    print("[RUNNING] test_blender_live_integration inside Blender...")

    # 1. Test clear scene
    blender_actions.execute_action({"action": "clear_scene"})

    # 2. Create base cylinder and neck
    res1 = blender_actions.execute_action({"action": "create_cylinder", "name": "Body", "location": [0, 0, 1.75], "radius": 0.75, "depth": 3.5})
    assert res1["status"] == "success"

    res2 = blender_actions.execute_action({"action": "create_cylinder", "name": "Neck", "location": [0, 0, 3.75], "radius": 0.35, "depth": 0.5})
    assert res2["status"] == "success"

    # 3. Create dropper at an arbitrary offset location
    res3 = blender_actions.execute_action({"action": "create_sphere", "name": "Dropper", "location": [2, 2, 0], "radius": 0.35})
    assert res3["status"] == "success"

    # 4. Place dropper relative to neck (above, gap=0.0, align_xy=True)
    res_place = blender_actions.execute_action({
        "action": "place_relative",
        "name": "Dropper",
        "reference_object": "Neck",
        "relationship": "above",
        "gap": 0.0,
        "align_xy": True
    })
    assert res_place["status"] == "success"

    # 5. Verify assembly
    res_verify = blender_actions.execute_action({
        "action": "verify_assembly",
        "object_a": "Dropper",
        "object_b": "Neck",
        "expected_relationship": "above",
        "tolerance": 0.005
    })
    assert res_verify["status"] == "success"
    assert res_verify["verified"] is True
    assert res_verify["vertical_gap"]["is_touching"] is True
    assert abs(res_verify["vertical_gap"]["a_above_b"]) < 1e-4

    # 6. Mode Safety & Preservation:
    # Switch active object to EDIT mode
    body_obj = bpy.data.objects["Body"]
    bpy.context.view_layer.objects.active = body_obj
    body_obj.select_set(True)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='EDIT')
        assert bpy.context.mode in {'EDIT', 'EDIT_MESH'}, f"Expected EDIT mode, got {bpy.context.mode}"

    # Priority 2 verification: read-only actions MUST preserve EDIT mode!
    res_scene = blender_actions.execute_action({"action": "get_scene_info"})
    assert res_scene["status"] == "success"
    assert bpy.context.mode in {'EDIT', 'EDIT_MESH'}, f"Read-only action disrupted mode to {bpy.context.mode}!"

    res_bounds = blender_actions.execute_action({"action": "get_object_bounds", "name": "Neck"})
    assert res_bounds["status"] == "success"
    assert bpy.context.mode in {'EDIT', 'EDIT_MESH'}, f"Read-only action disrupted mode to {bpy.context.mode}!"

    res_verify_mode = blender_actions.execute_action({
        "action": "verify_assembly",
        "object_a": "Dropper",
        "object_b": "Neck",
        "expected_relationship": "above"
    })
    assert res_verify_mode["status"] == "success"
    assert bpy.context.mode in {'EDIT', 'EDIT_MESH'}, f"Read-only action disrupted mode to {bpy.context.mode}!"

    # Object-mode-required action MUST safely switch mode to OBJECT
    res_create = blender_actions.execute_action({"action": "create_cube", "name": "ModeTestCube", "size": 1.0})
    assert res_create["status"] == "success"
    assert bpy.context.mode == 'OBJECT', f"Expected mode OBJECT after create_cube, got {bpy.context.mode}"

    # 7. Test 1 & 2: Duplicate Object Names (Same Primitive & Different Primitives)
    # Clear scene first
    blender_actions.execute_action({"action": "clear_scene"})

    res_box1 = blender_actions.execute_action({"action": "create_cube", "name": "Box", "location": [0, 0, 0], "size": 2.0})
    assert res_box1["status"] == "success"
    assert res_box1["object"] == "Box"
    assert "Box" in bpy.data.objects

    # Second cube with duplicate requested name "Box" -> Blender renames to "Box.001"
    res_box2 = blender_actions.execute_action({"action": "create_cube", "name": "Box", "location": [0, 0, 0], "size": 2.0})
    assert res_box2["status"] == "success"
    assert res_box2["object"] == "Box.001", f"Expected 'Box.001', got '{res_box2['object']}'"
    assert "Box.001" in bpy.data.objects

    # Target Box.001 specifically with set_location
    res_move = blender_actions.execute_action({"action": "set_location", "name": "Box.001", "location": [10, 10, 10]})
    assert res_move["status"] == "success"
    assert res_move["object"] == "Box.001"
    # Ensure Box was not moved, while Box.001 was moved
    assert list(bpy.data.objects["Box"].location) == [0.0, 0.0, 0.0]
    assert list(bpy.data.objects["Box.001"].location) == [10.0, 10.0, 10.0]

    # Third object: different primitive (cylinder) with duplicate name "Box" -> Blender renames to "Box.002"
    res_box3 = blender_actions.execute_action({"action": "create_cylinder", "name": "Box", "location": [0, 0, 5], "radius": 1.0, "depth": 2.0})
    assert res_box3["status"] == "success"
    assert res_box3["object"] == "Box.002", f"Expected 'Box.002', got '{res_box3['object']}'"
    assert "Box.002" in bpy.data.objects
    assert bpy.data.objects["Box.002"].type == 'MESH'

    # 8. Test 3: apply_modifier to initially unselected object
    res_mod_box = blender_actions.execute_action({"action": "create_cube", "name": "ModBox", "size": 2.0})
    assert res_mod_box["status"] == "success"
    res_add_bev = blender_actions.execute_action({"action": "add_bevel", "name": "ModBox", "width": 0.1, "segments": 2})
    assert res_add_bev["status"] == "success"
    assert "Bevel" in bpy.data.objects["ModBox"].modifiers

    # Create another object and explicitly deselect ModBox
    res_other = blender_actions.execute_action({"action": "create_cube", "name": "OtherObj", "size": 1.0})
    other_obj = bpy.data.objects["OtherObj"]
    mod_box_obj = bpy.data.objects["ModBox"]
    mod_box_obj.select_set(False)
    bpy.context.view_layer.objects.active = other_obj
    other_obj.select_set(True)
    assert bpy.context.view_layer.objects.active.name != "ModBox"
    assert not mod_box_obj.select_get()

    # Apply modifier to unselected ModBox
    res_apply = blender_actions.execute_action({"action": "apply_modifier", "name": "ModBox", "modifier_name": "Bevel"})
    assert res_apply["status"] == "success", f"apply_modifier failed: {res_apply}"
    assert "Bevel" not in bpy.data.objects["ModBox"].modifiers
    # Verify selection state was restored
    assert bpy.context.view_layer.objects.active == other_obj
    assert other_obj.select_get()
    assert not mod_box_obj.select_get(), "Target ModBox must not remain selected after apply_modifier!"

    # 9. Test 4 & 5: create_plane and create_torus with color
    res_plane = blender_actions.execute_action({
        "action": "create_plane",
        "name": "ColoredPlane",
        "location": [0, 0, -1],
        "size": 5.0,
        "color": [0.1, 0.7, 0.3, 1.0]
    })
    assert res_plane["status"] == "success"
    assert "ColoredPlane" in bpy.data.objects
    plane_obj = bpy.data.objects["ColoredPlane"]
    assert len(plane_obj.data.materials) > 0
    mat_plane = plane_obj.data.materials[0]
    assert mat_plane.name == "ColoredPlane_Material"
    assert abs(mat_plane.diffuse_color[0] - 0.1) < 1e-3
    assert abs(mat_plane.diffuse_color[1] - 0.7) < 1e-3
    assert abs(mat_plane.diffuse_color[2] - 0.3) < 1e-3

    res_torus = blender_actions.execute_action({
        "action": "create_torus",
        "name": "ColoredTorus",
        "location": [0, 0, 2],
        "major_radius": 1.5,
        "minor_radius": 0.3,
        "color": [0.8, 0.2, 0.5, 1.0]
    })
    assert res_torus["status"] == "success"
    assert "ColoredTorus" in bpy.data.objects
    torus_obj = bpy.data.objects["ColoredTorus"]
    assert len(torus_obj.data.materials) > 0
    mat_torus = torus_obj.data.materials[0]
    assert mat_torus.name == "ColoredTorus_Material"
    assert abs(mat_torus.diffuse_color[0] - 0.8) < 1e-3
    assert abs(mat_torus.diffuse_color[1] - 0.2) < 1e-3
    assert abs(mat_torus.diffuse_color[2] - 0.5) < 1e-3

    # 10. Test 6: Standardized Primitive Response Schema
    for res_obj, exp_act in [(res_plane, "create_plane"), (res_torus, "create_torus"), (res_box1, "create_cube")]:
        for field in ("status", "action", "object", "message", "location", "rotation", "scale", "dimensions", "bounds"):
            assert field in res_obj, f"Missing field '{field}' in {exp_act} response"
        assert res_obj["status"] == "success"
        assert res_obj["action"] == exp_act
        assert isinstance(res_obj["location"], list) and len(res_obj["location"]) == 3
        assert isinstance(res_obj["rotation"], list) and len(res_obj["rotation"]) == 3
        assert isinstance(res_obj["scale"], list) and len(res_obj["scale"]) == 3
        assert isinstance(res_obj["dimensions"], list) and len(res_obj["dimensions"]) == 3
        assert "min" in res_obj["bounds"] and "max" in res_obj["bounds"] and "center" in res_obj["bounds"]

    # 11. Test 7: Standardized Transform Response Schema
    res_set_loc = blender_actions.execute_action({"action": "set_location", "name": "Box", "location": [3, 4, 5]})
    assert res_set_loc["status"] == "success"
    assert res_set_loc["action"] == "set_location"
    assert res_set_loc["location"] == [3.0, 4.0, 5.0]
    assert res_set_loc["bounds"]["center"] == [3.0, 4.0, 5.0]

    res_set_rot = blender_actions.execute_action({"action": "set_rotation", "name": "Box", "rotation": [0, 45, 90]})
    assert res_set_rot["status"] == "success"
    assert res_set_rot["action"] == "set_rotation"
    assert res_set_rot["rotation"] == [0.0, 45.0, 90.0]

    res_set_scale = blender_actions.execute_action({"action": "set_scale", "name": "Box", "scale": [2, 1, 0.5]})
    assert res_set_scale["status"] == "success"
    assert res_set_scale["action"] == "set_scale"
    assert res_set_scale["scale"] == [2.0, 1.0, 0.5]

    res_set_dims = blender_actions.execute_action({"action": "set_dimensions", "name": "Box", "dimensions": [4, 4, 4]})
    assert res_set_dims["status"] == "success"
    assert res_set_dims["action"] == "set_dimensions"
    assert res_set_dims["dimensions"] == [4.0, 4.0, 4.0]
    assert "min" in res_set_dims["bounds"] and "max" in res_set_dims["bounds"] and "center" in res_set_dims["bounds"]

    # 12. Group 2 Integration Tests:
    blender_actions.execute_action({"action": "clear_scene"})

    # Test 1: Non-uniform cube dimensions
    res_nu_cube = blender_actions.execute_action({
        "action": "create_cube",
        "name": "NUCube",
        "dimensions": [1.5, 2.5, 3.5]
    })
    assert res_nu_cube["status"] == "success"
    nu_obj = bpy.data.objects["NUCube"]
    assert abs(nu_obj.dimensions.x - 1.5) < 1e-3
    assert abs(nu_obj.dimensions.y - 2.5) < 1e-3
    assert abs(nu_obj.dimensions.z - 3.5) < 1e-3

    # Test 2: Rectangular plane dimensions & flat geometry
    res_rect_plane = blender_actions.execute_action({
        "action": "create_plane",
        "name": "RectPlane",
        "dimensions": [4.0, 2.0]
    })
    assert res_rect_plane["status"] == "success"
    rp_obj = bpy.data.objects["RectPlane"]
    assert abs(rp_obj.dimensions.x - 4.0) < 1e-3
    assert abs(rp_obj.dimensions.y - 2.0) < 1e-3
    assert abs(rp_obj.dimensions.z - 0.0) < 1e-4

    # Test 3: Cylinder with canonical height and resolution
    res_cyl_can = blender_actions.execute_action({
        "action": "create_cylinder",
        "name": "CanCyl",
        "radius": 0.75,
        "height": 3.5,
        "vertices": 64
    })
    assert res_cyl_can["status"] == "success"
    cyl_obj = bpy.data.objects["CanCyl"]
    assert abs(cyl_obj.dimensions.z - 3.5) < 1e-3
    assert abs(cyl_obj.dimensions.x - 1.5) < 1e-3
    # 64 top rim + 64 bottom rim = 128
    assert len(cyl_obj.data.vertices) == 128

    # Test 4: Cylinder using legacy depth
    res_cyl_leg = blender_actions.execute_action({
        "action": "create_cylinder",
        "name": "LegCyl",
        "radius": 0.5,
        "depth": 4.0
    })
    assert res_cyl_leg["status"] == "success"
    assert abs(bpy.data.objects["LegCyl"].dimensions.z - 4.0) < 1e-3

    # Test 5: Cylinder diameter normalization (via validator + handler)
    ok, _, norm_cyl = action_validator.validate_payload({"action": "create_cylinder", "name": "DiaCyl", "diameter": 3.0, "height": 2.0})
    assert ok
    res_cyl_dia = blender_actions.execute_action(norm_cyl[0])
    assert res_cyl_dia["status"] == "success"
    assert abs(bpy.data.objects["DiaCyl"].dimensions.x - 3.0) < 1e-3

    # Test 6: Cylinder custom vertex count (hexagonal prism)
    res_hex = blender_actions.execute_action({
        "action": "create_cylinder",
        "name": "HexPrism",
        "radius": 1.0,
        "height": 2.0,
        "vertices": 6
    })
    assert res_hex["status"] == "success"
    # 6 top rim + 6 bottom rim = 12 vertices
    assert len(bpy.data.objects["HexPrism"].data.vertices) == 12

    # Test 7: UV sphere custom segments & rings
    res_sph_res = blender_actions.execute_action({
        "action": "create_sphere",
        "name": "HiResSphere",
        "radius": 1.0,
        "segments": 40,
        "ring_count": 20
    })
    assert res_sph_res["status"] == "success"
    sph_obj = bpy.data.objects["HiResSphere"]
    # 40 * 19 + 2 = 762 vertices
    assert len(sph_obj.data.vertices) == 762

    # Test 8: Sphere diameter normalization
    ok, _, norm_sph = action_validator.validate_payload({"action": "create_sphere", "name": "DiaSph", "diameter": 4.0})
    assert ok
    res_sph_dia = blender_actions.execute_action(norm_sph[0])
    assert res_sph_dia["status"] == "success"
    assert abs(bpy.data.objects["DiaSph"].dimensions.x - 4.0) < 1e-3

    # Test 9: Pointed cone (radius_top=0.0)
    res_pt_cone = blender_actions.execute_action({
        "action": "create_cone",
        "name": "PointedCone",
        "radius": 1.0,
        "radius_top": 0.0,
        "height": 3.0,
        "vertices": 32
    })
    assert res_pt_cone["status"] == "success"
    # 32 base rim + 1 apex = 33 vertices
    assert len(bpy.data.objects["PointedCone"].data.vertices) == 33

    # Test 10: Truncated cone / frustum (radius_top=1.0)
    res_tr_cone = blender_actions.execute_action({
        "action": "create_cone",
        "name": "Frustum",
        "radius": 2.0,
        "radius_top": 1.0,
        "height": 3.0,
        "vertices": 32
    })
    assert res_tr_cone["status"] == "success"
    frust_obj = bpy.data.objects["Frustum"]
    # 32 base rim + 32 top rim = 64 vertices
    assert len(frust_obj.data.vertices) == 64
    assert abs(frust_obj.dimensions.z - 3.0) < 1e-3
    assert abs(frust_obj.dimensions.x - 4.0) < 1e-3

    # Test 11: Cone custom resolution
    res_cone_16 = blender_actions.execute_action({
        "action": "create_cone",
        "name": "Cone16",
        "vertices": 16
    })
    assert res_cone_16["status"] == "success"
    # 16 base rim + 1 apex = 17 vertices
    assert len(bpy.data.objects["Cone16"].data.vertices) == 17

    # Test 12: Torus custom topology
    res_torus_res = blender_actions.execute_action({
        "action": "create_torus",
        "name": "CustomTorus",
        "major_radius": 2.0,
        "minor_radius": 0.5,
        "major_segments": 32,
        "minor_segments": 16
    })
    assert res_torus_res["status"] == "success"
    t_obj = bpy.data.objects["CustomTorus"]
    # 32 * 16 = 512 vertices
    assert len(t_obj.data.vertices) == 512
    # dimensions = 2 * (2 + 0.5) = 5.0 in X and Y, 2 * 0.5 = 1.0 in Z
    assert abs(t_obj.dimensions.x - 5.0) < 1e-3
    assert abs(t_obj.dimensions.y - 5.0) < 1e-3
    assert abs(t_obj.dimensions.z - 1.0) < 1e-3

    # -----------------------------------------------------------------------
    # 13. Group 3A Integration Tests
    # -----------------------------------------------------------------------
    blender_actions.execute_action({"action": "clear_scene"})

    # Test 1: Transform actions executed while initially in EDIT mode
    res_box = blender_actions.execute_action({"action": "create_cube", "name": "EditModeCube", "size": 2.0})
    cube_obj = bpy.data.objects["EditModeCube"]
    bpy.context.view_layer.objects.active = cube_obj
    cube_obj.select_set(True)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='EDIT')
    assert bpy.context.mode in {'EDIT', 'EDIT_MESH'}

    # set_location from EDIT mode -> should safely switch to OBJECT mode and apply location
    res_loc = blender_actions.execute_action({"action": "set_location", "name": "EditModeCube", "location": [5.0, 5.0, 5.0]})
    assert res_loc["status"] == "success"
    assert bpy.context.mode == 'OBJECT'
    assert list(cube_obj.location) == [5.0, 5.0, 5.0]

    # Re-enter EDIT mode and test set_rotation, set_scale, set_dimensions
    bpy.ops.object.mode_set(mode='EDIT')
    res_rot = blender_actions.execute_action({"action": "set_rotation", "name": "EditModeCube", "rotation": [0, 45, 90]})
    assert res_rot["status"] == "success"
    assert bpy.context.mode == 'OBJECT'

    bpy.ops.object.mode_set(mode='EDIT')
    res_scale = blender_actions.execute_action({"action": "set_scale", "name": "EditModeCube", "scale": [2.0, 1.0, 0.5]})
    assert res_scale["status"] == "success"
    assert bpy.context.mode == 'OBJECT'
    assert list(cube_obj.scale) == [2.0, 1.0, 0.5]

    bpy.ops.object.mode_set(mode='EDIT')
    res_dims = blender_actions.execute_action({"action": "set_dimensions", "name": "EditModeCube", "dimensions": [4.0, 2.0, 1.0]})
    assert res_dims["status"] == "success"
    assert bpy.context.mode == 'OBJECT'
    assert abs(cube_obj.dimensions.x - 4.0) < 1e-3
    assert abs(cube_obj.dimensions.y - 2.0) < 1e-3
    assert abs(cube_obj.dimensions.z - 1.0) < 1e-3

    # Test 2: Planar dimensions on Plane
    res_plane = blender_actions.execute_action({"action": "create_plane", "name": "PlanarTestPlane", "size": 2.0})
    assert res_plane["status"] == "success"
    res_plane_dims = blender_actions.execute_action({"action": "set_dimensions", "name": "PlanarTestPlane", "dimensions": [6.0, 3.0, 0.0]})
    assert res_plane_dims["status"] == "success"
    pl_obj = bpy.data.objects["PlanarTestPlane"]
    assert abs(pl_obj.dimensions.x - 6.0) < 1e-3
    assert abs(pl_obj.dimensions.y - 3.0) < 1e-3
    assert abs(pl_obj.dimensions.z - 0.0) < 1e-3
    assert res_plane_dims["bounds"]["min"][2] == res_plane_dims["bounds"]["max"][2]

    # Test 3: Attempting zero Z-dimension on 3D Cube -> should fail
    res_fail_zero = blender_actions.execute_action({"action": "set_dimensions", "name": "EditModeCube", "dimensions": [4.0, 2.0, 0.0]})
    assert res_fail_zero["status"] == "error"
    assert "non-planar axis" in res_fail_zero["error"].lower() or "cannot set zero dimension" in res_fail_zero["error"].lower()

    # Test 4: apply_transform(scale=True) on non-uniformly scaled cube
    blender_actions.execute_action({"action": "create_cube", "name": "ScaleApplyCube", "size": 2.0})
    blender_actions.execute_action({"action": "set_scale", "name": "ScaleApplyCube", "scale": [2.0, 0.5, 3.0]})
    sac_obj = bpy.data.objects["ScaleApplyCube"]
    assert list(sac_obj.scale) == [2.0, 0.5, 3.0]
    expected_dims = [4.0, 1.0, 6.0]
    for idx, exp_d in enumerate(expected_dims):
        assert abs(sac_obj.dimensions[idx] - exp_d) < 1e-3

    # Execute apply_transform
    res_app_scale = blender_actions.execute_action({"action": "apply_transform", "name": "ScaleApplyCube", "scale": True})
    assert res_app_scale["status"] == "success"
    assert res_app_scale["action"] == "apply_transform"
    # Scale must now be [1.0, 1.0, 1.0]
    assert list(sac_obj.scale) == [1.0, 1.0, 1.0]
    # Dimensions and world-space bounds must remain unchanged
    for idx, exp_d in enumerate(expected_dims):
        assert abs(sac_obj.dimensions[idx] - exp_d) < 1e-3
    assert abs(res_app_scale["bounds"]["max"][0] - 2.0) < 1e-3
    assert abs(res_app_scale["bounds"]["min"][0] - (-2.0)) < 1e-3

    # Test 5: apply_transform with rotation and scale independently
    blender_actions.execute_action({"action": "create_cube", "name": "RotScaleCube", "size": 2.0})
    blender_actions.execute_action({"action": "set_rotation", "name": "RotScaleCube", "rotation": [0, 0, 45]})
    blender_actions.execute_action({"action": "set_scale", "name": "RotScaleCube", "scale": [2.0, 2.0, 1.0]})
    rsc_obj = bpy.data.objects["RotScaleCube"]

    # Apply only rotation
    res_app_rot = blender_actions.execute_action({"action": "apply_transform", "name": "RotScaleCube", "rotation": True, "scale": False})
    assert res_app_rot["status"] == "success"
    assert list(rsc_obj.rotation_euler) == [0.0, 0.0, 0.0]
    assert list(rsc_obj.scale) == [2.0, 2.0, 1.0]

    # Apply only scale
    res_app_sc2 = blender_actions.execute_action({"action": "apply_transform", "name": "RotScaleCube", "rotation": False, "scale": True})
    assert res_app_sc2["status"] == "success"
    assert list(rsc_obj.scale) == [1.0, 1.0, 1.0]

    # Test 6: Selection and active-object context restoration
    blender_actions.execute_action({"action": "create_cube", "name": "ActiveObj", "size": 1.0})
    blender_actions.execute_action({"action": "create_cube", "name": "TargetObj", "size": 1.0})
    blender_actions.execute_action({"action": "set_scale", "name": "TargetObj", "scale": [3.0, 1.0, 1.0]})
    active_obj = bpy.data.objects["ActiveObj"]
    target_obj = bpy.data.objects["TargetObj"]

    # Explicitly set ActiveObj as active and selected, TargetObj unselected
    target_obj.select_set(False)
    bpy.context.view_layer.objects.active = active_obj
    active_obj.select_set(True)
    assert bpy.context.view_layer.objects.active.name == "ActiveObj"
    assert not target_obj.select_get()

    res_app_ctx = blender_actions.execute_action({"action": "apply_transform", "name": "TargetObj", "scale": True})
    assert res_app_ctx["status"] == "success"
    assert list(target_obj.scale) == [1.0, 1.0, 1.0]
    # Prior active object and selection restored
    assert bpy.context.view_layer.objects.active.name == "ActiveObj"
    assert active_obj.select_get()
    assert not target_obj.select_get(), "TargetObj must remain unselected after apply_transform!"

    # Test 7: Context restoration when previous active object was None
    bpy.context.view_layer.objects.active = None
    target_obj.select_set(False)
    active_obj.select_set(False)
    assert bpy.context.view_layer.objects.active is None
    assert not target_obj.select_get()

    blender_actions.execute_action({"action": "set_scale", "name": "TargetObj", "scale": [2.0, 2.0, 2.0]})
    res_app_none = blender_actions.execute_action({"action": "apply_transform", "name": "TargetObj", "scale": True})
    assert res_app_none["status"] == "success"
    assert list(target_obj.scale) == [1.0, 1.0, 1.0]
    # Active must be restored to None and TargetObj must not be left selected
    assert bpy.context.view_layer.objects.active is None
    assert not target_obj.select_get()

    # Test 8: apply_transform with location=True (baking location into vertices)
    blender_actions.execute_action({"action": "create_cube", "name": "LocCube", "location": [3.0, 4.0, 5.0], "size": 2.0})
    loc_obj = bpy.data.objects["LocCube"]
    assert list(loc_obj.location) == [3.0, 4.0, 5.0]

    res_app_loc = blender_actions.execute_action({
        "action": "apply_transform",
        "name": "LocCube",
        "location": True,
        "rotation": False,
        "scale": False
    })
    assert res_app_loc["status"] == "success"
    # Object location is reset to origin
    assert list(loc_obj.location) == [0.0, 0.0, 0.0]
    # World bounds center remains [3.0, 4.0, 5.0]
    assert res_app_loc["bounds"]["center"] == [3.0, 4.0, 5.0]
    # Raw mesh vertices in local coordinates now reflect the baked offset
    xs = [v.co.x for v in loc_obj.data.vertices]
    assert abs(min(xs) - 2.0) < 1e-3 and abs(max(xs) - 4.0) < 1e-3

    # Test 9: apply_transform from EDIT mode
    blender_actions.execute_action({"action": "create_cube", "name": "EditTransformCube", "size": 2.0})
    blender_actions.execute_action({"action": "set_scale", "name": "EditTransformCube", "scale": [2.0, 3.0, 4.0]})
    etc_obj = bpy.data.objects["EditTransformCube"]
    bpy.context.view_layer.objects.active = etc_obj
    etc_obj.select_set(True)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='EDIT')
    assert bpy.context.mode in {'EDIT', 'EDIT_MESH'}

    res_app_edit = blender_actions.execute_action({"action": "apply_transform", "name": "EditTransformCube", "scale": True})
    assert res_app_edit["status"] == "success"
    assert bpy.context.mode == 'OBJECT'
    assert list(etc_obj.scale) == [1.0, 1.0, 1.0]

    # Test 10: apply_modifier context restoration with active=None
    blender_actions.execute_action({"action": "create_cube", "name": "NoneActiveModBox", "size": 2.0})
    blender_actions.execute_action({"action": "add_bevel", "name": "NoneActiveModBox", "width": 0.1, "segments": 2})
    none_mod_obj = bpy.data.objects["NoneActiveModBox"]
    none_mod_obj.select_set(False)
    bpy.context.view_layer.objects.active = None
    assert bpy.context.view_layer.objects.active is None

    res_app_mod_none = blender_actions.execute_action({"action": "apply_modifier", "name": "NoneActiveModBox", "modifier_name": "Bevel"})
    assert res_app_mod_none["status"] == "success"
    assert bpy.context.view_layer.objects.active is None
    assert not none_mod_obj.select_get()

    # Test 11: Failure handling with non-existent object
    res_app_fail = blender_actions.execute_action({"action": "apply_transform", "name": "NonExistentObjectXYZ", "scale": True})
    assert res_app_fail["status"] == "error"
    assert "not found" in res_app_fail["error"].lower()

    # -----------------------------------------------------------------------
    # 14. Group 3B-1 Integration Tests: Modifier Hardening
    # -----------------------------------------------------------------------
    blender_actions.execute_action({"action": "clear_scene"})

    # Test 1: Multiple Bevel Modifiers on same object with auto-suffix and custom names
    res_bev_cube = blender_actions.execute_action({"action": "create_cube", "name": "MultiBevelCube", "size": 2.0})
    bev_cube = bpy.data.objects["MultiBevelCube"]

    # 1st bevel (default name "Bevel")
    res_b1 = blender_actions.execute_action({"action": "add_bevel", "name": "MultiBevelCube", "width": 0.05, "segments": 2})
    assert res_b1["status"] == "success"
    assert res_b1["modifier_name"] == "Bevel"
    assert res_b1["width"] == 0.05
    assert res_b1["segments"] == 2

    # 2nd bevel (default name "Bevel" -> Blender assigns "Bevel.001")
    res_b2 = blender_actions.execute_action({"action": "add_bevel", "name": "MultiBevelCube", "width": 0.02, "segments": 4})
    assert res_b2["status"] == "success"
    assert res_b2["modifier_name"] == "Bevel.001"
    assert res_b2["width"] == 0.02
    assert res_b2["segments"] == 4

    # 3rd bevel with custom name "CustomBevel"
    res_b3 = blender_actions.execute_action({"action": "add_bevel", "name": "MultiBevelCube", "modifier_name": "CustomBevel", "width": 0.01, "segments": 1})
    assert res_b3["status"] == "success"
    assert res_b3["modifier_name"] == "CustomBevel"

    # 4th bevel with duplicate custom name "CustomBevel" -> Blender assigns "CustomBevel.001"
    res_b4 = blender_actions.execute_action({"action": "add_bevel", "name": "MultiBevelCube", "modifier_name": "CustomBevel", "width": 0.005, "segments": 1})
    assert res_b4["status"] == "success"
    assert res_b4["modifier_name"] == "CustomBevel.001"

    # Verify all 4 modifiers exist on the object
    mod_names = [m.name for m in bev_cube.modifiers]
    assert mod_names == ["Bevel", "Bevel.001", "CustomBevel", "CustomBevel.001"]

    # Apply exact modifier using the returned modifier_name "Bevel.001"
    res_app_b2 = blender_actions.execute_action({"action": "apply_modifier", "name": "MultiBevelCube", "modifier_name": res_b2["modifier_name"]})
    assert res_app_b2["status"] == "success"
    assert res_app_b2["modifier_name"] == "Bevel.001"

    # Verify "Bevel.001" was applied and removed, while the other 3 remain
    mod_names_after = [m.name for m in bev_cube.modifiers]
    assert "Bevel.001" not in mod_names_after
    assert "Bevel" in mod_names_after
    assert "CustomBevel" in mod_names_after
    assert "CustomBevel.001" in mod_names_after

    # Test 2: set_shade_smooth context isolation with multiple selected objects
    blender_actions.execute_action({"action": "create_cube", "name": "ShadeObjA", "size": 1.0})
    blender_actions.execute_action({"action": "create_cube", "name": "ShadeObjB", "size": 1.0})
    blender_actions.execute_action({"action": "create_cube", "name": "ShadeObjC", "size": 1.0})
    obj_a = bpy.data.objects["ShadeObjA"]
    obj_b = bpy.data.objects["ShadeObjB"]
    obj_c = bpy.data.objects["ShadeObjC"]

    # Select ObjA and ObjB, set ObjA active, ObjC unselected
    obj_c.select_set(False)
    obj_b.select_set(True)
    obj_a.select_set(True)
    bpy.context.view_layer.objects.active = obj_a
    assert bpy.context.view_layer.objects.active.name == "ShadeObjA"
    assert obj_a.select_get()
    assert obj_b.select_get()
    assert not obj_c.select_get()

    # Call set_shade_smooth on ObjC
    res_smooth = blender_actions.execute_action({"action": "set_shade_smooth", "name": "ShadeObjC", "smooth": True})
    assert res_smooth["status"] == "success"
    assert res_smooth["action"] == "set_shade_smooth"
    assert res_smooth["smooth"] is True

    # Verify context restoration: ObjA remains active and selected, ObjB remains selected, ObjC remains unselected
    assert bpy.context.view_layer.objects.active.name == "ShadeObjA"
    assert obj_a.select_get()
    assert obj_b.select_get()
    assert not obj_c.select_get(), "Target ObjC must not be left selected after set_shade_smooth!"

    # Test flat shading path
    res_flat = blender_actions.execute_action({"action": "set_shade_smooth", "name": "ShadeObjC", "smooth": False})
    assert res_flat["status"] == "success"
    assert res_flat["smooth"] is False
    assert bpy.context.view_layer.objects.active.name == "ShadeObjA"
    assert not obj_c.select_get()

    # Test 3: Failure paths on missing objects/modifiers
    # Non-existent modifier on object
    res_fail_mod = blender_actions.execute_action({"action": "apply_modifier", "name": "MultiBevelCube", "modifier_name": "NonExistentMod"})
    assert res_fail_mod["status"] == "error"
    assert "not found" in res_fail_mod["error"].lower()

    # Non-existent object in add_bevel
    res_fail_bev = blender_actions.execute_action({"action": "add_bevel", "name": "GhostObjectXYZ"})
    assert res_fail_bev["status"] == "error"
    assert "not found" in res_fail_bev["error"].lower()

    # Non-existent object in set_shade_smooth
    res_fail_smooth = blender_actions.execute_action({"action": "set_shade_smooth", "name": "GhostObjectXYZ"})
    assert res_fail_smooth["status"] == "error"
    assert "not found" in res_fail_smooth["error"].lower()

    # Non-mesh object in add_bevel (Camera)
    blender_actions.execute_action({"action": "create_camera", "name": "NonMeshCamera"})
    res_fail_camera = blender_actions.execute_action({"action": "add_bevel", "name": "NonMeshCamera"})
    assert res_fail_camera["status"] == "error"
    assert "not a mesh" in res_fail_camera["error"].lower()

    # -----------------------------------------------------------------------
    # Group 3B-2: Surface Quality live integration tests
    # -----------------------------------------------------------------------
    # 1. Bevel parameter assignment and RNA verification
    res_cyl_a = blender_actions.execute_action({
        "action": "create_cylinder",
        "name": "SQ_CylAngle",
        "vertices": 16,
        "radius": 1.0,
        "height": 2.0
    })
    assert res_cyl_a["status"] == "success"
    cyl_a_obj = bpy.data.objects["SQ_CylAngle"]

    # Add bevel with defaults (limit_method=ANGLE, angle_limit=30.0, harden_normals=True)
    res_bev_def = blender_actions.execute_action({
        "action": "add_bevel",
        "name": "SQ_CylAngle",
        "width": 0.1,
        "segments": 2
    })
    assert res_bev_def["status"] == "success"
    assert res_bev_def["limit_method"] == "ANGLE"
    assert res_bev_def["angle_limit"] == 30.0
    assert res_bev_def["harden_normals"] is True
    mod_a = cyl_a_obj.modifiers[res_bev_def["modifier_name"]]
    assert mod_a.limit_method == "ANGLE"
    assert abs(math.degrees(mod_a.angle_limit) - 30.0) < 1e-4
    assert mod_a.harden_normals is True

    # Add bevel with explicit limit_method="NONE", custom angle_limit, harden_normals=False
    res_cyl_n = blender_actions.execute_action({
        "action": "create_cylinder",
        "name": "SQ_CylNone",
        "vertices": 16,
        "radius": 1.0,
        "height": 2.0
    })
    assert res_cyl_n["status"] == "success"
    cyl_n_obj = bpy.data.objects["SQ_CylNone"]

    res_bev_none = blender_actions.execute_action({
        "action": "add_bevel",
        "name": "SQ_CylNone",
        "width": 0.1,
        "segments": 2,
        "limit_method": "NONE",
        "angle_limit": 45.0,
        "harden_normals": False
    })
    assert res_bev_none["status"] == "success"
    assert res_bev_none["limit_method"] == "NONE"
    assert res_bev_none["angle_limit"] == 45.0
    assert res_bev_none["harden_normals"] is False
    mod_n = cyl_n_obj.modifiers[res_bev_none["modifier_name"]]
    assert mod_n.limit_method == "NONE"
    assert mod_n.harden_normals is False

    # 2. Geometric evidence: Compare evaluated geometry between ANGLE and NONE on 16-seg cylinder
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_mesh_a = cyl_a_obj.evaluated_get(depsgraph).to_mesh()
    eval_mesh_n = cyl_n_obj.evaluated_get(depsgraph).to_mesh()

    # ANGLE mode skips 16 longitudinal body edges (dihedral angle 22.5 deg < 30.0 deg)
    # Only the two circular end rims (90 deg) are beveled: 96 vertices, 176 edges, 82 polygons
    # NONE mode bevels all 16 body edges + 2 rims: 224 vertices, 432 edges, 210 polygons
    assert len(eval_mesh_a.polygons) == 82, f"Expected 82 polygons in ANGLE mode, got {len(eval_mesh_a.polygons)}"
    assert len(eval_mesh_n.polygons) == 210, f"Expected 210 polygons in NONE mode, got {len(eval_mesh_n.polygons)}"
    assert len(eval_mesh_a.edges) == 176, f"Expected 176 edges in ANGLE mode, got {len(eval_mesh_a.edges)}"
    assert len(eval_mesh_n.edges) == 432, f"Expected 432 edges in NONE mode, got {len(eval_mesh_n.edges)}"
    assert len(eval_mesh_a.vertices) == 96, f"Expected 96 vertices in ANGLE mode, got {len(eval_mesh_a.vertices)}"
    assert len(eval_mesh_n.vertices) == 224, f"Expected 224 vertices in NONE mode, got {len(eval_mesh_n.vertices)}"
    cyl_a_obj.to_mesh_clear()
    cyl_n_obj.to_mesh_clear()

    # Verify returned modifier name is directly usable by apply_modifier
    res_apply_a = blender_actions.execute_action({
        "action": "apply_modifier",
        "name": "SQ_CylAngle",
        "modifier_name": res_bev_def["modifier_name"]
    })
    assert res_apply_a["status"] == "success"
    assert len(cyl_a_obj.modifiers) == 0
    assert len(cyl_a_obj.data.polygons) == 82, "Baked mesh must retain evaluated 82 polygons after apply_modifier"

    # 3. Smooth-by-Angle live integration tests
    res_smooth_cyl = blender_actions.execute_action({
        "action": "create_cylinder",
        "name": "SQ_SmoothCyl",
        "vertices": 16,
        "radius": 1.0,
        "height": 2.0
    })
    assert res_smooth_cyl["status"] == "success"
    smooth_cyl_obj = bpy.data.objects["SQ_SmoothCyl"]

    # Call shade_smooth with auto_smooth=True and angle=30.0
    res_ss_angle = blender_actions.execute_action({
        "action": "set_shade_smooth",
        "name": "SQ_SmoothCyl",
        "smooth": True,
        "auto_smooth": True,
        "angle": 30.0
    })
    assert res_ss_angle["status"] == "success"
    assert res_ss_angle["smooth"] is True
    assert res_ss_angle["auto_smooth"] is True
    assert res_ss_angle["angle"] == 30.0

    # Inspect polygons and edges: all polygons smooth; 32 rim edges sharp, 16 body edges smooth
    assert all(p.use_smooth for p in smooth_cyl_obj.data.polygons), "All polygons must have use_smooth=True"
    sharp_edges = [e for e in smooth_cyl_obj.data.edges if e.use_edge_sharp]
    assert len(sharp_edges) == 32, f"Expected 32 sharp rim edges, got {len(sharp_edges)}"
    assert len(smooth_cyl_obj.data.edges) == 48, f"Expected 48 total edges, got {len(smooth_cyl_obj.data.edges)}"

    # Call normal shade_smooth (smooth=True, auto_smooth=False) on standard cube
    res_smooth_cube = blender_actions.execute_action({
        "action": "create_cube",
        "name": "SQ_SmoothNormalCube",
        "size": 1.5
    })
    assert res_smooth_cube["status"] == "success"
    cube_obj = bpy.data.objects["SQ_SmoothNormalCube"]

    res_ss_norm = blender_actions.execute_action({
        "action": "set_shade_smooth",
        "name": "SQ_SmoothNormalCube",
        "smooth": True,
        "auto_smooth": False
    })
    assert res_ss_norm["status"] == "success"
    assert res_ss_norm["smooth"] is True
    assert res_ss_norm["auto_smooth"] is False
    assert all(p.use_smooth for p in cube_obj.data.polygons)
    assert not any(e.use_edge_sharp for e in cube_obj.data.edges), "Normal smooth shading must not mark edges sharp"

    # Call flat shading (smooth=False, auto_smooth=True) -> Must apply shade_flat regardless of auto_smooth
    res_ss_flat = blender_actions.execute_action({
        "action": "set_shade_smooth",
        "name": "SQ_SmoothCyl",
        "smooth": False,
        "auto_smooth": True
    })
    assert res_ss_flat["status"] == "success"
    assert res_ss_flat["smooth"] is False
    assert not any(p.use_smooth for p in smooth_cyl_obj.data.polygons), "Flat shading must clear polygon use_smooth"

    # Context & selection isolation: unselected target and active=None restoration
    res_ref_box = blender_actions.execute_action({"action": "create_cube", "name": "SQ_RefBox", "size": 1.0})
    ref_box_obj = bpy.data.objects["SQ_RefBox"]

    # Select RefBox, set active, leave SQ_SmoothCyl unselected
    smooth_cyl_obj.select_set(False)
    ref_box_obj.select_set(True)
    bpy.context.view_layer.objects.active = ref_box_obj
    assert bpy.context.view_layer.objects.active == ref_box_obj
    assert not smooth_cyl_obj.select_get()

    res_ss_iso = blender_actions.execute_action({
        "action": "set_shade_smooth",
        "name": "SQ_SmoothCyl",
        "smooth": True,
        "auto_smooth": True,
        "angle": 45.0
    })
    assert res_ss_iso["status"] == "success"
    assert not smooth_cyl_obj.select_get(), "SQ_SmoothCyl must remain unselected"
    assert ref_box_obj.select_get(), "SQ_RefBox must remain selected"
    assert bpy.context.view_layer.objects.active == ref_box_obj, "SQ_RefBox must remain active"

    # active=None restoration
    bpy.context.view_layer.objects.active = None
    res_ss_no_active = blender_actions.execute_action({
        "action": "set_shade_smooth",
        "name": "SQ_SmoothCyl",
        "smooth": True,
        "auto_smooth": False
    })
    assert res_ss_no_active["status"] == "success"
    assert bpy.context.view_layer.objects.active is None, "active=None must be restored"

    # Context preservation on failure
    ref_box_obj.select_set(True)
    bpy.context.view_layer.objects.active = ref_box_obj
    res_fail_target = blender_actions.execute_action({
        "action": "set_shade_smooth",
        "name": "NonExistentObject123",
        "smooth": True
    })
    assert res_fail_target["status"] == "error"
    assert ref_box_obj.select_get()
    assert bpy.context.view_layer.objects.active == ref_box_obj

    # Clean up
    blender_actions.execute_action({"action": "clear_scene"})
    print("  [PASSED] Blender live integration, mode preservation & context resilience verified.")


def main():
    print("=" * 60)
    print("BLENDER AI ASSISTANT — STABILITY & SAFETY TEST SUITE")
    print("=" * 60)

    test_action_validation()
    test_invalid_requests()
    test_task_lifecycle_and_timeout_sync()
    test_chunked_execution()
    test_blender_live_integration()

    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
