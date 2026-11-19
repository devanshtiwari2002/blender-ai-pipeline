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
