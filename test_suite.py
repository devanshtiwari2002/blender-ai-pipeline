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
        {"action": "set_dimensions", "name": "Cyl", "dimensions": [1, 1, 2]},
        {"action": "add_bevel", "name": "Cyl", "width": 0.05, "segments": 3},
        {"action": "set_shade_smooth", "name": "Cyl", "smooth": True},
        {"action": "set_material", "name": "Cyl", "color": [0.8, 0.2, 0.1, 1.0], "metallic": 0.5, "roughness": 0.2},
        {"action": "get_scene_info"}
    ]
    ok, err, acts = action_validator.validate_payload(payload)
    assert ok, f"Action list failed: {err}"
    assert len(acts) == 6

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
    assert not ok and "must be positive" in err

    # 7. Exceeding max actions limit
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
