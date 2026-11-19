import sys
import json
import socket
import requests
import action_validator

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
BLENDER_HOST = "127.0.0.1"
BLENDER_PORT = 5001

SYSTEM_PROMPT = """You are a 3D Blender Assistant specializing in precision assembly and spatial reasoning.
Convert the user request into a structured JSON execution plan containing a sequence of Blender actions.

Output format:
{
  "actions": [
    { ...action 1... },
    { ...action 2... }
  ]
}

Supported Actions & Parameters:
1. Object Creation:
   - {"action": "create_cube", "name": str, "location": [x,y,z], "size": float}
   - {"action": "create_cylinder", "name": str, "location": [x,y,z], "radius": float, "depth": float}
   - {"action": "create_sphere", "name": str, "location": [x,y,z], "radius": float}
   - {"action": "create_plane", "name": str, "location": [x,y,z], "size": float}
   - {"action": "create_cone", "name": str, "location": [x,y,z], "radius": float, "depth": float}
   - {"action": "create_torus", "name": str, "location": [x,y,z], "major_radius": float, "minor_radius": float}

2. Modification:
   - {"action": "set_location", "name": str, "location": [x,y,z]}
   - {"action": "set_rotation", "name": str, "rotation": [deg_x, deg_y, deg_z]}
   - {"action": "set_scale", "name": str, "scale": [x,y,z]}
   - {"action": "set_dimensions", "name": str, "dimensions": [x,y,z]}
   - {"action": "rename_object", "name": str, "new_name": str}
   - {"action": "delete_object", "name": str}
   - {"action": "clear_scene"}

3. Geometry & Materials:
   - {"action": "add_bevel", "name": str, "width": float, "segments": int}
   - {"action": "set_shade_smooth", "name": str, "smooth": bool}
   - {"action": "apply_modifier", "name": str, "modifier_name": str}
   - {"action": "set_material", "name": str, "material_name": str, "color": [r,g,b,a], "metallic": float, "roughness": float, "alpha": float}

4. Scene & Lighting:
   - {"action": "create_light", "name": str, "light_type": "POINT"|"SUN"|"SPOT"|"AREA", "location": [x,y,z], "energy": float, "color": [r,g,b]}
   - {"action": "create_camera", "name": str, "location": [x,y,z], "rotation": [deg_x, deg_y, deg_z]}
   - {"action": "set_background_color", "color": [r,g,b,a], "strength": float}
   - {"action": "get_scene_info"}

5. Object-Relative Positioning & Assembly (PREFERRED FOR POSITIONING EXISTING OBJECTS):
   - {"action": "get_object_bounds", "name": str}
     Returns world-space min, max, center, and dimensions.
   - {"action": "place_relative", "name": str, "reference_object": str, "relationship": "above"|"below"|"beside"|"centred_on"|"inside"|"around", "direction": "+X"|"-X"|"+Y"|"-Y" (for beside), "gap": float (default 0.0), "overlap": float (default 0.0), "align_xy": bool (default true for above/below)}
     Positions 'name' relative to 'reference_object' using actual world-space bounding boxes without modifying geometry.
   - {"action": "align_objects", "name": str, "reference_object": str, "axes": ["X", "Y"]|["Z"], "alignment": "center"|"min"|"max"}
   - {"action": "set_relative_scale", "name": str, "reference_object": str, "factor": float, "match_dimension": "width"|"depth"|"height"|"all"}
   - {"action": "verify_assembly", "object_a": str, "object_b": str, "expected_relationship": "above"|"below", "tolerance": float}
     Verifies alignment, contact gap, and bounding-box overlap.

Rules:
- When asked to position, align, attach, or touch existing objects:
  DO NOT create new objects or recreate meshes!
  USE "place_relative" (or "align_objects") followed by "verify_assembly".
- For "Position A above B and touch bottom of A to top of B":
  Use {"action": "place_relative", "name": "A", "reference_object": "B", "relationship": "above", "gap": 0.0, "align_xy": true}.
- For "Position A below B":
  Use relationship="below", gap=0.0, align_xy=true.
- For "Position A beside B":
  Use relationship="beside", direction="+X"|"-X"|"+Y"|"-Y", gap=0.0.
- Only scale an object if the user explicitly asks to resize or scale it. Do not change object scale otherwise.
- Output ONLY valid JSON matching {"actions": [...]}. No explanatory text or markdown blocks.
"""

DEFAULT_PROMPT = "Position White_Dropper directly above Bottle_Neck. Align their X and Y centres. Make the bottom of White_Dropper touch the top of Bottle_Neck."


def prompt_ollama(user_prompt: str, model: str = OLLAMA_MODEL) -> dict:
    """Send natural language prompt to Ollama and get structured JSON action plan."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        raw_output = res_json.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Ollama. Make sure Ollama is running ('ollama serve').")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 60 seconds.")
    except Exception as exc:
        raise RuntimeError(f"Ollama generation failed: {exc}")

    try:
        plan = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama returned invalid JSON: {raw_output}") from exc

    return plan


def send_to_blender(payload: dict, host: str = BLENDER_HOST, port: int = BLENDER_PORT) -> dict:
    """Send validated action payload over socket to Blender and receive execution results."""
    data = json.dumps(payload).encode("utf-8")
    num_actions = len(payload.get("actions", [1]))
    socket_timeout = max(10.0, num_actions * 3.0)

    try:
        with socket.create_connection((host, port), timeout=socket_timeout) as client:
            client.sendall(data)
            client.shutdown(socket.SHUT_WR)

            chunks = []
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)

            raw_response = b"".join(chunks).decode("utf-8")
    except ConnectionRefusedError:
        raise RuntimeError(
            f"Connection refused at {host}:{port}. Ensure blender_server.py is running inside Blender."
        )
    except socket.timeout:
        raise RuntimeError(f"Socket timed out ({socket_timeout}s) waiting for Blender response.")
    except Exception as exc:
        raise RuntimeError(f"Socket error: {exc}")

    if not raw_response:
        raise RuntimeError("Received empty response from Blender server.")

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"status": "raw", "response": raw_response}


def execute_pipeline(prompt_text: str):
    print("=" * 72)
    print(f"[User Prompt] {prompt_text}")
    print("=" * 72)

    # 1. Query Ollama
    print(f"[*] Querying Ollama ({OLLAMA_MODEL})...")
    raw_plan = prompt_ollama(prompt_text)

    # 2. Validate client-side
    print("[*] Validating action plan schema...")
    is_valid, validation_err, normalized_actions = action_validator.validate_payload(raw_plan)
    if not is_valid:
        raise ValueError(f"Action validation failed: {validation_err}")

    print(f"[+] Validation passed: {len(normalized_actions)} action(s) planned.")
    print("-" * 72)
    for idx, act in enumerate(normalized_actions, 1):
        name_str = f"'{act.get('name')}'" if 'name' in act else ""
        ref_str = f"-> ref='{act.get('reference_object')}'" if 'reference_object' in act else ""
        print(f"  {idx}. {act.get('action'):<20} {name_str:<16} {ref_str}")
    print("-" * 72)

    # 3. Transmit to Blender
    print(f"[*] Sending actions to Blender ({BLENDER_HOST}:{BLENDER_PORT})...")
    result = send_to_blender({"actions": normalized_actions})

    # 4. Display Results
    print("=" * 72)
    print("[Blender Execution Results]")
    print("=" * 72)

    if result.get("status") in {"success", "partial_success"}:
        results_list = result.get("results", [])
        success_count = 0
        for r in results_list:
            st = r.get("status")
            act_name = r.get("action")
            msg = r.get("message") or r.get("error")

            if st == "success":
                success_count += 1
                print(f"  [✓ SUCCESS] {act_name:<20} : {msg}")
            else:
                print(f"  [✗ FAILED ] {act_name:<20} : {msg}")

            # Detailed assembly verification output
            if act_name == "verify_assembly":
                print(f"    - Object A: {r.get('object_a')} | Object B: {r.get('object_b')}")
                print(f"    - Relationship verified : {r.get('verified')} ({r.get('message')})")
                print(f"    - XY Centered           : {r.get('aligned_xy')} (offsets: {r.get('center_offsets')})")
                print(f"    - Vertical Gap (A over B): {r.get('vertical_gap', {}).get('a_above_b')} | Touching: {r.get('vertical_gap', {}).get('is_touching')}")
                print(f"    - Bounding-Box Overlap  : {r.get('bounding_box_overlap')}")
                print(f"    - Method                : {r.get('collision_detection_method')}")
                if "bounds_a" in r and "bounds_b" in r:
                    print(f"    - Bounds A min/max      : {r['bounds_a']['min']} to {r['bounds_a']['max']}")
                    print(f"    - Bounds B min/max      : {r['bounds_b']['min']} to {r['bounds_b']['max']}")

            # Detailed bounding box inspection
            if act_name in {"get_object_bounds", "place_relative"} and "bounds" in r:
                b = r["bounds"]
                print(f"    -> Bounds: min={b['min']}, max={b['max']}, center={b['center']}, dims={b['dimensions']}")

            # Scene objects output
            if act_name == "get_scene_info" and "objects" in r:
                print("\n  --- Current Scene Objects ---")
                for obj in r["objects"]:
                    print(f"    - {obj['name']} ({obj['type']}): loc={obj['location']}, dim={obj['dimensions']}, mats={obj['materials']}")
                print("  -----------------------------\n")

        total = result.get("total_actions", len(results_list))
        print("-" * 72)
        print(f"[SUMMARY] {success_count}/{total} actions executed successfully.")
    elif result.get("status") == "timeout":
        state = result.get("state", "unknown")
        print(f"[TIMEOUT WARNING] Blender request timed out (task state: {state}).")
        if state == "running":
            print("  [!] WARNING: The task was already actively running on the Blender main thread when the socket timed out.")
            print("  [!] Do NOT automatically retry the request immediately, as partial execution may have occurred.")
        elif state == "cancelled":
            print("  [i] The task was cancelled in the queue before execution started.")
        print(f"  Details: {result.get('error')}")
    else:
        print(f"[FATAL ERROR] Blender reported: {result.get('error', result)}")

    print("=" * 72)
    return result


def main():
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        if sys.stdin.isatty():
            try:
                user_prompt = input("Enter prompt for Blender (Press Enter for default relative positioning test):\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                user_prompt = ""
        else:
            user_prompt = ""

        if not user_prompt:
            user_prompt = DEFAULT_PROMPT

    try:
        execute_pipeline(user_prompt)
    except Exception as err:
        print(f"\n[ERROR] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
