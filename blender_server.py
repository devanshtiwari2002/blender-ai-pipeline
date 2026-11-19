import os
import sys
import bpy
import socket
import json
import threading
import queue
import importlib

# ---------------------------------------------------------------------------
# Robust Project Path Detection & Dependency Setup for Blender Text Editor
# ---------------------------------------------------------------------------

FALLBACK_PROJECT_DIR = r"C:\Users\Devansh Tiwari\Desktop\Blender-ai-assistant"


def resolve_project_directory():
    """Detect the actual project directory, handling Blender Text Editor quirks."""
    candidates = []

    # 1. Prefer actual filepath of the active Blender text datablock if available
    try:
        if hasattr(bpy.context, "space_data") and getattr(bpy.context.space_data, "text", None):
            txt = bpy.context.space_data.text
            if getattr(txt, "filepath", None):
                real_path = bpy.path.abspath(txt.filepath)
                if os.path.isfile(real_path):
                    candidates.append(os.path.dirname(os.path.abspath(real_path)))

        # Also search text datablocks named 'blender_server.py' or 'blender_server'
        for name in ("blender_server.py", "blender_server"):
            txt = bpy.data.texts.get(name)
            if txt and getattr(txt, "filepath", None):
                real_path = bpy.path.abspath(txt.filepath)
                if os.path.isfile(real_path):
                    candidates.append(os.path.dirname(os.path.abspath(real_path)))
    except Exception:
        pass

    # 2. Check __file__ ONLY if it points to a real existing file on disk
    if "__file__" in globals() and isinstance(__file__, str):
        clean_file = __file__.strip()
        if not clean_file.startswith("<") and clean_file not in {"\\Text", "/Text"}:
            try:
                abs_f = os.path.abspath(clean_file)
                if os.path.isfile(abs_f):
                    candidates.append(os.path.dirname(abs_f))
            except Exception:
                pass

    # 3. Add explicit project directory fallback
    candidates.append(FALLBACK_PROJECT_DIR)

    # Find the first candidate directory that contains both dependency files
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            has_val = os.path.isfile(os.path.join(candidate, "action_validator.py"))
            has_act = os.path.isfile(os.path.join(candidate, "blender_actions.py"))
            if has_val and has_act:
                return os.path.abspath(candidate), True, True

    # If neither candidate has both, fallback to default and report status
    chosen = os.path.abspath(FALLBACK_PROJECT_DIR)
    has_val = os.path.isfile(os.path.join(chosen, "action_validator.py"))
    has_act = os.path.isfile(os.path.join(chosen, "blender_actions.py"))
    return chosen, has_val, has_act


# Initialize environment
project_dir, validator_exists, actions_exists = resolve_project_directory()

# Ensure project_dir is at the front of sys.path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
elif sys.path[0] != project_dir:
    sys.path.remove(project_dir)
    sys.path.insert(0, project_dir)

# Invalidate import caches to guarantee fresh discovery from disk
importlib.invalidate_caches()

# Force clean reload: purge cached modules from sys.modules to load newest code from disk
sys.modules.pop("action_validator", None)
sys.modules.pop("blender_actions", None)

try:
    import action_validator
    import blender_actions
except Exception as exc:
    print(f"[Blender AI] Import error: {exc}")
    raise

# Print comprehensive startup diagnostics as requested
print("=" * 72)
print("[Blender AI] Server Environment Initialized")
print(f"[Blender AI] Project Directory : {project_dir}")
print(f"[Blender AI] action_validator  : {action_validator.__file__}")
print(f"[Blender AI] Supported Actions ({len(action_validator.SUPPORTED_ACTIONS)} total):")
print(f"    {sorted(list(action_validator.SUPPORTED_ACTIONS))}")
print(f"[Blender AI] blender_actions   : {blender_actions.__file__}")
print(f"[Blender AI] Dispatched Actions ({len(blender_actions.ACTION_DISPATCH)} total):")
print(f"    {sorted(list(blender_actions.ACTION_DISPATCH.keys()))}")

# Verify all 5 spatial actions are loaded
spatial_actions = ["get_object_bounds", "place_relative", "align_objects", "set_relative_scale", "verify_assembly"]
missing_val = [a for a in spatial_actions if a not in action_validator.SUPPORTED_ACTIONS]
missing_act = [a for a in spatial_actions if a not in blender_actions.ACTION_DISPATCH]

if missing_val or missing_act:
    print(f"[Blender AI] WARNING: Missing spatial actions!")
    if missing_val:
        print(f"  Missing in validator: {missing_val}")
    if missing_act:
        print(f"  Missing in dispatch: {missing_act}")
else:
    print("[Blender AI] All 5 spatial actions verified active in validator and dispatcher.")
print("=" * 72)

# ---------------------------------------------------------------------------
# Server Architecture & Action Execution
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 5001


class BlenderAIServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.running = False
        self.server_sock = None
        self.thread = None
        self.task_queue = queue.Queue()

    def process_queue(self):
        """Called by bpy.app.timers on the Blender main thread."""
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
            except queue.Empty:
                break

            try:
                actions = task["actions"]
                # Always use current module reference
                current_actions_mod = sys.modules.get("blender_actions", blender_actions)
                results = current_actions_mod.execute_action_list(actions)
                task["result"] = results
            except Exception as exc:
                err_msg = f"Fatal execution error on main thread: {str(exc)}"
                print(f"[Blender AI] {err_msg}")
                task["result"] = {"status": "error", "error": err_msg}
            finally:
                task["event"].set()

        return 0.1 if self.running else None

    def socket_listener(self):
        """Socket listener running in a background thread."""
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(5)
            self.server_sock.settimeout(1.0)
            print(f"[Blender AI] Server listening on {self.host}:{self.port}")
        except Exception as exc:
            print(f"[Blender AI] Failed to bind socket on {self.host}:{self.port}: {exc}")
            self.running = False
            return

        while self.running:
            try:
                conn, _ = self.server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                conn.settimeout(10.0)
                chunks = []
                while True:
                    try:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    except (socket.timeout, OSError):
                        break

                raw_data = b"".join(chunks).decode("utf-8").strip()
                if not raw_data:
                    conn.sendall(json.dumps({"status": "error", "error": "Empty payload received"}).encode())
                    continue

                try:
                    payload = json.loads(raw_data)
                except Exception as exc:
                    conn.sendall(json.dumps({"status": "error", "error": f"JSON decode error: {exc}"}).encode())
                    continue

                # Validate with active action_validator module
                current_val_mod = sys.modules.get("action_validator", action_validator)
                is_valid, validation_err, normalized_actions = current_val_mod.validate_payload(payload)
                if not is_valid:
                    conn.sendall(json.dumps({"status": "error", "error": f"Validation failed: {validation_err}"}).encode())
                    continue

                task = {
                    "actions": normalized_actions,
                    "event": threading.Event(),
                    "result": None
                }
                self.task_queue.put(task)

                # Wait for main thread to execute actions (timeout scales with action count)
                timeout_sec = max(5.0, len(normalized_actions) * 2.0)
                completed = task["event"].wait(timeout=timeout_sec)

                if completed and task["result"]:
                    response = task["result"]
                else:
                    response = {
                        "status": "error",
                        "error": f"Action execution timed out after {timeout_sec}s on Blender main thread"
                    }

                try:
                    conn.sendall(json.dumps(response).encode("utf-8"))
                except Exception as exc:
                    print(f"[Blender AI] Failed to send response: {exc}")

    def start(self):
        self.running = True
        bpy.app.timers.register(self.process_queue)
        self.thread = threading.Thread(target=self.socket_listener, daemon=True)
        self.thread.start()
        print("[Blender AI] Listener service initialized")

    def stop(self):
        self.running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        if bpy.app.timers.is_registered(self.process_queue):
            try:
                bpy.app.timers.unregister(self.process_queue)
            except Exception:
                pass
        print("[Blender AI] Previous listener service stopped")


def start_server():
    old_server = bpy.app.driver_namespace.get("BLENDER_AI_SERVER")
    if old_server:
        try:
            old_server.stop()
        except Exception as e:
            print(f"[Blender AI] Error stopping old server: {e}")

    server = BlenderAIServer()
    bpy.app.driver_namespace["BLENDER_AI_SERVER"] = server
    server.start()


start_server()
