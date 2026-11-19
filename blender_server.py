import os
import sys
import socket
import json
import threading
import queue
import importlib

try:
    import bpy
except ImportError:
    bpy = None

# ---------------------------------------------------------------------------
# Project Path Detection & Dependency Setup for Blender Text Editor
# ---------------------------------------------------------------------------

FALLBACK_PROJECT_DIR = r"C:\Users\Devansh Tiwari\Desktop\Blender-ai-assistant"
MAX_PAYLOAD_SIZE = 5 * 1024 * 1024  # 5 MB safety limit
DEFAULT_CHUNK_SIZE = 15             # Max actions executed per timer tick


def resolve_project_directory():
    """Detect the actual project directory, handling Blender Text Editor quirks."""
    candidates = []

    try:
        if hasattr(bpy.context, "space_data") and getattr(bpy.context.space_data, "text", None):
            txt = bpy.context.space_data.text
            if getattr(txt, "filepath", None):
                real_path = bpy.path.abspath(txt.filepath)
                if os.path.isfile(real_path):
                    candidates.append(os.path.dirname(os.path.abspath(real_path)))

        for name in ("blender_server.py", "blender_server"):
            txt = bpy.data.texts.get(name)
            if txt and getattr(txt, "filepath", None):
                real_path = bpy.path.abspath(txt.filepath)
                if os.path.isfile(real_path):
                    candidates.append(os.path.dirname(os.path.abspath(real_path)))
    except Exception:
        pass

    if "__file__" in globals() and isinstance(__file__, str):
        clean_file = __file__.strip()
        if not clean_file.startswith("<") and clean_file not in {"\\Text", "/Text"}:
            try:
                abs_f = os.path.abspath(clean_file)
                if os.path.isfile(abs_f):
                    candidates.append(os.path.dirname(abs_f))
            except Exception:
                pass

    candidates.append(FALLBACK_PROJECT_DIR)

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            has_val = os.path.isfile(os.path.join(candidate, "action_validator.py"))
            has_act = os.path.isfile(os.path.join(candidate, "blender_actions.py"))
            if has_val and has_act:
                return os.path.abspath(candidate), True, True

    chosen = os.path.abspath(FALLBACK_PROJECT_DIR)
    has_val = os.path.isfile(os.path.join(chosen, "action_validator.py"))
    has_act = os.path.isfile(os.path.join(chosen, "blender_actions.py"))
    return chosen, has_val, has_act


# Initialize environment
project_dir, validator_exists, actions_exists = resolve_project_directory()

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
elif sys.path[0] != project_dir:
    sys.path.remove(project_dir)
    sys.path.insert(0, project_dir)

importlib.invalidate_caches()
sys.modules.pop("action_validator", None)
sys.modules.pop("blender_actions", None)

try:
    import action_validator
    import blender_actions
except Exception as exc:
    print(f"[Blender AI] Startup Error: Failed to import dependencies from {project_dir} - {exc}")
    raise

# ---------------------------------------------------------------------------
# Task Lifecycle States
# ---------------------------------------------------------------------------

TASK_QUEUED = "queued"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_CANCELLED = "cancelled"
TASK_FAILED = "failed"

# ---------------------------------------------------------------------------
# Server Architecture & Action Execution
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 5001


class BlenderAIServer:
    def __init__(self, host=HOST, port=PORT, chunk_size=DEFAULT_CHUNK_SIZE):
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.running = False
        self.server_sock = None
        self.thread = None
        self.task_queue = queue.Queue()
        self.current_task = None

    def process_queue(self):
        """Called by bpy.app.timers on the Blender main thread.
        Executes actions in bounded chunks, yielding control back to Blender's main loop between chunks.
        """
        if not self.running:
            return None

        # 1. Fetch new task if no active task is currently being processed
        if self.current_task is None:
            while not self.task_queue.empty():
                try:
                    candidate = self.task_queue.get_nowait()
                except queue.Empty:
                    break

                with candidate["lock"]:
                    if candidate["state"] == TASK_CANCELLED:
                        continue
                    if candidate["state"] == TASK_QUEUED:
                        candidate["state"] = TASK_RUNNING
                        self.current_task = candidate
                        break

        # 2. If no task to process, return default poll interval
        if self.current_task is None:
            return 0.1 if self.running else None

        task = self.current_task

        # 3. Process next bounded chunk of actions
        actions = task["actions"]
        start_idx = task["current_index"]
        end_idx = min(start_idx + self.chunk_size, len(actions))
        chunk = actions[start_idx:end_idx]

        current_actions_mod = sys.modules.get("blender_actions", blender_actions)
        try:
            chunk_results = []
            has_error = False
            for idx_offset, act in enumerate(chunk):
                actual_idx = start_idx + idx_offset
                res = current_actions_mod.execute_action(act)
                res["index"] = actual_idx
                res["action"] = act.get("action", "unknown")
                chunk_results.append(res)
                if res.get("status") == "error":
                    has_error = True

            task["results_accumulator"].extend(chunk_results)
            if has_error:
                task["has_error"] = True
            task["current_index"] = end_idx
        except Exception as exc:
            err_msg = f"Fatal execution error on main thread: {str(exc)}"
            print(f"[Blender AI] {err_msg}")
            with task["lock"]:
                task["state"] = TASK_FAILED
                task["result"] = {"status": "error", "error": err_msg}
                task["event"].set()
            self.current_task = None
            return 0.1 if self.running else None

        # 4. Check if task completed or has more chunks
        if task["current_index"] >= len(actions):
            total = len(actions)
            has_error = task.get("has_error", False)
            overall_status = "error" if (has_error and total == 1) else ("partial_success" if has_error else "success")
            final_result = {
                "status": overall_status,
                "total_actions": total,
                "results": task["results_accumulator"]
            }
            with task["lock"]:
                task["state"] = TASK_COMPLETED
                task["result"] = final_result
                task["event"].set()
            self.current_task = None
            return 0.1 if self.running else None
        else:
            # Yield to Blender UI loop and schedule continuation on next frame
            return 0.01 if self.running else None

    def socket_listener(self):
        """Socket listener running in a background thread."""
        while self.running:
            try:
                conn, _ = self.server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                with conn:
                    conn.settimeout(10.0)
                    chunks = []
                    total_bytes = 0
                    payload_too_large = False

                    while True:
                        try:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            total_bytes += len(chunk)
                            if total_bytes > MAX_PAYLOAD_SIZE:
                                payload_too_large = True
                                break
                            chunks.append(chunk)

                            # Early break if message framing delimiter detected
                            if chunk.endswith(b"\n"):
                                buf = b"".join(chunks).strip()
                                if (buf.startswith(b"{") and buf.endswith(b"}")) or (buf.startswith(b"[") and buf.endswith(b"]")):
                                    break
                        except (socket.timeout, OSError):
                            break

                    if payload_too_large:
                        conn.sendall(json.dumps({"status": "error", "error": f"Payload exceeds {MAX_PAYLOAD_SIZE // (1024*1024)}MB limit"}).encode())
                        continue

                    raw_data = b"".join(chunks).decode("utf-8").strip()
                    if not raw_data:
                        conn.sendall(json.dumps({"status": "error", "error": "Empty payload received"}).encode())
                        continue

                    try:
                        payload = json.loads(raw_data)
                    except Exception as exc:
                        conn.sendall(json.dumps({"status": "error", "error": f"JSON decode error: {exc}"}).encode())
                        continue

                    current_val_mod = sys.modules.get("action_validator", action_validator)
                    is_valid, validation_err, normalized_actions = current_val_mod.validate_payload(payload)
                    if not is_valid:
                        conn.sendall(json.dumps({"status": "error", "error": f"Validation failed: {validation_err}"}).encode())
                        continue

                    # Create task with synchronized state lifecycle
                    task = {
                        "actions": normalized_actions,
                        "lock": threading.Lock(),
                        "state": TASK_QUEUED,
                        "event": threading.Event(),
                        "result": None,
                        "current_index": 0,
                        "results_accumulator": [],
                        "has_error": False
                    }
                    self.task_queue.put(task)

                    # Dynamic timeout scaling with action count
                    timeout_sec = max(10.0, len(normalized_actions) * 2.0)
                    completed = task["event"].wait(timeout=timeout_sec)

                    with task["lock"]:
                        if completed and task["result"]:
                            response = task["result"]
                        else:
                            if task["state"] == TASK_QUEUED:
                                # Not yet started: mark cancelled so process_queue skips it
                                task["state"] = TASK_CANCELLED
                                response = {
                                    "status": "error",
                                    "error": f"Action execution timed out after {timeout_sec}s while queued in Blender",
                                    "state": TASK_CANCELLED
                                }
                            elif task["state"] == TASK_RUNNING:
                                # Already running on main thread: do NOT claim cancelled
                                response = {
                                    "status": "timeout",
                                    "error": f"Action execution timed out after {timeout_sec}s while running on Blender main thread. The task may still be executing or its outcome is pending.",
                                    "state": TASK_RUNNING
                                }
                            elif task["state"] == TASK_COMPLETED:
                                response = task["result"]
                            else:
                                response = {
                                    "status": "error",
                                    "error": f"Action ended in state '{task['state']}'",
                                    "state": task["state"]
                                }

                    try:
                        conn.sendall(json.dumps(response).encode("utf-8"))
                    except Exception as exc:
                        print(f"[Blender AI] Failed to send response: {exc}")
            except Exception as exc:
                print(f"[Blender AI] Client connection error: {exc}")

    def start(self):
        """Synchronously bind socket before starting thread and registering timer."""
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(5)
            self.server_sock.settimeout(1.0)
        except Exception as exc:
            print(f"[Blender AI] Startup Error: Failed to bind {self.host}:{self.port} - {exc}")
            self.running = False
            if self.server_sock:
                try:
                    self.server_sock.close()
                except Exception:
                    pass
                self.server_sock = None
            return False

        self.running = True
        bpy.app.timers.register(self.process_queue)
        self.thread = threading.Thread(target=self.socket_listener, daemon=True)
        self.thread.start()

        action_count = len(action_validator.SUPPORTED_ACTIONS)
        print(f"[Blender AI] Status: ONLINE | Port: {self.port} | Actions: {action_count} registered")
        print(f"[Blender AI] Modules: action_validator, blender_actions ({project_dir})")
        return True

    def stop(self):
        """Predictable shutdown: close socket, unregister timer, and await thread exit."""
        self.running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None
        if bpy.app.timers.is_registered(self.process_queue):
            try:
                bpy.app.timers.unregister(self.process_queue)
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("[Blender AI] Status: STOPPED")


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


if __name__ == "__main__":
    if bpy is not None:
        start_server()
    else:
        print("[ERROR] blender_server.py must be run inside Blender.")
