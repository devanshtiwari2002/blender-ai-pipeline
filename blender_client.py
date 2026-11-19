import socket
import json
import sys
import action_validator

HOST = "127.0.0.1"
PORT = 5001


def send_action(payload):
    is_valid, err, normalized = action_validator.validate_payload(payload)
    if not is_valid:
        print(f"[ERROR] Validation failed: {err}")
        return None

    data = json.dumps({"actions": normalized}).encode("utf-8")
    timeout = max(5.0, len(normalized) * 2.0)

    print(f"[*] Connecting to Blender at {HOST}:{PORT} (sending {len(normalized)} actions)...")
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout) as client:
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
        print(f"[ERROR] Connection refused at {HOST}:{PORT}. Ensure blender_server.py is running in Blender.")
        return None
    except socket.timeout:
        print(f"[ERROR] Connection timed out waiting for Blender.")
        return None
    except Exception as exc:
        print(f"[ERROR] Socket error: {exc}")
        return None

    try:
        response = json.loads(raw_response)
        print("Blender response:\n", json.dumps(response, indent=2))
        return response
    except Exception:
        print("Blender response (raw):", raw_response)
        return raw_response


if __name__ == "__main__":
    test_batch = [
        {"action": "create_cylinder", "name": "BottleBody", "location": [0, 0, 1.75], "radius": 0.75, "depth": 3.5},
        {"action": "set_dimensions", "name": "BottleBody", "dimensions": [1.5, 1.5, 3.5]},
        {"action": "add_bevel", "name": "BottleBody", "width": 0.05, "segments": 3},
        {"action": "set_shade_smooth", "name": "BottleBody", "smooth": True},
        {"action": "get_scene_info"}
    ]

    send_action(test_batch)
