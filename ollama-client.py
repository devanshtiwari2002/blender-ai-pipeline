#!/usr/bin/env python
"""Wrapper to allow executing 'python ollama-client.py' identically to 'ollama_client.py'."""
import sys
from ollama_client import execute_pipeline, DEFAULT_PROMPT

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        if sys.stdin.isatty():
            try:
                user_prompt = input(f"Enter prompt for Blender (Press Enter for default relative positioning test):\n> ").strip()
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
