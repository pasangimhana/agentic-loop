import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import config


THINK_SCHEMA = {
    "name": "think",
    "description": (
        "Use this tool to think through a problem step-by-step before acting. "
        "Write out your reasoning, plan, or analysis. This does not produce any "
        "side effects — it simply lets you reason before deciding what to do next."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your step-by-step reasoning or analysis",
            },
        },
        "required": ["thought"],
    },
}

CREATE_TOOL_SCHEMA = {
    "name": "create_tool",
    "description": (
        "Create a new tool that persists to disk and becomes immediately available. "
        "Use this when you need a capability that doesn't exist yet. "
        "The code must define an `execute(**kwargs)` function that returns a string."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Snake_case name for the tool (e.g. 'web_search')",
            },
            "description": {
                "type": "string",
                "description": "What the tool does — shown to the LLM",
            },
            "parameters": {
                "type": "object",
                "description": "JSON Schema describing the tool's parameters",
            },
            "code": {
                "type": "string",
                "description": (
                    "Python source code for the tool. Must define an `execute(**kwargs)` "
                    "function that returns a string result."
                ),
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of pip packages the tool needs (e.g. ['requests'])",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorizing the tool (e.g. ['io', 'filesystem'])",
            },
        },
        "required": ["tool_name", "description", "parameters", "code"],
    },
}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}  # name -> {"manifest": ..., "module": ...}
        self._ensure_tools_dir()

    def _ensure_tools_dir(self):
        """Make sure tools/ directory exists."""
        config.TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    def load_all(self):
        """Scan the tools directory and load every tool that has a manifest."""
        if not config.TOOLS_DIR.exists():
            return
        for tool_dir in sorted(config.TOOLS_DIR.iterdir()):
            if not tool_dir.is_dir():
                continue
            manifest_path = tool_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                # Ensure __init__.py exists for import
                init_file = tool_dir / "__init__.py"
                if not init_file.exists():
                    continue
                module = importlib.import_module(f"tools.{tool_dir.name}")
                self._tools[manifest["name"]] = {
                    "manifest": manifest,
                    "module": module,
                }
            except Exception:
                print(f"[warn] Failed to load tool '{tool_dir.name}': {traceback.format_exc()}")

    def register(self, name: str, manifest: dict, module):
        self._tools[name] = {"manifest": manifest, "module": module}

    def execute(self, name: str, **kwargs) -> str:
        if name == "think":
            return kwargs.get("thought", "")
        if name == "create_tool":
            return self.create_tool(**kwargs)

        entry = self._tools.get(name)
        if not entry:
            return f"Error: tool '{name}' not found"
        try:
            result = entry["module"].execute(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}\n{traceback.format_exc()}"

    def get_schemas(self) -> list[dict]:
        schemas = [THINK_SCHEMA, CREATE_TOOL_SCHEMA]
        for entry in self._tools.values():
            m = entry["manifest"]
            schema = {
                "name": m["name"],
                "description": m["description"],
                "parameters": m["parameters"],
            }
            schemas.append(schema)
        return schemas

    def has(self, name: str) -> bool:
        return name in self._tools or name in ("create_tool", "think")

    def create_tool(
        self,
        tool_name: str,
        description: str,
        parameters: dict,
        code: str,
        dependencies: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        tool_dir = config.TOOLS_DIR / tool_name
        try:
            # 1. Create directory
            tool_dir.mkdir(parents=True, exist_ok=True)

            # 2. Write manifest
            manifest = {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
            }
            if tags:
                manifest["tags"] = tags
            (tool_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

            # 3. Write code
            (tool_dir / "__init__.py").write_text(code)

            # 4. Install dependencies
            if dependencies:
                req_file = tool_dir / "requirements.txt"
                req_file.write_text("\n".join(dependencies) + "\n")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            # 5. Import and register
            importlib.invalidate_caches()
            mod_name = f"tools.{tool_name}"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            module = importlib.import_module(mod_name)
            self.register(tool_name, manifest, module)

            return f"Tool '{tool_name}' created and registered successfully."
        except Exception as e:
            return f"Failed to create tool '{tool_name}': {e}\n{traceback.format_exc()}"
