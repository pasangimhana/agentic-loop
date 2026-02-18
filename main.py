import json
import select
import sys
from datetime import datetime, timezone

import config
from event_queue import AgentEvent, EventQueue
from logger import SessionLogger
from providers import get_provider, ToolResult
from registry import ToolRegistry

SYSTEM_PROMPT = """\
You are a capable AI assistant with the ability to create and use tools.

You have two built-in tools:
- `think` — use this to reason step-by-step before acting. Call it whenever you \
need to plan, analyze, or work through a problem. This is free and has no side effects.
- `create_tool` — use this to build new tools on the fly. Tools persist across sessions.

When you need a capability (web requests, file I/O, math, etc.), first check if \
a tool already exists. If not, use `think` to plan the tool, then `create_tool` to build it.

Guidelines for creating tools:
- The `code` must define an `execute(**kwargs)` function returning a string.
- List any pip dependencies in `dependencies`.
- Keep tools focused — one tool per capability.
- Handle errors gracefully inside tool code.

Available tools:
{tool_list}
"""


def build_system_prompt(registry: ToolRegistry) -> str:
    schemas = registry.get_schemas()
    tool_list = json.dumps(schemas, indent=2)
    return SYSTEM_PROMPT.format(tool_list=tool_list)


def format_event_as_input(event: AgentEvent) -> str:
    """Convert an AgentEvent into a natural-language string for the agent."""
    p = event.payload
    if isinstance(p, dict) and "text" in p:
        return f"[{event.source}/{event.event_type}]: {p['text']}"
    return f"[{event.source}/{event.event_type}]: {json.dumps(p)}"


def run_agent_loop(user_input: str, messages: list, registry: ToolRegistry, provider, logger: SessionLogger):
    logger.update_status("processing")
    messages.append({"role": "user", "content": user_input})
    logger.log_user_message(user_input)

    for iteration in range(config.MAX_ITERATIONS):
        system_prompt = build_system_prompt(registry)
        llm_messages = [{"role": "system", "content": system_prompt}] + messages

        tools = registry.get_schemas()
        logger.log_llm_request(llm_messages, tools)

        try:
            response = provider.chat(llm_messages, tools)
        except Exception as e:
            logger.log_error(str(e), "llm_call")
            print(f"\n[error] LLM call failed: {e}")
            return

        logger.log_llm_response(response.content, response.tool_calls, response.usage)

        if response.content:
            print(f"\n{response.content}")

        # No tool calls — done
        if not response.tool_calls:
            messages.append(provider.build_assistant_message(response))
            return

        messages.append(provider.build_assistant_message(response))

        # Execute each tool call
        tool_results = []
        for tc in response.tool_calls:
            if tc.name == "think":
                print(f"\n[think] {tc.arguments.get('thought', '')[:300]}")
                result = registry.execute(tc.name, **tc.arguments)
            else:
                logger.update_status("tool_executing", current_tool=tc.name)
                print(f"\n[tool] {tc.name}({json.dumps(tc.arguments, indent=2)[:200]}...)")
                result = registry.execute(tc.name, **tc.arguments)
                print(f"[result] {result[:300]}")

            logger.log_tool_exec(tc.name, tc.arguments, result)
            tool_results.append(ToolResult(tool_call_id=tc.id, name=tc.name, result=result))

        logger.update_status("processing")

        result_messages = provider.build_tool_results_message(tool_results)
        if isinstance(result_messages, list):
            messages.extend(result_messages)
        else:
            messages.append(result_messages)

    print("\n[warn] Max iterations reached.")


def check_stdin_ready() -> bool:
    """Non-blocking check if stdin has data ready to read."""
    ready, _, _ = select.select([sys.stdin], [], [], 1.0)
    return bool(ready)


def setup_listeners(event_queue: EventQueue):
    """Discover and start any listener modules the agent has created."""
    import importlib
    import pkgutil
    import listeners as listeners_pkg
    from listeners.base import BaseListener

    for finder, module_name, _ in pkgutil.iter_modules(listeners_pkg.__path__):
        if module_name == "base":
            continue
        try:
            mod = importlib.import_module(f"listeners.{module_name}")
            for attr in dir(mod):
                cls = getattr(mod, attr)
                if (
                    isinstance(cls, type)
                    and issubclass(cls, BaseListener)
                    and cls is not BaseListener
                ):
                    instance = cls()
                    event_queue.register_listener(instance)
                    print(f"[listener] {instance.name} enabled")
        except Exception as e:
            print(f"[warn] Failed to load listener '{module_name}': {e}")

    event_queue.start_listeners()


def main():
    print("Agentic Loop — type 'quit' to exit\n")

    registry = ToolRegistry()
    registry.load_all()
    provider = get_provider(config.LLM_PROVIDER)
    logger = SessionLogger()
    event_queue = EventQueue()

    tool_count = len(registry.get_schemas()) - 2  # minus think + create_tool
    print(f"Provider: {config.LLM_PROVIDER} ({config.get_model()})")
    print(f"Loaded {tool_count} tool(s) from disk")
    print(f"Session log: {logger.log_file}")

    setup_listeners(event_queue)
    print()

    messages = []
    logger.update_status("idle")

    try:
        while True:
            # Check for external events first
            pending = event_queue.get_all_pending()
            if pending:
                for event in pending:
                    formatted = format_event_as_input(event)
                    print(f"\n[event] {formatted}")
                    logger.log_external_event(event.source, event.event_type, event.payload)
                    run_agent_loop(formatted, messages, registry, provider, logger)
                    print()
                logger.update_status("idle")

            # Non-blocking stdin check
            if check_stdin_ready():
                try:
                    user_input = sys.stdin.readline().strip()
                except EOFError:
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    break

                run_agent_loop(user_input, messages, registry, provider, logger)
                logger.update_status("idle")
                print()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        event_queue.stop_listeners()
        logger.update_status("idle")
        logger.close()


if __name__ == "__main__":
    main()
