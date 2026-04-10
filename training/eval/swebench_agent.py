"""SWE-bench agent v2: uses OpenAI function calling API (not XML parsing).

v1 problem: 12/20 instances got empty patches because Qwen3.5 didn't
produce <tool>...</tool> XML blocks reliably. It would output thinking
text without the required format, triggering nudge messages 25 times.

v2 fix: use vLLM's native OpenAI function calling API. The model
outputs structured tool_calls in the response, which vLLM parses
using Qwen3.5's chat template. Much more reliable.

Also adds:
- Conversation trimming: drops old tool results when approaching 24K tokens
- Dynamic max_tokens: adjusts to avoid context overflow
- Better error handling: catches partial tool call parse failures
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Tool definitions (OpenAI function calling format) ─────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository. Returns content with line numbers. For large files, returns first 300 lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from repo root"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace an exact string in a file. The old string must appear exactly once. Include enough context to make it unique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from repo root"},
                    "old": {"type": "string", "description": "Exact string to find (must be unique in file)"},
                    "new": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite a file completely. Prefer str_replace for surgical edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from repo root"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the repo root. Timeout 60s. Use for running tests, finding files, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in Python files under a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in (default: '.')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: '.')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that you are done fixing the bug. The current state of the working tree will be diffed to produce your patch.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a software engineer fixing a bug in a Python codebase. You have tools to read, edit, and run code in a sandboxed copy of the repository.

Your goal: produce a minimal patch that resolves the issue. Focus on the specific bug — do not refactor unrelated code.

Workflow:
1. Understand the issue: read the problem statement carefully.
2. Locate relevant code: use grep or list_dir.
3. Read the code and surrounding context.
4. Make a minimal edit with str_replace.
5. Verify: run a quick test or syntax check with bash.
6. Call finish() when done."""


USER_PROMPT_TEMPLATE = """## Issue

{problem_statement}

## Repository: {repo}

You are at the base commit. The working directory is the repo root. Begin by exploring the codebase to find the bug, then fix it and call finish()."""


@dataclass
class AgentResult:
    instance_id: str
    patch: str
    n_steps: int
    finished: bool
    error: str | None = None


def _format_file_with_lines(content: str, max_lines: int = 300) -> str:
    """Add line numbers, truncate long files."""
    lines = content.splitlines()
    if len(lines) > max_lines:
        head = lines[:max_lines]
        return "\n".join(f"{i+1:5d}\t{line}" for i, line in enumerate(head)) + \
               f"\n... ({len(lines) - max_lines} more lines truncated)"
    return "\n".join(f"{i+1:5d}\t{line}" for i, line in enumerate(lines))


class SWEBenchAgent:
    """Tool-calling agent that produces patches for SWE-bench instances."""

    def __init__(self, llm_client, model: str, max_steps: int = 30,
                 max_tokens: int = 4096, temperature: float = 0.6,
                 context_limit: int = 28000):
        self.client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_limit = context_limit  # start trimming at this token estimate

    # ── Tool implementations ──────────────────────────────────────────

    def _exec_read_file(self, repo_dir: Path, path: str) -> str:
        try:
            full = (repo_dir / path).resolve()
            if not str(full).startswith(str(repo_dir.resolve())):
                return "ERROR: path escapes repo"
            if not full.exists():
                return f"ERROR: file not found: {path}"
            content = full.read_text(errors="replace")
            return _format_file_with_lines(content)
        except Exception as e:
            return f"ERROR: {e}"

    def _exec_str_replace(self, repo_dir: Path, path: str, old: str, new: str) -> str:
        try:
            full = (repo_dir / path).resolve()
            if not str(full).startswith(str(repo_dir.resolve())):
                return "ERROR: path escapes repo"
            if not full.exists():
                return f"ERROR: file not found: {path}"
            content = full.read_text(errors="replace")
            count = content.count(old)
            if count == 0:
                return "ERROR: string not found in file. Make sure you include exact whitespace and context."
            if count > 1:
                return f"ERROR: string appears {count} times. Add more surrounding context to make it unique."
            content = content.replace(old, new, 1)
            full.write_text(content)
            return f"OK: replaced 1 occurrence in {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def _exec_write_file(self, repo_dir: Path, path: str, content: str) -> str:
        try:
            full = (repo_dir / path).resolve()
            if not str(full).startswith(str(repo_dir.resolve())):
                return "ERROR: path escapes repo"
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return f"OK: wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def _exec_bash(self, repo_dir: Path, cmd: str) -> str:
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(repo_dir),
                capture_output=True, text=True, timeout=60,
            )
            out = result.stdout[-2000:] if result.stdout else ""
            err = result.stderr[-1000:] if result.stderr else ""
            parts = [f"exit={result.returncode}"]
            if out:
                parts.append(f"stdout:\n{out}")
            if err:
                parts.append(f"stderr:\n{err}")
            return "\n".join(parts)
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out (60s)"
        except Exception as e:
            return f"ERROR: {e}"

    def _exec_grep(self, repo_dir: Path, pattern: str, path: str = ".") -> str:
        try:
            cmd = ["grep", "-rn", "--include=*.py", "-l", pattern, path]
            result = subprocess.run(
                cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=30,
            )
            files = result.stdout.strip()
            if not files:
                return "(no matches)"
            # Also show matching lines for first few files
            cmd2 = ["grep", "-rn", "--include=*.py", pattern, path]
            result2 = subprocess.run(
                cmd2, cwd=str(repo_dir), capture_output=True, text=True, timeout=30,
            )
            return result2.stdout[:3000]
        except Exception as e:
            return f"ERROR: {e}"

    def _exec_list_dir(self, repo_dir: Path, path: str = ".") -> str:
        try:
            full = (repo_dir / path).resolve()
            if not str(full).startswith(str(repo_dir.resolve())):
                return "ERROR: path escapes repo"
            if not full.is_dir():
                return f"ERROR: not a directory: {path}"
            entries = sorted(full.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            lines = []
            for e in entries[:200]:
                marker = "/" if e.is_dir() else ""
                lines.append(f"{e.name}{marker}")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    def _execute_tool(self, repo_dir: Path, name: str, args: dict) -> tuple[str, bool]:
        """Execute a tool. Returns (result_text, is_finish)."""
        if name == "finish":
            return "Finishing — generating patch from working tree.", True
        dispatch = {
            "read_file": lambda: self._exec_read_file(repo_dir, args.get("path", "")),
            "str_replace": lambda: self._exec_str_replace(repo_dir, args.get("path", ""), args.get("old", ""), args.get("new", "")),
            "write_file": lambda: self._exec_write_file(repo_dir, args.get("path", ""), args.get("content", "")),
            "bash": lambda: self._exec_bash(repo_dir, args.get("cmd", "")),
            "grep": lambda: self._exec_grep(repo_dir, args.get("pattern", ""), args.get("path", ".")),
            "list_dir": lambda: self._exec_list_dir(repo_dir, args.get("path", ".")),
        }
        fn = dispatch.get(name)
        if fn is None:
            return f"ERROR: unknown tool '{name}'", False
        return fn(), False

    # ── Conversation management ───────────────────────────────────────

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        # Tool calls add overhead
        total_chars += sum(len(json.dumps(m.get("tool_calls", []))) for m in messages if "tool_calls" in m)
        return total_chars // 4

    def _trim_conversation(self, messages: list[dict]) -> list[dict]:
        """Drop old tool results when conversation approaches context limit.

        Keeps: system prompt, user prompt (first 2 messages), and recent messages.
        Replaces old tool results with a summary.
        """
        est = self._estimate_tokens(messages)
        if est < self.context_limit:
            return messages

        # Keep system + user + last 10 messages
        head = messages[:2]
        tail = messages[-10:]
        n_dropped = len(messages) - 2 - 10

        if n_dropped <= 0:
            # Can't trim more — just return as is
            return messages

        logger.info(f"  trimming conversation: dropped {n_dropped} middle messages ({est} est tokens -> ~{self._estimate_tokens(head + tail)})")
        summary_msg = {
            "role": "user",
            "content": f"[{n_dropped} earlier messages trimmed to save context. Continue from where you left off.]",
        }
        return head + [summary_msg] + tail

    # ── Main loop ─────────────────────────────────────────────────────

    def solve(self, instance: dict, repo_dir: Path) -> AgentResult:
        """Run the agent loop on one SWE-bench instance."""
        instance_id = instance["instance_id"]

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                problem_statement=instance["problem_statement"],
                repo=instance.get("repo", "unknown"),
            )},
        ]

        finished = False
        error = None
        n_steps = 0

        for step in range(self.max_steps):
            n_steps = step + 1

            # Trim if approaching context limit
            messages = self._trim_conversation(messages)

            # Compute dynamic max_tokens to avoid overflow
            est_input = self._estimate_tokens(messages)
            dynamic_max = min(self.max_tokens, max(1024, 32000 - est_input))

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    temperature=self.temperature,
                    top_p=0.95,
                    max_tokens=dynamic_max,
                )
            except Exception as e:
                error = f"LLM call failed at step {step}: {e}"
                logger.error(f"  {error}")
                break

            choice = resp.choices[0]
            assistant_msg = {"role": "assistant", "content": choice.message.content or ""}

            # Check for tool calls
            if choice.message.tool_calls:
                # Process first tool call (one per turn)
                tc = choice.message.tool_calls[0]
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                assistant_msg["tool_calls"] = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": name, "arguments": tc.function.arguments or "{}"},
                }]
                messages.append(assistant_msg)

                output, is_finish = self._execute_tool(repo_dir, name, args)
                logger.info(f"[{instance_id}] step {step}: {name}({list(args.keys())}) -> {len(output)} chars")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

                if is_finish:
                    finished = True
                    break
            else:
                # No tool call — model just responded with text
                messages.append(assistant_msg)
                # Check if it said "finish" in plain text
                content_lower = (choice.message.content or "").lower()
                if "finish" in content_lower and step > 2:
                    logger.info(f"[{instance_id}] step {step}: model said 'finish' in text, treating as finish")
                    finished = True
                    break
                logger.info(f"[{instance_id}] step {step}: no tool call, text response ({len(choice.message.content or '')} chars)")

        patch = self._generate_patch(repo_dir)

        return AgentResult(
            instance_id=instance_id,
            patch=patch,
            n_steps=n_steps,
            finished=finished,
            error=error,
        )

    def _generate_patch(self, repo_dir: Path) -> str:
        """Generate unified diff of all changes vs base commit."""
        try:
            subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), capture_output=True, timeout=30)
            result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=30,
            )
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to generate patch: {e}")
            return ""
