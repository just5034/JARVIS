"""Minimal SWE-bench agent: tool-call loop against an OpenAI-compatible LLM.

Design goals:
- Self-contained: no SWE-agent / OpenHands dependency
- Transparent: ~400 LOC, the whole loop fits in one read-through
- Compatible with vLLM's OpenAI-compatible chat completions API
- Outputs unified diffs in SWE-bench predictions format

Tools the agent has:
- read_file(path)              — read file contents
- write_file(path, content)    — overwrite file
- str_replace(path, old, new)  — surgical edit
- bash(cmd)                    — run shell command in repo dir (timeout 60s)
- grep(pattern, path)          — search files
- list_dir(path)               — directory listing
- finish()                     — produce final patch and exit

Each turn:
1. Send conversation + tool descriptions to LLM
2. Parse tool call from response (JSON in <tool> block)
3. Execute tool, append result to conversation
4. Repeat until finish() or max_steps

The patch is computed as `git diff` against the base commit at the end.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a software engineer fixing a bug in a Python codebase. You have access to a sandboxed copy of the repository at the base commit, and a set of tools to read, edit, and run code.

Your goal: produce a minimal patch that resolves the issue described below. Focus on the specific bug — do not refactor unrelated code.

You have these tools, each invoked by emitting a JSON block inside <tool>...</tool> tags:

<tool>
{"name": "read_file", "args": {"path": "src/foo.py"}}
</tool>

Available tools:
- read_file(path): Read a file. Returns content with line numbers.
- write_file(path, content): Overwrite a file. Use sparingly; prefer str_replace.
- str_replace(path, old, new): Replace exact string `old` with `new` in file. The `old` string must be unique in the file.
- bash(cmd): Run a shell command in the repo root. Timeout 60s. Use for `python -c`, `pytest`, `find`, etc.
- grep(pattern, path): Search for a regex pattern under path. Returns matching lines.
- list_dir(path): List a directory.
- finish(): Signal that you're done. The current state of the working tree will be diffed against the base commit to produce your patch.

Guidelines:
1. Start by understanding the issue: read the problem statement carefully.
2. Locate the relevant code: use grep or list_dir to find files mentioned in the issue.
3. Read the code that needs to change. Read the surrounding context too.
4. Make a minimal edit. Use str_replace for surgical changes.
5. Verify your fix runs without syntax errors: `bash python -c "import <module>"`.
6. Call finish() when done.

Output format: think briefly, then emit exactly ONE tool call per turn in <tool>...</tool> tags. Do not chain tools in one turn."""


USER_PROMPT_TEMPLATE = """## Issue

{problem_statement}

## Repository

You are at the base commit of the {repo} repository. The working directory is the repo root.

Begin by exploring the codebase. Find the bug, fix it, then call finish()."""


@dataclass
class AgentResult:
    instance_id: str
    patch: str
    n_steps: int
    finished: bool
    error: str | None = None
    history: list[dict] = field(default_factory=list)


def _parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract a tool call from model output. Returns (tool_name, args) or None."""
    # Match <tool>...</tool> block, taking the LAST one if multiple
    matches = re.findall(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not matches:
        return None

    raw = matches[-1].strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        call = json.loads(raw)
    except json.JSONDecodeError:
        return None

    name = call.get("name")
    args = call.get("args", {}) or {}
    if not name:
        return None
    return name, args


def _format_file_with_lines(content: str, max_lines: int = 500) -> str:
    """Add line numbers to file content, truncating very long files."""
    lines = content.splitlines()
    if len(lines) > max_lines:
        head = lines[:max_lines]
        tail_count = len(lines) - max_lines
        return "\n".join(f"{i+1:5d}\t{line}" for i, line in enumerate(head)) + \
               f"\n... ({tail_count} more lines truncated)"
    return "\n".join(f"{i+1:5d}\t{line}" for i, line in enumerate(lines))


class SWEBenchAgent:
    """Tool-using agent that produces patches for SWE-bench instances."""

    def __init__(self, llm_client, model: str, max_steps: int = 25,
                 max_tokens: int = 4096, temperature: float = 0.6):
        self.client = llm_client
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ── Tool implementations ──────────────────────────────────────────

    def _tool_read_file(self, repo_dir: Path, path: str) -> str:
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

    def _tool_write_file(self, repo_dir: Path, path: str, content: str) -> str:
        try:
            full = (repo_dir / path).resolve()
            if not str(full).startswith(str(repo_dir.resolve())):
                return "ERROR: path escapes repo"
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return f"OK: wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def _tool_str_replace(self, repo_dir: Path, path: str, old: str, new: str) -> str:
        try:
            full = (repo_dir / path).resolve()
            if not str(full).startswith(str(repo_dir.resolve())):
                return "ERROR: path escapes repo"
            if not full.exists():
                return f"ERROR: file not found: {path}"
            content = full.read_text(errors="replace")
            count = content.count(old)
            if count == 0:
                return "ERROR: `old` string not found in file"
            if count > 1:
                return f"ERROR: `old` string is not unique ({count} occurrences). Add more context."
            content = content.replace(old, new, 1)
            full.write_text(content)
            return f"OK: replaced 1 occurrence in {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def _tool_bash(self, repo_dir: Path, cmd: str) -> str:
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(repo_dir),
                capture_output=True, text=True, timeout=60,
            )
            stdout = result.stdout[-2000:] if result.stdout else ""
            stderr = result.stderr[-1000:] if result.stderr else ""
            output = f"exit={result.returncode}\n"
            if stdout:
                output += f"--- stdout ---\n{stdout}\n"
            if stderr:
                output += f"--- stderr ---\n{stderr}\n"
            return output.strip()
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out (60s)"
        except Exception as e:
            return f"ERROR: {e}"

    def _tool_grep(self, repo_dir: Path, pattern: str, path: str = ".") -> str:
        try:
            cmd = ["grep", "-rn", "--include=*.py", pattern, path]
            result = subprocess.run(
                cmd, cwd=str(repo_dir),
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout[:3000]
            if not output:
                return "(no matches)"
            return output
        except Exception as e:
            return f"ERROR: {e}"

    def _tool_list_dir(self, repo_dir: Path, path: str = ".") -> str:
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
        """Execute a tool. Returns (output, is_finish)."""
        if name == "finish":
            return "OK: finishing", True
        if name == "read_file":
            return self._tool_read_file(repo_dir, args.get("path", "")), False
        if name == "write_file":
            return self._tool_write_file(repo_dir, args.get("path", ""), args.get("content", "")), False
        if name == "str_replace":
            return self._tool_str_replace(
                repo_dir, args.get("path", ""), args.get("old", ""), args.get("new", "")
            ), False
        if name == "bash":
            return self._tool_bash(repo_dir, args.get("cmd", "")), False
        if name == "grep":
            return self._tool_grep(repo_dir, args.get("pattern", ""), args.get("path", ".")), False
        if name == "list_dir":
            return self._tool_list_dir(repo_dir, args.get("path", ".")), False
        return f"ERROR: unknown tool '{name}'", False

    # ── LLM call ──────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        """One LLM call. Returns the response text."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    # ── Main loop ─────────────────────────────────────────────────────

    def solve(self, instance: dict, repo_dir: Path) -> AgentResult:
        """Run the agent loop on one SWE-bench instance.

        Args:
            instance: dict with at least 'instance_id', 'repo', 'problem_statement'
            repo_dir: path to the repo at base_commit (caller is responsible for setup)

        Returns:
            AgentResult with the final patch (git diff vs base commit).
        """
        instance_id = instance["instance_id"]

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                problem_statement=instance["problem_statement"],
                repo=instance.get("repo", "the repository"),
            )},
        ]

        finished = False
        error = None
        n_steps = 0

        for step in range(self.max_steps):
            n_steps = step + 1
            try:
                response = self._call_llm(messages)
            except Exception as e:
                error = f"LLM call failed at step {step}: {e}"
                logger.error(error)
                break

            messages.append({"role": "assistant", "content": response})

            tool_call = _parse_tool_call(response)
            if tool_call is None:
                # No valid tool call — nudge the model
                messages.append({
                    "role": "user",
                    "content": "I didn't see a valid <tool>...</tool> JSON block in your response. Please emit exactly one tool call.",
                })
                continue

            name, args = tool_call
            output, is_finish = self._execute_tool(repo_dir, name, args)
            logger.info(f"[{instance_id}] step {step}: {name}({list(args.keys())}) -> {len(output)} chars")

            if is_finish:
                finished = True
                break

            messages.append({
                "role": "user",
                "content": f"Tool result:\n{output}",
            })

        # Generate the patch as `git diff` from base
        patch = self._generate_patch(repo_dir)

        return AgentResult(
            instance_id=instance_id,
            patch=patch,
            n_steps=n_steps,
            finished=finished,
            error=error,
            history=messages,
        )

    def _generate_patch(self, repo_dir: Path) -> str:
        """Generate a unified diff of all changes vs the base commit."""
        try:
            # Stage everything (including new files) so `git diff --cached` sees them
            subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), capture_output=True, timeout=30)
            result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=30,
            )
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to generate patch: {e}")
            return ""
