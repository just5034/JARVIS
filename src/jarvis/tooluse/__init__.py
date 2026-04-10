"""Tool-use experiment package.

Independent of the main JARVIS serving stack. This subpackage stands up an
OpenAI-compatible chat-completions proxy with structured tool-call support,
backed by a separate vLLM OpenAI server launched with --enable-auto-tool-choice.

Nothing here imports from jarvis.api, jarvis.brains, jarvis.router, or
jarvis.inference. The migration eval path on Delta is unaffected.
"""
