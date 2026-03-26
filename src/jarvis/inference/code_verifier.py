"""S* code execution verification — selects best code candidate by test pass rate.

Generates test inputs, executes each candidate in a sandboxed subprocess,
and selects the candidate that passes the most tests. Falls back to
ThinkPRM or self-consistency voting if all candidates fail.

Reference: AlphaCode, CodeT, S* algorithms.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case with input and expected output."""

    input: str
    expected_output: str
    source: str = "generated"  # "generated", "extracted", "mutated"


@dataclass
class ExecutionResult:
    """Result of executing a code candidate against test cases."""

    candidate_idx: int
    code: str
    passed: int = 0
    total: int = 0
    errors: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


class CodeExtractor:
    """Extracts executable code from model output."""

    @staticmethod
    def extract(text: str) -> str | None:
        """Extract the last fenced code block from model output."""
        # Try ```python ... ``` first
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()

        # Try indented code blocks (4+ spaces)
        indented = re.findall(r"(?:^    .+\n?)+", text, re.MULTILINE)
        if indented:
            return indented[-1].strip()

        return None

    @staticmethod
    def is_valid_python(code: str) -> bool:
        """Check if code parses as valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class TestGenerator:
    """Generates test cases for code verification."""

    @staticmethod
    def extract_from_problem(problem_text: str) -> list[TestCase]:
        """Extract test cases from the problem statement.

        Looks for patterns like:
          Input: ...
          Output: ...
        Or:
          >>> func(args)
          result
        """
        tests = []

        # Pattern 1: Input/Output blocks
        io_pattern = re.findall(
            r"(?:Input|Example\s*\d*)\s*[:=]\s*(.+?)\s*(?:Output|Expected)\s*[:=]\s*(.+?)(?:\n\n|\Z)",
            problem_text,
            re.DOTALL | re.IGNORECASE,
        )
        for inp, out in io_pattern:
            tests.append(TestCase(
                input=inp.strip(),
                expected_output=out.strip(),
                source="extracted",
            ))

        # Pattern 2: Doctest-style >>> blocks
        doctest_pattern = re.findall(
            r">>>\s*(.+?)\n(.+?)(?:\n>>>|\n\n|\Z)",
            problem_text,
        )
        for call, result in doctest_pattern:
            tests.append(TestCase(
                input=call.strip(),
                expected_output=result.strip(),
                source="extracted",
            ))

        return tests

    @staticmethod
    def build_test_generation_prompt(problem_text: str, code: str) -> str:
        """Build a prompt asking the LLM to generate test cases."""
        return textwrap.dedent(f"""\
            Given this problem and solution, generate exactly 5 test cases.
            Each test case should be a JSON object with "input" and "expected_output" fields.
            Include edge cases: empty input, single element, large values, boundary conditions.
            Return ONLY a JSON array, no other text.

            Problem:
            {problem_text[:2000]}

            Solution:
            ```python
            {code[:1500]}
            ```

            Test cases (JSON array):""")

    @staticmethod
    def parse_generated_tests(llm_output: str) -> list[TestCase]:
        """Parse LLM-generated test cases from JSON output."""
        tests = []
        # Extract JSON array from response
        json_match = re.search(r"\[.*\]", llm_output, re.DOTALL)
        if not json_match:
            return tests

        try:
            parsed = json.loads(json_match.group(0))
            for tc in parsed:
                if isinstance(tc, dict) and "input" in tc and "expected_output" in tc:
                    tests.append(TestCase(
                        input=str(tc["input"]),
                        expected_output=str(tc["expected_output"]),
                        source="generated",
                    ))
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse generated test cases")

        return tests


class CodeExecutor:
    """Executes Python code in a sandboxed subprocess."""

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 256,
    ) -> None:
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute(self, code: str, stdin: str = "") -> tuple[bool, str, float]:
        """Execute Python code and return (success, output_or_error, runtime_seconds).

        Runs in a subprocess with resource limits. No Docker required —
        subprocess isolation is sufficient for evaluation purposes.
        Docker can be used in production via the sandbox config.
        """
        import time

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = Path(f.name)

        try:
            start = time.monotonic()
            result = subprocess.run(
                [sys.executable, str(tmp_path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - start

            if result.returncode == 0:
                return True, result.stdout, elapsed
            return False, result.stderr[:500], elapsed
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT", float(self.timeout)
        except Exception as e:
            return False, str(e)[:500], 0.0
        finally:
            tmp_path.unlink(missing_ok=True)

    def run_tests(
        self, code: str, tests: list[TestCase]
    ) -> ExecutionResult:
        """Run code against all test cases."""
        result = ExecutionResult(candidate_idx=-1, code=code, total=len(tests))

        for tc in tests:
            # Build a test harness that feeds input and checks output
            harness = self._build_harness(code, tc)
            success, output, runtime = self.execute(harness, stdin=tc.input)
            result.runtime_seconds += runtime

            if success and output.strip() == tc.expected_output.strip():
                result.passed += 1
            elif not success:
                result.errors.append(output[:200])

        return result

    @staticmethod
    def _build_harness(code: str, test: TestCase) -> str:
        """Build a test harness that wraps the code with stdin/stdout testing.

        If the code reads from stdin, we just run it directly.
        If the code defines a function, we try to call it.
        """
        # Check if code already has a main block or reads from stdin
        if "input()" in code or "sys.stdin" in code or "if __name__" in code:
            return code

        # Otherwise, wrap with a simple caller
        # Try to find the main function name
        func_match = re.search(r"def\s+(\w+)\s*\(", code)
        if func_match:
            func_name = func_match.group(1)
            # Skip common non-solution functions
            if func_name not in ("main", "helper", "solve", "__init__"):
                return f"{code}\n\nprint({func_name}({test.input}))"
            elif func_name in ("main", "solve"):
                return f"{code}\n\n{func_name}()"

        # Fallback: just run the code as-is
        return code


class CodeVerifier:
    """S* verification — selects best code candidate by execution results."""

    def __init__(
        self,
        timeout: int = 30,
        max_test_inputs: int = 10,
    ) -> None:
        self.extractor = CodeExtractor()
        self.test_gen = TestGenerator()
        self.executor = CodeExecutor(timeout=timeout)
        self.max_test_inputs = max_test_inputs

    def verify_candidates(
        self,
        candidates: list[str],
        problem_text: str = "",
        external_tests: list[dict[str, str]] | None = None,
    ) -> tuple[str, float, list[ExecutionResult]]:
        """Verify code candidates by execution against test cases.

        Args:
            candidates: List of full model response texts.
            problem_text: The original problem description (for test extraction).
            external_tests: Optional pre-existing test cases from the problem.

        Returns:
            Tuple of (best candidate text, pass rate, all execution results).
        """
        if not candidates:
            raise ValueError("No candidates to verify")

        # Step 1: Extract code from each candidate, filter invalid
        extracted = []
        for i, candidate in enumerate(candidates):
            code = self.extractor.extract(candidate)
            if code and self.extractor.is_valid_python(code):
                extracted.append((i, candidate, code))
            else:
                logger.debug("Candidate %d: no valid code extracted", i)

        if not extracted:
            logger.warning("No valid code in any candidate — returning first candidate")
            return candidates[0], 0.0, []

        # Step 2: Collect test cases
        tests: list[TestCase] = []

        # From external test cases (e.g., benchmark data)
        if external_tests:
            for tc in external_tests[: self.max_test_inputs]:
                tests.append(TestCase(
                    input=str(tc.get("input", "")),
                    expected_output=str(tc.get("output", tc.get("expected_output", ""))),
                    source="external",
                ))

        # From problem statement
        if problem_text:
            extracted_tests = self.test_gen.extract_from_problem(problem_text)
            tests.extend(extracted_tests[: self.max_test_inputs - len(tests)])

        if not tests:
            logger.debug("No test cases available — falling back to first valid candidate")
            return extracted[0][1], 0.0, []

        # Step 3: Execute each candidate against tests
        results: list[ExecutionResult] = []
        for i, candidate_text, code in extracted:
            exec_result = self.executor.run_tests(code, tests)
            exec_result.candidate_idx = i
            results.append(exec_result)
            logger.debug(
                "Candidate %d: %d/%d tests passed (%.1f%%)",
                i, exec_result.passed, exec_result.total,
                exec_result.pass_rate * 100,
            )

        # Step 4: Select best candidate
        # Primary: highest pass rate
        # Tiebreak: shortest code (Occam's razor)
        results.sort(key=lambda r: (-r.pass_rate, len(r.code)))
        best = results[0]

        # Find the full candidate text for the winner
        winner_text = candidates[best.candidate_idx]

        logger.info(
            "S* verification: best candidate %d with %d/%d tests passed (%.1f%%)",
            best.candidate_idx, best.passed, best.total, best.pass_rate * 100,
        )

        return winner_text, best.pass_rate, results
