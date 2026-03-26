"""
LLM-based labeling pipeline for C functions.

Uses Claude API to generate structured labels for each function.
Labels include:
- high_level_purpose: One sentence description
- control_flow_elements: Deterministically extracted via AST/regex
- side_effects: Predicted by LLM (io, memory, hardware, none)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .extract import CFunction, extract_control_flow

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LabeledFunction:
    """A function with its semantic labels."""
    function_name: str
    function_code: str
    file_path: str
    labels: Dict

    def to_dict(self) -> dict:
        return asdict(self)


# Label definitions
SIDE_EFFECTS = ["io", "memory", "hardware", "network", "global_state", "none"]
COMPLEXITY_LEVELS = ["low", "medium", "high"]
ERROR_HANDLING_TYPES = ["returns_code", "uses_errno", "assertions", "none"]
CONTROL_FLOW_ELEMENTS = ["if", "for", "while", "switch", "goto", "return"]

LABEL_PROMPT = """Analyze this C function and return ONLY valid JSON with these fields:

1. high_level_purpose: one sentence description of what this function does

2. side_effects: array containing applicable items from ["io", "memory", "hardware", "network", "global_state", "none"]
   - "io": file operations (fopen, fread, fwrite), console I/O (printf, scanf, puts)
   - "memory": heap allocation/deallocation (malloc, free, realloc, calloc, memcpy, memset)
   - "hardware": register access, port I/O, interrupt handling, DMA operations
   - "network": socket operations (socket, send, recv, connect, bind, listen)
   - "global_state": reads or modifies global/static variables
   - "none": pure computation with no side effects (only use if no other applies)

3. complexity: one of ["low", "medium", "high"]
   - "low": simple logic, few branches, < 20 lines, single responsibility
   - "medium": moderate branching, loops, 20-50 lines, some edge cases
   - "high": complex control flow, nested loops, > 50 lines, multiple responsibilities

4. error_handling: one of ["returns_code", "uses_errno", "assertions", "none"]
   - "returns_code": returns error codes (0/-1, NULL, enum values)
   - "uses_errno": uses errno or perror for error reporting
   - "assertions": uses assert() or similar runtime checks
   - "none": no explicit error handling

Function:
```c
{code}
```

Return ONLY the JSON object, no explanation or markdown:"""


class FunctionLabeler:
    """Labels C functions using Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def label_function(self, func: CFunction) -> LabeledFunction:
        """Generate labels for a single function."""
        # Get control flow elements deterministically (no model needed)
        control_flow = extract_control_flow(func.function_code)

        # Use LLM for semantic labels
        llm_labels = self._get_llm_labels(func.function_code)

        labels = {
            "high_level_purpose": llm_labels.get("high_level_purpose", "Unknown purpose"),
            "control_flow_elements": control_flow,
            "side_effects": llm_labels.get("side_effects", ["none"]),
            "complexity": llm_labels.get("complexity", "medium"),
            "error_handling": llm_labels.get("error_handling", "none")
        }

        return LabeledFunction(
            function_name=func.function_name,
            function_code=func.function_code,
            file_path=func.file_path,
            labels=labels
        )

    def _get_llm_labels(self, code: str) -> Dict:
        """Call Claude API to get semantic labels."""
        try:
            # Truncate very long functions to stay within token limits
            truncated_code = code[:4000] if len(code) > 4000 else code

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0,  # Deterministic output
                messages=[{
                    "role": "user",
                    "content": LABEL_PROMPT.format(code=truncated_code)
                }]
            )

            response_text = response.content[0].text.strip()
            # Handle potential markdown code blocks in response
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return {"high_level_purpose": "Unknown", "side_effects": ["none"]}
        except Exception as e:
            print(f"Warning: LLM call failed: {e}")
            return {"high_level_purpose": "Unknown", "side_effects": ["none"]}

    def label_functions(self, functions: List[CFunction], show_progress: bool = True) -> List[LabeledFunction]:
        """Label multiple functions with progress bar."""
        labeled = []
        iterator = tqdm(functions, desc="Labeling functions") if show_progress else functions

        for func in iterator:
            labeled_func = self.label_function(func)
            labeled.append(labeled_func)

        return labeled


class OfflineLabeler:
    """
    Offline labeling without API calls.
    Uses heuristics for side effects, complexity, and error handling.
    """

    SIDE_EFFECT_PATTERNS = {
        "io": [
            r'\bprintf\b', r'\bscanf\b', r'\bfprintf\b', r'\bfscanf\b',
            r'\bfopen\b', r'\bfclose\b', r'\bfread\b', r'\bfwrite\b',
            r'\bputs\b', r'\bgets\b', r'\bfgets\b', r'\bfputs\b',
            r'\bperror\b', r'\bgetchar\b', r'\bputchar\b'
        ],
        "memory": [
            r'\bmalloc\b', r'\bcalloc\b', r'\brealloc\b', r'\bfree\b',
            r'\bmemcpy\b', r'\bmemset\b', r'\bmemmove\b', r'\bstrdup\b',
            r'\bkmalloc\b', r'\bkfree\b', r'\bvmalloc\b', r'\balloca\b'
        ],
        "hardware": [
            r'\binb\b', r'\boutb\b', r'\binw\b', r'\boutw\b',
            r'\breadl\b', r'\bwritel\b', r'\bioread\b', r'\biowrite\b',
            r'\birq\b', r'\binterrupt\b', r'volatile\s+\w+\s*\*',
            r'->regs', r'__iomem', r'\bDMA\b', r'\bPCI\b', r'\bGPIO\b'
        ],
        "network": [
            r'\bsocket\b', r'\bconnect\b', r'\bbind\b', r'\blisten\b',
            r'\baccept\b', r'\bsend\b', r'\brecv\b', r'\bsendto\b',
            r'\brecvfrom\b', r'\bgetaddrinfo\b', r'\bgethostbyname\b',
            r'\binet_\w+\b', r'\bhtons\b', r'\bntohs\b'
        ],
        "global_state": [
            r'\bstatic\s+\w+\s+\w+\s*=', r'\bextern\b',
            r'\bg_\w+\b', r'\bs_\w+\b',  # Common global/static prefixes
        ]
    }

    ERROR_HANDLING_PATTERNS = {
        "returns_code": [
            r'return\s+-1\b', r'return\s+NULL\b', r'return\s+0\b.*error',
            r'return\s+ERROR', r'return\s+FAIL', r'return\s+false\b'
        ],
        "uses_errno": [
            r'\berrno\b', r'\bperror\b', r'\bstrerror\b'
        ],
        "assertions": [
            r'\bassert\b', r'\bASSERT\b', r'\bBUG_ON\b', r'\bWARN_ON\b'
        ]
    }

    def label_function(self, func: CFunction) -> LabeledFunction:
        """Generate labels using heuristics (no API needed)."""
        import re

        code = func.function_code
        control_flow = extract_control_flow(code)

        # Detect side effects via regex patterns
        side_effects = []
        for effect, patterns in self.SIDE_EFFECT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    side_effects.append(effect)
                    break

        if not side_effects:
            side_effects = ["none"]

        # Determine complexity based on code metrics
        complexity = self._estimate_complexity(code, control_flow)

        # Detect error handling pattern
        error_handling = self._detect_error_handling(code)

        # Generate simple purpose from function name
        purpose = self._generate_purpose(func.function_name)

        labels = {
            "high_level_purpose": purpose,
            "control_flow_elements": control_flow,
            "side_effects": list(set(side_effects)),
            "complexity": complexity,
            "error_handling": error_handling
        }

        return LabeledFunction(
            function_name=func.function_name,
            function_code=func.function_code,
            file_path=func.file_path,
            labels=labels
        )

    def _estimate_complexity(self, code: str, control_flow: list) -> str:
        """Estimate function complexity based on code metrics."""
        lines = len(code.split('\n'))
        num_control_flow = len(control_flow)

        # Count nested braces as proxy for nesting depth
        max_depth = 0
        depth = 0
        for char in code:
            if char == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == '}':
                depth -= 1

        # Scoring: lines + control flow elements + nesting
        score = 0
        if lines > 50:
            score += 2
        elif lines > 20:
            score += 1

        if num_control_flow >= 4:
            score += 2
        elif num_control_flow >= 2:
            score += 1

        if max_depth >= 4:
            score += 2
        elif max_depth >= 3:
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"

    def _detect_error_handling(self, code: str) -> str:
        """Detect the primary error handling pattern used."""
        import re

        for pattern_type, patterns in self.ERROR_HANDLING_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return pattern_type

        return "none"

    def _generate_purpose(self, name: str) -> str:
        """Generate a basic purpose from function name."""
        # Convert camelCase and snake_case to words
        import re
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        words = words.replace('_', ' ').lower()

        if 'init' in words:
            return f"Initializes {words.replace('init', '').strip()}"
        elif 'get' in words:
            return f"Retrieves {words.replace('get', '').strip()}"
        elif 'set' in words:
            return f"Sets {words.replace('set', '').strip()}"
        elif 'read' in words:
            return f"Reads {words.replace('read', '').strip()}"
        elif 'write' in words:
            return f"Writes {words.replace('write', '').strip()}"
        elif 'free' in words or 'destroy' in words or 'cleanup' in words:
            return f"Cleans up or frees {words}"
        elif 'create' in words or 'alloc' in words:
            return f"Allocates or creates {words}"
        else:
            return f"Performs {words} operation"

    def label_functions(self, functions: List[CFunction], show_progress: bool = True) -> List[LabeledFunction]:
        """Label multiple functions."""
        labeled = []
        iterator = tqdm(functions, desc="Labeling functions") if show_progress else functions

        for func in iterator:
            labeled.append(self.label_function(func))

        return labeled


def save_labeled_dataset(labeled_functions: List[LabeledFunction], output_path: str):
    """Save labeled functions to a JSON file."""
    data = [lf.to_dict() for lf in labeled_functions]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} labeled functions to {output_path}")


def load_labeled_dataset(input_path: str) -> List[LabeledFunction]:
    """Load labeled functions from a JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    return [LabeledFunction(**item) for item in data]


if __name__ == "__main__":
    import sys
    from .extract import CFunctionExtractor, extract_functions_from_directory

    if len(sys.argv) < 3:
        print("Usage: python -m src.label <c_file_or_dir> <output.json> [--offline]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    use_offline = "--offline" in sys.argv

    # Extract functions
    from pathlib import Path
    path = Path(input_path)
    extractor = CFunctionExtractor()

    if path.is_file():
        functions = extractor.extract_from_file(str(path))
    else:
        functions = extract_functions_from_directory(str(path))

    print(f"Extracted {len(functions)} functions")

    # Label functions
    if use_offline:
        labeler = OfflineLabeler()
    else:
        labeler = FunctionLabeler()

    labeled = labeler.label_functions(functions)
    save_labeled_dataset(labeled, output_path)
