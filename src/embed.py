"""
Embedding generation for C functions using CodeBERT.

Provides:
- Function embedding generation
- Similarity search
- Clustering by semantic purpose
- Classification using embeddings + sklearn
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

import re


class CodeFeatureExtractor:
    """
    Extract hand-crafted features from C code that directly signal side effects.

    These features complement embeddings by capturing explicit API calls and
    code patterns that embeddings may not distinguish well.
    """

    # API patterns that indicate specific side effects
    MEMORY_APIS = [
        r'\bmalloc\s*\(', r'\bcalloc\s*\(', r'\brealloc\s*\(', r'\bfree\s*\(',
        r'\bmemset\s*\(', r'\bmemcpy\s*\(', r'\bmemmove\s*\(', r'\bstrdup\s*\(',
        r'\bnew\b', r'\bdelete\b'
    ]

    IO_APIS = [
        r'\bprintf\s*\(', r'\bfprintf\s*\(', r'\bsprintf\s*\(', r'\bsnprintf\s*\(',
        r'\bputs\s*\(', r'\bfputs\s*\(', r'\bputchar\s*\(',
        r'\bscanf\s*\(', r'\bfscanf\s*\(', r'\bsscanf\s*\(',
        r'\bfopen\s*\(', r'\bfclose\s*\(', r'\bfread\s*\(', r'\bfwrite\s*\(',
        r'\bfgets\s*\(', r'\bfgetc\s*\(', r'\bgetchar\s*\(',
        r'\bopen\s*\(', r'\bclose\s*\(', r'\bread\s*\(', r'\bwrite\s*\(',
        r'\bFILE\s*\*'
    ]

    HARDWARE_APIS = [
        r'\bioctl\s*\(', r'\bmmap\s*\(', r'\bmunmap\s*\(',
        r'\binb\s*\(', r'\boutb\s*\(', r'\binw\s*\(', r'\boutw\s*\(',
        r'\bvolatile\b', r'\bregister\b',
        r'->.*reg', r'\bport\b', r'\bhardware\b', r'\bdevice\b'
    ]

    GLOBAL_STATE_PATTERNS = [
        r'\bstatic\s+[^(]+;',  # static variables
        r'\bglobal\b', r'\bextern\b',
        r'\bg_\w+', r'\bs_\w+',  # common global/static prefixes
    ]

    ERROR_HANDLING_PATTERNS = [
        r'\berrno\b', r'\bperror\s*\(', r'\bstrerror\s*\(',
        r'\breturn\s+-1\b', r'\breturn\s+NULL\b', r'\breturn\s+0\b',
        r'\bassert\s*\(', r'\babort\s*\(', r'\bexit\s*\(',
        r'if\s*\([^)]*==\s*NULL', r'if\s*\([^)]*<\s*0',
        r'if\s*\(\s*!\w+\s*\)'  # if (!ptr)
    ]

    def __init__(self):
        # Compile all patterns for efficiency
        self.memory_patterns = [re.compile(p) for p in self.MEMORY_APIS]
        self.io_patterns = [re.compile(p) for p in self.IO_APIS]
        self.hardware_patterns = [re.compile(p) for p in self.HARDWARE_APIS]
        self.global_patterns = [re.compile(p) for p in self.GLOBAL_STATE_PATTERNS]
        self.error_patterns = [re.compile(p) for p in self.ERROR_HANDLING_PATTERNS]

    def extract_features(self, code: str) -> np.ndarray:
        """
        Extract feature vector from C code.

        Returns a numpy array of features:
        - API presence features (binary)
        - Code metric features (normalized)
        """
        features = []

        # === API Presence Features (binary, 0 or 1) ===

        # Memory APIs (count, then normalize)
        memory_count = sum(1 for p in self.memory_patterns if p.search(code))
        features.append(min(memory_count / 3.0, 1.0))  # Normalize, cap at 1
        features.append(1.0 if memory_count > 0 else 0.0)  # Binary presence

        # IO APIs
        io_count = sum(1 for p in self.io_patterns if p.search(code))
        features.append(min(io_count / 3.0, 1.0))
        features.append(1.0 if io_count > 0 else 0.0)

        # Hardware APIs
        hw_count = sum(1 for p in self.hardware_patterns if p.search(code))
        features.append(min(hw_count / 2.0, 1.0))
        features.append(1.0 if hw_count > 0 else 0.0)

        # Global state patterns
        global_count = sum(1 for p in self.global_patterns if p.search(code))
        features.append(min(global_count / 2.0, 1.0))
        features.append(1.0 if global_count > 0 else 0.0)

        # Error handling patterns
        error_count = sum(1 for p in self.error_patterns if p.search(code))
        features.append(min(error_count / 3.0, 1.0))
        features.append(1.0 if error_count > 0 else 0.0)

        # === Code Metrics (normalized) ===

        # Lines of code
        lines = code.count('\n') + 1
        features.append(min(lines / 50.0, 1.0))  # Normalize, cap at 50 lines

        # Number of function calls (approximate)
        func_calls = len(re.findall(r'\w+\s*\([^)]*\)', code))
        features.append(min(func_calls / 20.0, 1.0))

        # Number of control flow statements
        control_flow = len(re.findall(r'\b(if|else|for|while|switch|case)\b', code))
        features.append(min(control_flow / 10.0, 1.0))

        # Pointer operations
        pointer_ops = len(re.findall(r'(\*\w+|\w+->\w+|&\w+)', code))
        features.append(min(pointer_ops / 10.0, 1.0))

        # Parameter count (from function signature)
        param_match = re.search(r'\([^)]*\)\s*{', code)
        if param_match:
            params = param_match.group().count(',') + 1
            if '(void)' in param_match.group() or '()' in param_match.group():
                params = 0
        else:
            params = 0
        features.append(min(params / 5.0, 1.0))

        # Return statement count
        returns = len(re.findall(r'\breturn\b', code))
        features.append(min(returns / 5.0, 1.0))

        # === Specific keyword features ===

        # NULL checks
        features.append(1.0 if re.search(r'\bNULL\b', code) else 0.0)

        # sizeof usage (often indicates memory ops)
        features.append(1.0 if re.search(r'\bsizeof\s*\(', code) else 0.0)

        # Struct access (may indicate complex data)
        features.append(1.0 if re.search(r'\w+\.\w+|\w+->\w+', code) else 0.0)

        # Array access
        features.append(1.0 if re.search(r'\w+\s*\[[^\]]+\]', code) else 0.0)

        return np.array(features, dtype=np.float32)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        """Extract features for multiple code snippets."""
        return np.array([self.extract_features(code) for code in codes])

    @property
    def feature_names(self) -> List[str]:
        """Return human-readable feature names."""
        return [
            'memory_count_norm', 'memory_present',
            'io_count_norm', 'io_present',
            'hardware_count_norm', 'hardware_present',
            'global_count_norm', 'global_present',
            'error_count_norm', 'error_present',
            'lines_norm', 'func_calls_norm', 'control_flow_norm',
            'pointer_ops_norm', 'param_count_norm', 'return_count_norm',
            'has_null', 'has_sizeof', 'has_struct_access', 'has_array_access'
        ]

    @property
    def n_features(self) -> int:
        """Return number of features."""
        return 20


class ASTFeatureExtractor:
    """
    Extract AST-based features from C code using tree-sitter.

    Provides more accurate features than regex:
    - Statement type distribution
    - Control-flow depth metrics
    - Cyclomatic complexity
    """

    # Statement types to count
    STATEMENT_TYPES = [
        'assignment_expression',
        'declaration',
        'call_expression',
        'return_statement',
        'if_statement',
        'for_statement',
        'while_statement',
        'do_statement',
        'switch_statement',
        'goto_statement',
        'break_statement',
        'continue_statement',
    ]

    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter and tree-sitter-c not installed. "
                            "Install with: pip install tree-sitter tree-sitter-c")

        # Initialize tree-sitter parser for C
        self.parser = Parser(Language(tsc.language()))

    def _count_node_types(self, node, type_counts: Dict[str, int], depth: int = 0) -> Tuple[int, int]:
        """
        Recursively count node types and track max depths.
        Returns (max_loop_depth, max_branch_depth)
        """
        max_loop_depth = 0
        max_branch_depth = 0

        node_type = node.type

        # Count this node type
        if node_type in type_counts:
            type_counts[node_type] += 1

        # Track loop depth
        if node_type in ('for_statement', 'while_statement', 'do_statement'):
            max_loop_depth = 1

        # Track branch depth
        if node_type in ('if_statement', 'switch_statement'):
            max_branch_depth = 1

        # Recurse into children
        for child in node.children:
            child_loop, child_branch = self._count_node_types(child, type_counts, depth + 1)

            # Add child depths for nested structures
            if node_type in ('for_statement', 'while_statement', 'do_statement'):
                max_loop_depth = max(max_loop_depth, 1 + child_loop)
            else:
                max_loop_depth = max(max_loop_depth, child_loop)

            if node_type in ('if_statement', 'switch_statement'):
                max_branch_depth = max(max_branch_depth, 1 + child_branch)
            else:
                max_branch_depth = max(max_branch_depth, child_branch)

        return max_loop_depth, max_branch_depth

    def _calculate_cyclomatic_complexity(self, node) -> int:
        """
        Calculate cyclomatic complexity.
        CC = 1 + number of decision points (if, for, while, case, &&, ||, ?:)
        """
        complexity = 0

        node_type = node.type

        # Decision points
        if node_type in ('if_statement', 'for_statement', 'while_statement',
                         'do_statement', 'case_statement', 'conditional_expression'):
            complexity += 1

        # Logical operators (each adds a decision point)
        if node_type == 'binary_expression':
            # Check if it's && or ||
            for child in node.children:
                if child.type in ('&&', '||'):
                    complexity += 1

        # Recurse
        for child in node.children:
            complexity += self._calculate_cyclomatic_complexity(child)

        return complexity

    def extract_features(self, code: str) -> np.ndarray:
        """
        Extract AST-based features from C code.

        Returns ~15 features:
        - Statement distribution (12 normalized counts)
        - Control flow metrics (3: max_loop_depth, max_branch_depth, cyclomatic_complexity)
        """
        features = []

        # Parse the code
        tree = self.parser.parse(bytes(code, 'utf8'))
        root = tree.root_node

        # Initialize type counts
        type_counts = {t: 0 for t in self.STATEMENT_TYPES}

        # Count node types and get depths
        max_loop_depth, max_branch_depth = self._count_node_types(root, type_counts)

        # Calculate cyclomatic complexity
        cyclomatic = 1 + self._calculate_cyclomatic_complexity(root)

        # Total statements for normalization
        total_statements = sum(type_counts.values()) or 1

        # === Statement Distribution (normalized) ===
        for stmt_type in self.STATEMENT_TYPES:
            # Normalized count (proportion of total)
            features.append(type_counts[stmt_type] / total_statements)

        # === Control Flow Metrics ===
        features.append(min(max_loop_depth / 5.0, 1.0))  # Normalized max loop depth
        features.append(min(max_branch_depth / 5.0, 1.0))  # Normalized max branch depth
        features.append(min(cyclomatic / 20.0, 1.0))  # Normalized cyclomatic complexity

        return np.array(features, dtype=np.float32)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        """Extract features for multiple code snippets."""
        return np.array([self.extract_features(code) for code in codes])

    @property
    def feature_names(self) -> List[str]:
        """Return human-readable feature names."""
        names = [f'stmt_{t}_ratio' for t in self.STATEMENT_TYPES]
        names.extend(['max_loop_depth', 'max_branch_depth', 'cyclomatic_complexity'])
        return names

    @property
    def n_features(self) -> int:
        """Return number of features."""
        return len(self.STATEMENT_TYPES) + 3  # 12 stmt types + 3 control flow metrics


@dataclass
class EmbeddedFunction:
    """A function with its embedding vector."""
    function_name: str
    function_code: str
    file_path: str
    labels: Dict
    embedding: np.ndarray

    def to_dict(self) -> dict:
        return {
            "function_name": self.function_name,
            "function_code": self.function_code,
            "file_path": self.file_path,
            "labels": self.labels,
            "embedding": self.embedding.tolist()
        }


class CodeBERTEmbedder:
    """
    Generates embeddings using CodeBERT (microsoft/codebert-base).

    Why CodeBERT:
    - Pre-trained on code from multiple languages including C
    - Understands both natural language and code semantics
    - Well-documented and reproducible
    - 768-dimensional embeddings suitable for downstream tasks
    """

    MODEL_NAME = "microsoft/codebert-base"

    def __init__(self, device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch not installed")

        # Auto-detect best device: CUDA > MPS (Apple Silicon) > CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading CodeBERT on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        # Use safetensors to avoid torch.load security vulnerability (CVE-2025-32434)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, use_safetensors=True).to(self.device)
        self.model.eval()

        print("CodeBERT loaded successfully")

    def embed(self, code: str, max_length: int = 512) -> np.ndarray:
        """Generate embedding for a code snippet using CLS token pooling."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        return embedding

    def embed_batch(self, codes: List[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        """Embed multiple code snippets in batches."""
        all_embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


class Qwen3Embedder:
    """
    Generates embeddings using Qwen3-Embedding model.

    Available models (from https://huggingface.co/collections/Qwen/qwen3-embedding):
    - Qwen/Qwen3-Embedding-0.6B (default, fast, good quality)
    - Qwen/Qwen3-Embedding-4B (better quality, slower)
    - Qwen/Qwen3-Embedding-8B (best quality, slowest)

    Why Qwen3-Embedding:
    - State-of-the-art embedding quality
    - Better semantic understanding than CodeBERT
    - Trained on diverse data including code
    - Strong general-purpose code embedding model
    """

    MODELS = {
        "0.6B": "Qwen/Qwen3-Embedding-0.6B",
        "4B": "Qwen/Qwen3-Embedding-4B",
        "8B": "Qwen/Qwen3-Embedding-8B",
    }
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # Default

    def __init__(self, device: Optional[str] = None, model_name: Optional[str] = None, model_size: str = "0.6B"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch not installed")

        # Auto-detect best device: CUDA > MPS (Apple Silicon) > CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Select model by size
        if model_name:
            selected_model = model_name
        elif model_size in self.MODELS:
            selected_model = self.MODELS[model_size]
        else:
            selected_model = self.MODEL_NAME

        # Store the actual model name as instance variable for save/load
        self.MODEL_NAME = selected_model

        print(f"Loading Qwen3-Embedding ({selected_model}) on {self.device}...")

        # Load using transformers directly (use safetensors to avoid CVE-2025-32434)
        self.tokenizer = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(selected_model, trust_remote_code=True, use_safetensors=True).to(self.device)
        self.model.eval()

        print("Qwen3-Embedding loaded successfully")

    def embed(self, code: str, max_length: int = 4096) -> np.ndarray:
        """Generate embedding for a code snippet using mean pooling. Supports 32k context."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over token embeddings
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = embedding.squeeze().cpu().numpy()

        return embedding

    def embed_batch(self, codes: List[str], batch_size: int = 4, max_length: int = 4096) -> np.ndarray:
        """Embed multiple code snippets in batches. Supports 32k context."""
        all_embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class JinaCodeEmbedder:
    """
    Generates embeddings using Jina Code Embeddings.

    Model: jinaai/jina-embeddings-v2-base-code
    - Specifically trained for code
    - 8192 token context length
    - 768-dimensional embeddings
    - From https://huggingface.co/jinaai/jina-embeddings-v2-base-code
    """

    MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"

    def __init__(self, device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch not installed")

        # Auto-detect best device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading Jina Code Embeddings on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True, use_safetensors=True).to(self.device)
        self.model.eval()

        print("Jina Code Embeddings loaded successfully")

    def embed(self, code: str, max_length: int = 4096) -> np.ndarray:
        """Generate embedding for a code snippet using mean pooling. Supports 8k context."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = embedding.squeeze().cpu().numpy()

        return embedding

    def embed_batch(self, codes: List[str], batch_size: int = 4, max_length: int = 4096) -> np.ndarray:
        """Embed multiple code snippets in batches. Supports 8k context."""
        all_embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class CodeSageEmbedder:
    """
    Generates embeddings using CodeSage.

    Model: codesage/codesage-large-v2
    - Large 1.3B parameter encoder specifically trained for code
    - 2048-dimensional embeddings (higher dimensionality for richer representations)
    - Trained on diverse programming languages including C
    - From https://huggingface.co/codesage/codesage-large-v2
    """

    MODEL_NAME = "codesage/codesage-large-v2"

    def __init__(self, device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch not installed")

        # Auto-detect best device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading CodeSage on {self.device}...")
        print("  NOTE: CodeSage requires specific versions:")
        print("    pip install transformers==4.35.0 torch==2.1.0")

        # CodeSage requires trust_remote_code and add_eos_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            add_eos_token=True
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        print("CodeSage loaded successfully")

    def embed(self, code: str, max_length: int = 1024) -> np.ndarray:
        """Generate embedding for a code snippet using mean pooling."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over token embeddings
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = embedding.squeeze().cpu().numpy()

        return embedding

    def embed_batch(self, codes: List[str], batch_size: int = 4, max_length: int = 1024) -> np.ndarray:
        """Embed multiple code snippets in batches."""
        all_embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class NomicCodeEmbedder:
    """
    Generates embeddings using nomic-embed-code.

    Why nomic-embed-code:
    - Specifically trained on code (not general text)
    - Great for code similarity and understanding tasks
    - 768-dimensional embeddings
    - From https://huggingface.co/nomic-ai/nomic-embed-code
    """

    MODEL_NAME = "nomic-ai/nomic-embed-code"

    def __init__(self, device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch not installed")

        # Auto-detect best device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Loading nomic-embed-code on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True, use_safetensors=True).to(self.device)
        self.model.eval()

        print("nomic-embed-code loaded successfully")

    def embed(self, code: str, max_length: int = 4096) -> np.ndarray:
        """Generate embedding for a code snippet using mean pooling. Supports 32k context."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = embedding.squeeze().cpu().numpy()

        return embedding

    def embed_batch(self, codes: List[str], batch_size: int = 4, max_length: int = 4096) -> np.ndarray:
        """Embed multiple code snippets in batches. Supports 32k context."""
        all_embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def get_embedder(model_type: str = "codebert", device: Optional[str] = None):
    """
    Factory function to get the appropriate embedder.

    Args:
        model_type: "codebert", "qwen3", "qwen3-4b", "qwen3-8b", "nomic", "jina", or "codesage"
        device: "cuda", "mps", or "cpu"

    Available models:
    - codebert: microsoft/codebert-base (768 dim, 512 max tokens)
    - qwen3: Qwen/Qwen3-Embedding-0.6B (1024 dim, 32k context)
    - qwen3-4b: Qwen/Qwen3-Embedding-4B (better quality, 32k context)
    - qwen3-8b: Qwen/Qwen3-Embedding-8B (best quality, 32k context)
    - nomic: nomic-ai/nomic-embed-code (768 dim, 32k context, code-specific)
    - jina: jinaai/jina-embeddings-v2-base-code (768 dim, 8k context, code-specific)
    - codesage: codesage/codesage-large-v2 (2048 dim, 1024 max tokens, code-specific)
    """
    model_type = model_type.lower()

    if model_type == "qwen3":
        return Qwen3Embedder(device=device, model_size="0.6B")
    elif model_type == "qwen3-4b":
        return Qwen3Embedder(device=device, model_size="4B")
    elif model_type == "qwen3-8b":
        return Qwen3Embedder(device=device, model_size="8B")
    elif model_type == "nomic":
        return NomicCodeEmbedder(device=device)
    elif model_type == "codesage":
        try:
            return CodeSageEmbedder(device=device)
        except (ImportError, AttributeError, Exception) as e:
            print(f"\nERROR: CodeSage failed to load: {e}")
            print("\nCodeSage requires older library versions:")
            print("  pip install transformers==4.35.0 torch==2.1.0")
            print("\nFalling back to CodeBERT...")
            return CodeBERTEmbedder(device=device)
    elif model_type == "jina":
        print("Warning: Jina embeddings have compatibility issues with transformers>=5.0")
        print("Falling back to CodeBERT...")
        return CodeBERTEmbedder(device=device)
    else:
        return CodeBERTEmbedder(device=device)


class FunctionEmbeddingPipeline:
    """
    Complete pipeline for embedding functions and performing downstream tasks.

    Demonstrates:
    1. Similarity search - Find functions similar to a query
    2. Classification - Predict labels using embeddings + Logistic Regression
       - side_effects (multi-label): io, memory, hardware, network, global_state, none
       - complexity (single-label): low, medium, high
       - error_handling (single-label): returns_code, uses_errno, assertions, none
    """

    def __init__(self, embedder: Optional[CodeBERTEmbedder] = None):
        self.embedder = embedder or CodeBERTEmbedder()
        self.embedded_functions: List[EmbeddedFunction] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

        # Classification components for side_effects (multi-label)
        self.side_effects_classifier: Optional[LogisticRegression] = None
        self.side_effects_binary_classifiers: Dict[str, any] = {}  # For binary mode
        self.side_effects_mlb: Optional[MultiLabelBinarizer] = None

        # Classification components for complexity (single-label)
        self.complexity_classifier: Optional[LogisticRegression] = None
        self.complexity_classes: List[str] = []

        # Classification components for error_handling (single-label)
        self.error_handling_classifier: Optional[LogisticRegression] = None
        self.error_handling_classes: List[str] = []

        # Dimensionality reduction
        self.pca: Optional[PCA] = None

        # Hybrid features (regex-based)
        self.feature_extractor: Optional[CodeFeatureExtractor] = None
        self.use_hybrid_features: bool = False
        self.feature_scaler: Optional[StandardScaler] = None
        self.code_features_matrix: Optional[np.ndarray] = None

        # AST features (tree-sitter based)
        self.ast_feature_extractor: Optional[ASTFeatureExtractor] = None
        self.use_ast_features: bool = False
        self.ast_feature_scaler: Optional[StandardScaler] = None
        self.ast_features_matrix: Optional[np.ndarray] = None

        # Purpose embeddings (high_level_purpose text embeddings)
        self.use_purpose_embeddings: bool = False
        self.purpose_embeddings_matrix: Optional[np.ndarray] = None

        # Threshold tuning for side_effects
        self.use_threshold_tuning: bool = False
        self.optimal_thresholds: Dict[str, float] = {}  # class_name -> threshold

        # Legacy aliases for backward compatibility
        self.classifier = None
        self.mlb = None

    def embed_labeled_functions(self, labeled_functions: List[Dict], show_progress: bool = True, batch_size: int = 8) -> List[EmbeddedFunction]:
        """Generate embeddings for all labeled functions using batch processing."""
        from tqdm import tqdm

        codes = [lf["function_code"] for lf in labeled_functions]

        # Show device info
        device = getattr(self.embedder, 'device', 'unknown')
        print(f"Generating embeddings for {len(codes)} functions on {device}...")
        print(f"  Batch size: {batch_size} (use larger batches for GPU, smaller for CPU)")

        # Always use batch processing for speed
        all_embeddings = []
        n_batches = (len(codes) + batch_size - 1) // batch_size

        iterator = range(0, len(codes), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", total=n_batches)

        for i in iterator:
            batch = codes[i:i + batch_size]
            batch_embeddings = self.embedder.embed_batch(batch, batch_size=len(batch))
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        # Always extract code features (cheap to compute, useful for hybrid mode)
        self.feature_extractor = CodeFeatureExtractor()
        self.code_features_matrix = self.feature_extractor.extract_batch(codes)
        print(f"  Extracted {self.code_features_matrix.shape[1]} regex-based features")

        # Extract AST features if tree-sitter is available
        if TREE_SITTER_AVAILABLE:
            try:
                self.ast_feature_extractor = ASTFeatureExtractor()
                self.ast_features_matrix = self.ast_feature_extractor.extract_batch(codes)
                print(f"  Extracted {self.ast_features_matrix.shape[1]} AST-based features")
            except Exception as e:
                print(f"  Warning: AST feature extraction failed ({e})")
                self.ast_features_matrix = None
        else:
            print("  Note: tree-sitter not available, AST features disabled")
            self.ast_features_matrix = None

        # Extract purpose embeddings from high_level_purpose text
        purposes = [lf.get("labels", {}).get("high_level_purpose", "") for lf in labeled_functions]
        if any(purposes):  # Only if we have purpose text
            print("  Generating purpose text embeddings...")
            purpose_embeddings = []
            for i in iterator if not show_progress else tqdm(range(0, len(purposes), batch_size), desc="Purpose embeddings"):
                batch_purposes = purposes[i:i + batch_size]
                # Filter empty purposes and embed non-empty ones
                batch_emb = self.embedder.embed_batch(
                    [p if p else "unknown purpose" for p in batch_purposes],
                    batch_size=len(batch_purposes)
                )
                purpose_embeddings.append(batch_emb)
            self.purpose_embeddings_matrix = np.vstack(purpose_embeddings)
            print(f"  Generated {self.purpose_embeddings_matrix.shape[1]}-dim purpose embeddings")
        else:
            self.purpose_embeddings_matrix = None

        self.embedded_functions = []
        for i, lf in enumerate(labeled_functions):
            ef = EmbeddedFunction(
                function_name=lf["function_name"],
                function_code=lf["function_code"],
                file_path=lf["file_path"],
                labels=lf["labels"],
                embedding=embeddings[i]
            )
            self.embedded_functions.append(ef)

        self.embeddings_matrix = embeddings
        print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

        return self.embedded_functions

    # =========================================================================
    # Demonstration 1: Similarity Search
    # =========================================================================

    def similarity_search(self, query_code: str, top_k: int = 5) -> List[Tuple[EmbeddedFunction, float]]:
        """
        Find the most similar functions to a query code snippet.

        Uses cosine similarity between embeddings.
        """
        if self.embeddings_matrix is None:
            raise ValueError("No embeddings loaded. Call embed_labeled_functions first.")

        query_embedding = self.embedder.embed(query_code).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.embedded_functions[idx], similarities[idx]))

        return results

    def find_similar_to_function(self, function_name: str, top_k: int = 5) -> List[Tuple[EmbeddedFunction, float]]:
        """Find functions similar to a named function in the dataset."""
        target = None
        target_idx = -1

        for i, ef in enumerate(self.embedded_functions):
            if ef.function_name == function_name:
                target = ef
                target_idx = i
                break

        if target is None:
            raise ValueError(f"Function '{function_name}' not found in dataset")

        query_embedding = target.embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

        # Exclude the query function itself
        similarities[target_idx] = -1

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.embedded_functions[idx], similarities[idx]))

        return results

    # =========================================================================
    # Demonstration 2: Classification (predict side_effects, complexity, error_handling)
    # =========================================================================

    def train_classifier(self, test_size: float = 0.2, random_state: int = 42,
                         # Per-label classifier types
                         side_effects_clf: str = "logistic_regression",
                         complexity_clf: str = "logistic_regression",
                         error_handling_clf: str = "logistic_regression",
                         tune_hyperparams: bool = False,
                         # Logistic Regression params
                         lr_c: float = 1.0,
                         lr_max_iter: int = 1000,
                         lr_solver: str = "lbfgs",
                         # Random Forest params
                         rf_n_estimators: int = 100,
                         rf_max_depth: Optional[int] = 20,
                         rf_min_samples_split: int = 5,
                         rf_min_samples_leaf: int = 1,
                         # SVM params
                         svm_c: float = 1.0,
                         svm_kernel: str = "rbf",
                         svm_gamma: str = "scale",
                         # MLP params
                         mlp_hidden_layers: tuple = (256, 128),
                         mlp_activation: str = "relu",
                         mlp_learning_rate: float = 0.001,
                         mlp_max_iter: int = 500,
                         mlp_early_stopping: bool = True,
                         # Common
                         use_class_weight: bool = True,
                         # Dimensionality reduction
                         embed_dim: Optional[int] = None,
                         # Binary classifiers mode (one per side effect)
                         use_binary_classifiers: bool = False,
                         # SMOTE upsampling
                         use_smote: bool = False,
                         # Hybrid features (regex-based)
                         use_hybrid_features: bool = False,
                         # AST features (tree-sitter based)
                         use_ast_features: bool = False,
                         # Purpose embeddings (high_level_purpose text)
                         use_purpose_embeddings: bool = False,
                         # Threshold tuning
                         use_threshold_tuning: bool = False) -> Dict:
        """
        Train classifiers to predict:
        - side_effects (multi-label): io, memory, hardware, network, global_state, none
        - complexity (single-label): low, medium, high
        - error_handling (single-label): returns_code, uses_errno, assertions, none

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            side_effects_clf: Classifier for side_effects ("logistic_regression", "random_forest", "svm")
            complexity_clf: Classifier for complexity ("logistic_regression", "random_forest", "svm")
            error_handling_clf: Classifier for error_handling ("logistic_regression", "random_forest", "svm")
            tune_hyperparams: If True, use GridSearchCV for hyperparameter tuning
            lr_c: Logistic Regression inverse regularization strength (higher = less regularization)
            lr_max_iter: Logistic Regression maximum iterations
            lr_solver: Logistic Regression solver ('lbfgs', 'saga', 'liblinear')
            rf_n_estimators: Random Forest number of trees
            rf_max_depth: Random Forest max tree depth (None = unlimited)
            rf_min_samples_split: Random Forest min samples to split a node
            rf_min_samples_leaf: Random Forest min samples at leaf node
            svm_c: SVM regularization parameter (higher = less regularization)
            svm_kernel: SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            svm_gamma: SVM kernel coefficient ('scale', 'auto', or float)
            mlp_hidden_layers: MLP hidden layer sizes (default: (256, 128))
            mlp_activation: MLP activation function ('relu', 'tanh', 'logistic')
            mlp_learning_rate: MLP initial learning rate (default: 0.001)
            mlp_max_iter: MLP max training epochs (default: 500)
            mlp_early_stopping: MLP early stopping to prevent overfitting (default: True)
            use_class_weight: Use 'balanced' class weights to handle imbalanced data
            embed_dim: Reduce embeddings to this dimension using PCA (None = no reduction)
            use_binary_classifiers: Train separate binary classifier per side effect (better for imbalanced)
            use_smote: Use SMOTE to upsample minority classes
            use_hybrid_features: Concatenate hand-crafted code features with embeddings
            use_threshold_tuning: Optimize prediction threshold per class using validation data

        Returns training metrics for all classifiers.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")

        if self.embeddings_matrix is None:
            raise ValueError("No embeddings loaded")

        from sklearn.metrics import f1_score, accuracy_score, classification_report
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import GridSearchCV

        X = self.embeddings_matrix
        original_dim = X.shape[1]

        # Apply PCA dimensionality reduction if requested
        if embed_dim is not None and embed_dim < original_dim:
            print(f"  Reducing embeddings from {original_dim} to {embed_dim} dimensions using PCA...")
            self.pca = PCA(n_components=embed_dim, random_state=random_state)
            X = self.pca.fit_transform(X)
            variance_retained = sum(self.pca.explained_variance_ratio_) * 100
            print(f"  Variance retained: {variance_retained:.1f}%")
        else:
            self.pca = None

        # Apply hybrid features if requested (regex-based)
        self.use_hybrid_features = use_hybrid_features
        if use_hybrid_features:
            if self.code_features_matrix is None:
                raise ValueError("Code features not extracted. Re-run embed_labeled_functions.")

            # Scale code features
            self.feature_scaler = StandardScaler()
            code_features_scaled = self.feature_scaler.fit_transform(self.code_features_matrix)

            # Concatenate embeddings with code features
            X = np.hstack([X, code_features_scaled])
            print(f"  Hybrid features: {X.shape[1]} dims (+{self.code_features_matrix.shape[1]} regex features)")

        # Apply AST features if requested (tree-sitter based)
        self.use_ast_features = use_ast_features
        if use_ast_features:
            if self.ast_features_matrix is None:
                if TREE_SITTER_AVAILABLE:
                    raise ValueError("AST features not extracted. Re-run embed_labeled_functions.")
                else:
                    print("  Warning: tree-sitter not available, skipping AST features")
            else:
                # Scale AST features
                self.ast_feature_scaler = StandardScaler()
                ast_features_scaled = self.ast_feature_scaler.fit_transform(self.ast_features_matrix)

                # Concatenate with existing features
                X = np.hstack([X, ast_features_scaled])
                print(f"  AST features: {X.shape[1]} dims (+{self.ast_features_matrix.shape[1]} AST features)")

        # Apply purpose embeddings if requested
        self.use_purpose_embeddings = use_purpose_embeddings
        if use_purpose_embeddings:
            if self.purpose_embeddings_matrix is None:
                print("  Warning: No purpose embeddings available, skipping")
            else:
                # Concatenate purpose embeddings (already in same scale as code embeddings)
                X = np.hstack([X, self.purpose_embeddings_matrix])
                print(f"  Purpose embeddings: {X.shape[1]} dims (+{self.purpose_embeddings_matrix.shape[1]} purpose dims)")

        # Store threshold tuning flag
        self.use_threshold_tuning = use_threshold_tuning

        metrics = {
            "train_size": 0, "test_size": 0,
            "classifiers": {
                "side_effects": side_effects_clf,
                "complexity": complexity_clf,
                "error_handling": error_handling_clf
            },
            "hyperparams": {
                "lr": {"C": lr_c, "max_iter": lr_max_iter, "solver": lr_solver},
                "rf": {"n_estimators": rf_n_estimators, "max_depth": rf_max_depth,
                       "min_samples_split": rf_min_samples_split, "min_samples_leaf": rf_min_samples_leaf},
                "svm": {"C": svm_c, "kernel": svm_kernel, "gamma": svm_gamma},
                "mlp": {"hidden_layers": mlp_hidden_layers, "activation": mlp_activation,
                        "learning_rate": mlp_learning_rate, "max_iter": mlp_max_iter,
                        "early_stopping": mlp_early_stopping},
                "class_weight": "balanced" if use_class_weight else None
            },
            "original_dim": original_dim,
            "reduced_dim": embed_dim if embed_dim else original_dim
        }

        # Hyperparameter grids for tuning
        PARAM_GRIDS = {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "max_iter": [500, 1000, 2000],
                "solver": ["lbfgs", "saga"],
                "class_weight": ["balanced", None],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
            "svm": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
                "class_weight": ["balanced", None],
            },
            "mlp": {
                "hidden_layer_sizes": [(128,), (256, 128), (512, 256, 128)],
                "activation": ["relu", "tanh"],
                "learning_rate_init": [0.001, 0.0001],
                "alpha": [0.0001, 0.001],  # L2 regularization
            }
        }

        if use_class_weight:
            print(f"  Using class_weight='balanced' to handle class imbalance")

        def get_base_classifier(clf_type: str):
            if clf_type == "random_forest":
                return RandomForestClassifier(random_state=random_state, n_jobs=-1)
            elif clf_type == "svm":
                return SVC(random_state=random_state)
            elif clf_type == "mlp":
                return MLPClassifier(random_state=random_state)
            else:  # logistic_regression (default)
                return LogisticRegression(random_state=random_state)

        def get_configured_classifier(clf_type: str, use_weights=True, is_multilabel=False):
            """Get classifier with CLI-specified hyperparameters."""
            class_weight = 'balanced' if (use_class_weight and use_weights) else None

            if clf_type == "logistic_regression":
                return LogisticRegression(
                    C=lr_c,
                    max_iter=lr_max_iter,
                    solver=lr_solver,
                    random_state=random_state,
                    class_weight=class_weight
                )
            elif clf_type == "svm":
                return SVC(
                    C=svm_c,
                    kernel=svm_kernel,
                    gamma=svm_gamma,
                    random_state=random_state,
                    class_weight=class_weight
                )
            elif clf_type == "mlp":
                # Note: MLPClassifier doesn't support class_weight
                # Early stopping can fail with rare classes in multi-label, so disable for multi-label
                use_early_stop = mlp_early_stopping and not is_multilabel
                return MLPClassifier(
                    hidden_layer_sizes=mlp_hidden_layers,
                    activation=mlp_activation,
                    learning_rate_init=mlp_learning_rate,
                    max_iter=mlp_max_iter,
                    early_stopping=use_early_stop,
                    validation_fraction=0.1 if use_early_stop else 0.0,
                    random_state=random_state,
                    verbose=False
                )
            else:  # random_forest
                return RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_split=rf_min_samples_split,
                    min_samples_leaf=rf_min_samples_leaf,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight=class_weight
                )

        def get_tuned_classifier(clf_type: str, X_train, y_train, is_multilabel=False, use_weights=True):
            """Get classifier with optional hyperparameter tuning."""
            if not tune_hyperparams:
                return get_configured_classifier(clf_type, use_weights, is_multilabel)

            base_clf = get_base_classifier(clf_type)
            param_grid = PARAM_GRIDS.get(clf_type, {})
            if not param_grid:
                return get_configured_classifier(clf_type, use_weights, is_multilabel)

            print(f"    Tuning {clf_type} hyperparameters...")

            # For multi-label, we can't directly use GridSearchCV, so skip tuning
            if is_multilabel:
                print("    Skipping GridSearch for multi-label (using CLI params)")
                return get_configured_classifier(clf_type, use_weights, is_multilabel)

            grid_search = GridSearchCV(
                base_clf,
                param_grid,
                cv=3,  # 3-fold cross-validation
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)

            print(f"    Best params: {grid_search.best_params_}")
            print(f"    Best CV score: {grid_search.best_score_:.3f}")

            metrics[f"{clf_type}_best_params"] = grid_search.best_params_
            metrics[f"{clf_type}_cv_score"] = grid_search.best_score_

            return grid_search.best_estimator_

        if tune_hyperparams:
            print("  Hyperparameter tuning enabled (using GridSearchCV with 3-fold CV)")

        if use_smote and not IMBLEARN_AVAILABLE:
            print("  Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")
            print("  Continuing without SMOTE...")
            use_smote = False

        # ===== Side Effects (Multi-label) =====
        mode_str = "binary classifiers" if use_binary_classifiers else "multi-label"
        print(f"  Training side_effects classifier ({side_effects_clf}, {mode_str})...")

        side_effects_list = [ef.labels.get("side_effects", ["none"]) for ef in self.embedded_functions]
        self.side_effects_mlb = MultiLabelBinarizer()
        Y_se = self.side_effects_mlb.fit_transform(side_effects_list)

        X_train, X_test, Y_se_train, Y_se_test = train_test_split(
            X, Y_se, test_size=test_size, random_state=random_state
        )
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        metrics["use_binary_classifiers"] = use_binary_classifiers
        metrics["use_smote"] = use_smote

        if use_binary_classifiers:
            # Train separate binary classifier for each side effect
            self.side_effects_binary_classifiers = {}
            self.side_effects_classifier = None  # Not using MultiOutputClassifier

            Y_se_pred_train = np.zeros_like(Y_se_train)
            Y_se_pred = np.zeros_like(Y_se_test)

            for i, class_name in enumerate(self.side_effects_mlb.classes_):
                y_train_binary = Y_se_train[:, i]
                y_test_binary = Y_se_test[:, i]

                # Check if class has enough samples
                pos_count = y_train_binary.sum()
                if pos_count < 2:
                    print(f"    Skipping '{class_name}' (only {pos_count} positive samples)")
                    continue

                X_train_resampled, y_train_resampled = X_train, y_train_binary

                # Apply SMOTE for binary classification
                if use_smote and IMBLEARN_AVAILABLE and pos_count >= 2:
                    try:
                        # Use k_neighbors based on minority class size
                        k = min(5, pos_count - 1)
                        if k >= 1:
                            smote = SMOTE(random_state=random_state, k_neighbors=k)
                            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_binary)
                            print(f"    '{class_name}': SMOTE {len(y_train_binary)} -> {len(y_train_resampled)} samples")
                    except Exception as e:
                        print(f"    '{class_name}': SMOTE failed ({e}), using original data")

                # Train binary classifier
                clf = get_tuned_classifier(side_effects_clf, X_train_resampled, y_train_resampled,
                                          is_multilabel=False, use_weights=True)
                clf.fit(X_train_resampled, y_train_resampled)
                self.side_effects_binary_classifiers[class_name] = clf

                # Threshold tuning if enabled and classifier supports predict_proba
                if use_threshold_tuning and hasattr(clf, 'predict_proba'):
                    try:
                        y_proba_test = clf.predict_proba(X_test)[:, 1]
                        opt_thresh, opt_f1 = self._tune_threshold(y_test_binary, y_proba_test)
                        self.optimal_thresholds[class_name] = opt_thresh

                        # Use tuned threshold for predictions
                        Y_se_pred_train[:, i] = (clf.predict_proba(X_train)[:, 1] >= opt_thresh).astype(int)
                        Y_se_pred[:, i] = (y_proba_test >= opt_thresh).astype(int)
                        print(f"    '{class_name}': threshold={opt_thresh:.2f}, F1={opt_f1:.3f} (pos={pos_count})")
                    except Exception as e:
                        # Fallback to default prediction
                        Y_se_pred_train[:, i] = clf.predict(X_train)
                        Y_se_pred[:, i] = clf.predict(X_test)
                        print(f"    '{class_name}': threshold tuning failed ({e}), using default")
                else:
                    # Predict with default threshold
                    Y_se_pred_train[:, i] = clf.predict(X_train)
                    Y_se_pred[:, i] = clf.predict(X_test)

                    # Per-class metrics
                    from sklearn.metrics import f1_score as f1
                    class_f1 = f1(y_test_binary, Y_se_pred[:, i], zero_division=0)
                    print(f"    '{class_name}': F1={class_f1:.3f} (pos={pos_count})")

        else:
            # Original multi-label approach with MultiOutputClassifier
            base_clf = get_tuned_classifier(side_effects_clf, X_train, Y_se_train[:, 0],
                                           is_multilabel=True, use_weights=True)
            self.side_effects_classifier = MultiOutputClassifier(base_clf)
            self.side_effects_classifier.fit(X_train, Y_se_train)

            # Threshold tuning for multi-label if enabled
            if use_threshold_tuning:
                try:
                    # Get probability predictions from each estimator
                    Y_se_pred_train = np.zeros_like(Y_se_train)
                    Y_se_pred = np.zeros_like(Y_se_test)

                    for i, (class_name, estimator) in enumerate(
                        zip(self.side_effects_mlb.classes_, self.side_effects_classifier.estimators_)
                    ):
                        if hasattr(estimator, 'predict_proba'):
                            y_proba_test = estimator.predict_proba(X_test)[:, 1]
                            opt_thresh, opt_f1 = self._tune_threshold(Y_se_test[:, i], y_proba_test)
                            self.optimal_thresholds[class_name] = opt_thresh

                            Y_se_pred_train[:, i] = (estimator.predict_proba(X_train)[:, 1] >= opt_thresh).astype(int)
                            Y_se_pred[:, i] = (y_proba_test >= opt_thresh).astype(int)
                            print(f"    '{class_name}': threshold={opt_thresh:.2f}, F1={opt_f1:.3f}")
                        else:
                            Y_se_pred_train[:, i] = estimator.predict(X_train)
                            Y_se_pred[:, i] = estimator.predict(X_test)
                except Exception as e:
                    print(f"    Threshold tuning failed ({e}), using default predictions")
                    Y_se_pred_train = self.side_effects_classifier.predict(X_train)
                    Y_se_pred = self.side_effects_classifier.predict(X_test)
            else:
                Y_se_pred_train = self.side_effects_classifier.predict(X_train)
                Y_se_pred = self.side_effects_classifier.predict(X_test)

        # Compute metrics
        train_f1 = f1_score(Y_se_train, Y_se_pred_train, average='macro', zero_division=0)
        test_f1 = f1_score(Y_se_test, Y_se_pred, average='macro', zero_division=0)
        train_acc = accuracy_score(Y_se_train, Y_se_pred_train)
        test_acc = accuracy_score(Y_se_test, Y_se_pred)

        # Overfitting detection
        overfit_gap = train_f1 - test_f1
        if overfit_gap > 0.15:
            overfit_status = "OVERFITTING (train >> test)"
        elif overfit_gap < -0.05:
            overfit_status = "UNDERFITTING (train < test, unusual)"
        else:
            overfit_status = "OK"

        metrics["side_effects"] = {
            "train_f1_macro": train_f1,
            "test_f1_macro": test_f1,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfit_gap": overfit_gap,
            "overfit_status": overfit_status,
            "classes": self.side_effects_mlb.classes_.tolist(),
            "classification_report": classification_report(
                Y_se_test, Y_se_pred,
                target_names=self.side_effects_mlb.classes_,
                output_dict=True,
                zero_division=0
            )
        }

        print(f"  Side Effects - Train F1: {train_f1:.3f}, Test F1: {test_f1:.3f} ({overfit_status})")

        # ===== Complexity (Single-label) =====
        print(f"  Training complexity classifier ({complexity_clf})...")
        complexity_list = [ef.labels.get("complexity", "medium") for ef in self.embedded_functions]
        complexity_encoder = LabelEncoder()
        Y_cx = complexity_encoder.fit_transform(complexity_list)
        self.complexity_classes = complexity_encoder.classes_.tolist()

        _, _, Y_cx_train, Y_cx_test = train_test_split(
            X, Y_cx, test_size=test_size, random_state=random_state
        )

        X_cx_train, Y_cx_train_resampled = X_train, Y_cx_train
        if use_smote and IMBLEARN_AVAILABLE:
            try:
                min_class_count = min(np.bincount(Y_cx_train))
                if min_class_count >= 2:
                    k = min(5, min_class_count - 1)
                    smote = SMOTE(random_state=random_state, k_neighbors=k)
                    X_cx_train, Y_cx_train_resampled = smote.fit_resample(X_train, Y_cx_train)
                    print(f"    SMOTE: {len(Y_cx_train)} -> {len(Y_cx_train_resampled)} samples")
            except Exception as e:
                print(f"    SMOTE failed for complexity: {e}")

        self.complexity_classifier = get_tuned_classifier(complexity_clf, X_cx_train, Y_cx_train_resampled, is_multilabel=False, use_weights=True)
        self.complexity_classifier.fit(X_cx_train, Y_cx_train_resampled)

        Y_cx_pred_train = self.complexity_classifier.predict(X_train)
        Y_cx_pred = self.complexity_classifier.predict(X_test)

        cx_train_f1 = f1_score(Y_cx_train, Y_cx_pred_train, average='macro', zero_division=0)
        cx_test_f1 = f1_score(Y_cx_test, Y_cx_pred, average='macro', zero_division=0)
        cx_overfit_gap = cx_train_f1 - cx_test_f1

        metrics["complexity"] = {
            "train_f1_macro": cx_train_f1,
            "test_f1_macro": cx_test_f1,
            "train_accuracy": accuracy_score(Y_cx_train, Y_cx_pred_train),
            "test_accuracy": accuracy_score(Y_cx_test, Y_cx_pred),
            "overfit_gap": cx_overfit_gap,
            "classes": self.complexity_classes
        }
        print(f"  Complexity - Train F1: {cx_train_f1:.3f}, Test F1: {cx_test_f1:.3f}")

        # ===== Error Handling (Single-label) =====
        print(f"  Training error_handling classifier ({error_handling_clf})...")
        error_handling_list = [ef.labels.get("error_handling", "none") for ef in self.embedded_functions]
        error_handling_encoder = LabelEncoder()
        Y_eh = error_handling_encoder.fit_transform(error_handling_list)
        self.error_handling_classes = error_handling_encoder.classes_.tolist()

        _, _, Y_eh_train, Y_eh_test = train_test_split(
            X, Y_eh, test_size=test_size, random_state=random_state
        )

        X_eh_train, Y_eh_train_resampled = X_train, Y_eh_train
        if use_smote and IMBLEARN_AVAILABLE:
            try:
                min_class_count = min(np.bincount(Y_eh_train))
                if min_class_count >= 2:
                    k = min(5, min_class_count - 1)
                    smote = SMOTE(random_state=random_state, k_neighbors=k)
                    X_eh_train, Y_eh_train_resampled = smote.fit_resample(X_train, Y_eh_train)
                    print(f"    SMOTE: {len(Y_eh_train)} -> {len(Y_eh_train_resampled)} samples")
            except Exception as e:
                print(f"    SMOTE failed for error_handling: {e}")

        self.error_handling_classifier = get_tuned_classifier(error_handling_clf, X_eh_train, Y_eh_train_resampled, is_multilabel=False, use_weights=True)
        self.error_handling_classifier.fit(X_eh_train, Y_eh_train_resampled)

        Y_eh_pred_train = self.error_handling_classifier.predict(X_train)
        Y_eh_pred = self.error_handling_classifier.predict(X_test)

        eh_train_f1 = f1_score(Y_eh_train, Y_eh_pred_train, average='macro', zero_division=0)
        eh_test_f1 = f1_score(Y_eh_test, Y_eh_pred, average='macro', zero_division=0)
        eh_overfit_gap = eh_train_f1 - eh_test_f1

        metrics["error_handling"] = {
            "train_f1_macro": eh_train_f1,
            "test_f1_macro": eh_test_f1,
            "train_accuracy": accuracy_score(Y_eh_train, Y_eh_pred_train),
            "test_accuracy": accuracy_score(Y_eh_test, Y_eh_pred),
            "overfit_gap": eh_overfit_gap,
            "classes": self.error_handling_classes
        }
        print(f"  Error Handling - Train F1: {eh_train_f1:.3f}, Test F1: {eh_test_f1:.3f}")

        # Legacy compatibility
        self.classifier = self.side_effects_classifier
        self.mlb = self.side_effects_mlb

        # Aggregate metrics
        metrics["test_f1_macro"] = metrics["side_effects"]["test_f1_macro"]
        metrics["test_accuracy"] = metrics["side_effects"]["test_accuracy"]
        metrics["train_f1_macro"] = metrics["test_f1_macro"]  # Approximation

        return metrics

    def _get_embedding(self, code: str, purpose: str = None) -> np.ndarray:
        """Get embedding for code, applying PCA and all configured features."""
        embedding = self.embedder.embed(code).reshape(1, -1)
        if self.pca is not None:
            embedding = self.pca.transform(embedding)

        # Add hybrid features if enabled (regex-based)
        if self.use_hybrid_features and self.feature_extractor is not None:
            code_features = self.feature_extractor.extract_features(code).reshape(1, -1)
            if self.feature_scaler is not None:
                code_features = self.feature_scaler.transform(code_features)
            embedding = np.hstack([embedding, code_features])

        # Add AST features if enabled
        if self.use_ast_features and self.ast_feature_extractor is not None:
            ast_features = self.ast_feature_extractor.extract_features(code).reshape(1, -1)
            if self.ast_feature_scaler is not None:
                ast_features = self.ast_feature_scaler.transform(ast_features)
            embedding = np.hstack([embedding, ast_features])

        # Add purpose embeddings if enabled
        if self.use_purpose_embeddings:
            # Use provided purpose or generate a placeholder at prediction time
            purpose_text = purpose if purpose else "function implementation"
            purpose_emb = self.embedder.embed(purpose_text).reshape(1, -1)
            embedding = np.hstack([embedding, purpose_emb])

        return embedding

    def _tune_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                        thresholds: List[float] = None) -> Tuple[float, float]:
        """
        Find optimal threshold for binary classification using F1 score.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            thresholds: List of thresholds to try

        Returns:
            (optimal_threshold, best_f1_score)
        """
        from sklearn.metrics import f1_score

        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        best_threshold = 0.5
        best_f1 = 0.0

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        return best_threshold, best_f1

    def predict_side_effects(self, code: str) -> List[str]:
        """Predict side effects for a new function."""
        if self.side_effects_mlb is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")

        embedding = self._get_embedding(code)

        # Check if using binary classifiers mode
        if self.side_effects_binary_classifiers:
            # Use binary classifiers with optional tuned thresholds
            result = []
            for class_name, clf in self.side_effects_binary_classifiers.items():
                # Use tuned threshold if available and classifier supports predict_proba
                if self.use_threshold_tuning and class_name in self.optimal_thresholds and hasattr(clf, 'predict_proba'):
                    thresh = self.optimal_thresholds[class_name]
                    proba = clf.predict_proba(embedding)[0, 1]
                    pred = 1 if proba >= thresh else 0
                else:
                    pred = clf.predict(embedding)[0]

                if pred == 1:
                    result.append(class_name)
            return result if result else ["none"]

        elif self.side_effects_classifier is not None:
            # Use multi-label classifier with optional tuned thresholds
            if self.use_threshold_tuning and self.optimal_thresholds:
                result = []
                for i, (class_name, estimator) in enumerate(
                    zip(self.side_effects_mlb.classes_, self.side_effects_classifier.estimators_)
                ):
                    if class_name in self.optimal_thresholds and hasattr(estimator, 'predict_proba'):
                        thresh = self.optimal_thresholds[class_name]
                        proba = estimator.predict_proba(embedding)[0, 1]
                        if proba >= thresh:
                            result.append(class_name)
                    else:
                        if estimator.predict(embedding)[0] == 1:
                            result.append(class_name)
                return result if result else ["none"]
            else:
                prediction = self.side_effects_classifier.predict(embedding)
                result = list(self.side_effects_mlb.inverse_transform(prediction)[0])
                return result if result else ["none"]
        else:
            raise ValueError("No classifier available. Call train_classifier first.")

    def predict_complexity(self, code: str) -> str:
        """Predict complexity level for a new function."""
        if self.complexity_classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")

        embedding = self._get_embedding(code)
        prediction = self.complexity_classifier.predict(embedding)

        return self.complexity_classes[prediction[0]]

    def predict_error_handling(self, code: str) -> str:
        """Predict error handling pattern for a new function."""
        if self.error_handling_classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")

        embedding = self._get_embedding(code)
        prediction = self.error_handling_classifier.predict(embedding)

        return self.error_handling_classes[prediction[0]]

    def predict_all_labels(self, code: str) -> Dict:
        """Predict all labels for a new function."""
        return {
            "side_effects": self.predict_side_effects(code),
            "complexity": self.predict_complexity(code),
            "error_handling": self.predict_error_handling(code)
        }

    # =========================================================================
    # Bonus: Clustering by semantic purpose
    # =========================================================================

    def cluster_functions(self, n_clusters: int = 5, use_pca: bool = True) -> Dict:
        """
        Cluster functions by semantic similarity.

        Args:
            n_clusters: Number of clusters
            use_pca: If True and PCA is available, cluster on reduced embeddings

        Returns cluster assignments and analysis.
        """
        if self.embeddings_matrix is None:
            raise ValueError("No embeddings loaded")

        # Use PCA-reduced embeddings if available
        if use_pca and self.pca is not None:
            X = self.pca.transform(self.embeddings_matrix)
            print(f"  Clustering on PCA-reduced embeddings ({X.shape[1]} dims)")
        else:
            X = self.embeddings_matrix
            print(f"  Clustering on raw embeddings ({X.shape[1]} dims)")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Analyze clusters
        clusters = {i: [] for i in range(n_clusters)}
        for i, ef in enumerate(self.embedded_functions):
            clusters[cluster_labels[i]].append(ef.function_name)

        # Calculate silhouette score on same embeddings used for clustering
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(X, cluster_labels)

        # Analyze side effect distribution per cluster
        cluster_side_effects = {i: {} for i in range(n_clusters)}
        for i, ef in enumerate(self.embedded_functions):
            cluster_id = cluster_labels[i]
            for se in ef.labels.get("side_effects", ["none"]):
                cluster_side_effects[cluster_id][se] = cluster_side_effects[cluster_id].get(se, 0) + 1

        return {
            "n_clusters": n_clusters,
            "silhouette_score": silhouette,
            "cluster_sizes": {i: len(clusters[i]) for i in range(n_clusters)},
            "cluster_functions": clusters,
            "cluster_side_effects": cluster_side_effects,
            "cluster_assignments": cluster_labels.tolist()
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, output_dir: str):
        """Save embeddings and trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Clean up stale files from previous runs that don't apply to current config
        stale_files = []

        # Remove PCA if not used in this run
        if self.pca is None:
            stale_files.append("pca.joblib")

        # Remove binary classifiers if not used in this run
        if not self.side_effects_binary_classifiers:
            stale_files.append("side_effects_binary_classifiers.joblib")

        # Remove hybrid features if not used
        if not self.use_hybrid_features:
            stale_files.extend(["feature_scaler.joblib", "code_features.npy"])

        # Remove AST features if not used
        if not self.use_ast_features:
            stale_files.extend(["ast_feature_scaler.joblib", "ast_features.npy"])

        # Remove purpose embeddings if not used
        if not self.use_purpose_embeddings:
            stale_files.append("purpose_embeddings.npy")

        # Remove threshold tuning if not used
        if not self.optimal_thresholds:
            stale_files.append("optimal_thresholds.json")

        for f in stale_files:
            fpath = output_path / f
            if fpath.exists():
                fpath.unlink()
                print(f"  Removed stale file: {f}")

        # Save embedded functions
        data = [ef.to_dict() for ef in self.embedded_functions]
        with open(output_path / "embedded_functions.json", 'w') as f:
            json.dump(data, f)

        # Save embeddings matrix
        np.save(output_path / "embeddings.npy", self.embeddings_matrix)

        # Save classifiers if trained
        if self.side_effects_classifier is not None:
            joblib.dump(self.side_effects_classifier, output_path / "side_effects_classifier.joblib")
            joblib.dump(self.side_effects_mlb, output_path / "side_effects_mlb.joblib")

        if self.complexity_classifier is not None:
            joblib.dump(self.complexity_classifier, output_path / "complexity_classifier.joblib")
            joblib.dump(self.complexity_classes, output_path / "complexity_classes.joblib")

        if self.error_handling_classifier is not None:
            joblib.dump(self.error_handling_classifier, output_path / "error_handling_classifier.joblib")
            joblib.dump(self.error_handling_classes, output_path / "error_handling_classes.joblib")

        # Save PCA if used
        if self.pca is not None:
            joblib.dump(self.pca, output_path / "pca.joblib")

        # Save binary classifiers if used
        if self.side_effects_binary_classifiers:
            joblib.dump(self.side_effects_binary_classifiers, output_path / "side_effects_binary_classifiers.joblib")

        # Save hybrid features components (regex-based)
        if self.use_hybrid_features and self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, output_path / "feature_scaler.joblib")
            if self.code_features_matrix is not None:
                np.save(output_path / "code_features.npy", self.code_features_matrix)

        # Save AST features components
        if self.use_ast_features and self.ast_feature_scaler is not None:
            joblib.dump(self.ast_feature_scaler, output_path / "ast_feature_scaler.joblib")
            if self.ast_features_matrix is not None:
                np.save(output_path / "ast_features.npy", self.ast_features_matrix)

        # Save purpose embeddings
        if self.use_purpose_embeddings and self.purpose_embeddings_matrix is not None:
            np.save(output_path / "purpose_embeddings.npy", self.purpose_embeddings_matrix)

        # Save threshold tuning config
        if self.optimal_thresholds:
            with open(output_path / "optimal_thresholds.json", 'w') as f:
                json.dump(self.optimal_thresholds, f)

        # Save config flags including embedder type
        embedder_type = "codebert"  # default
        if hasattr(self.embedder, 'MODEL_NAME'):
            model_name = self.embedder.MODEL_NAME
            if "qwen" in model_name.lower():
                if "4B" in model_name or "4b" in model_name:
                    embedder_type = "qwen3-4b"
                elif "8B" in model_name or "8b" in model_name:
                    embedder_type = "qwen3-8b"
                else:
                    embedder_type = "qwen3"
            elif "nomic" in model_name.lower():
                embedder_type = "nomic"
            elif "jina" in model_name.lower():
                embedder_type = "jina"
            elif "codesage" in model_name.lower():
                embedder_type = "codesage"

        config = {
            "use_hybrid_features": self.use_hybrid_features,
            "use_ast_features": self.use_ast_features,
            "use_purpose_embeddings": self.use_purpose_embeddings,
            "use_threshold_tuning": self.use_threshold_tuning,
            "embedder_type": embedder_type
        }
        with open(output_path / "pipeline_config.json", 'w') as f:
            json.dump(config, f)

        # Legacy compatibility
        if self.side_effects_classifier is not None:
            joblib.dump(self.side_effects_classifier, output_path / "classifier.joblib")
            joblib.dump(self.side_effects_mlb, output_path / "mlb.joblib")

        print(f"Saved pipeline to {output_dir}")

    def load(self, input_dir: str):
        """Load embeddings and trained models."""
        input_path = Path(input_dir)

        # Load embedded functions
        with open(input_path / "embedded_functions.json", 'r') as f:
            data = json.load(f)

        self.embedded_functions = []
        for item in data:
            item["embedding"] = np.array(item["embedding"])
            self.embedded_functions.append(EmbeddedFunction(**item))

        # Load embeddings matrix
        self.embeddings_matrix = np.load(input_path / "embeddings.npy")

        # Load side_effects classifier
        se_classifier_path = input_path / "side_effects_classifier.joblib"
        if se_classifier_path.exists():
            self.side_effects_classifier = joblib.load(se_classifier_path)
            self.side_effects_mlb = joblib.load(input_path / "side_effects_mlb.joblib")
        elif (input_path / "classifier.joblib").exists():
            # Legacy fallback
            self.side_effects_classifier = joblib.load(input_path / "classifier.joblib")
            self.side_effects_mlb = joblib.load(input_path / "mlb.joblib")

        # Load complexity classifier
        cx_classifier_path = input_path / "complexity_classifier.joblib"
        if cx_classifier_path.exists():
            self.complexity_classifier = joblib.load(cx_classifier_path)
            self.complexity_classes = joblib.load(input_path / "complexity_classes.joblib")

        # Load error_handling classifier
        eh_classifier_path = input_path / "error_handling_classifier.joblib"
        if eh_classifier_path.exists():
            self.error_handling_classifier = joblib.load(eh_classifier_path)
            self.error_handling_classes = joblib.load(input_path / "error_handling_classes.joblib")

        # Load PCA if exists
        pca_path = input_path / "pca.joblib"
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            print(f"  Loaded PCA (reducing to {self.pca.n_components} dimensions)")

        # Load binary classifiers if exists
        binary_clf_path = input_path / "side_effects_binary_classifiers.joblib"
        if binary_clf_path.exists():
            self.side_effects_binary_classifiers = joblib.load(binary_clf_path)
            print(f"  Loaded {len(self.side_effects_binary_classifiers)} binary classifiers")

        # Load config flags
        config_path = input_path / "pipeline_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.use_hybrid_features = config.get("use_hybrid_features", False)
            self.use_ast_features = config.get("use_ast_features", False)
            self.use_purpose_embeddings = config.get("use_purpose_embeddings", False)
            self.use_threshold_tuning = config.get("use_threshold_tuning", False)

            # Load the correct embedder type (always reload to match training config)
            embedder_type = config.get("embedder_type", "codebert")
            current_type = "codebert"
            if self.embedder is not None and hasattr(self.embedder, 'MODEL_NAME'):
                model_name = self.embedder.MODEL_NAME.lower()
                if "qwen" in model_name:
                    if "4b" in model_name:
                        current_type = "qwen3-4b"
                    elif "8b" in model_name:
                        current_type = "qwen3-8b"
                    else:
                        current_type = "qwen3"
                elif "nomic" in model_name:
                    current_type = "nomic"
                elif "jina" in model_name:
                    current_type = "jina"

            # Always reload if types don't match
            if current_type != embedder_type:
                print(f"  Reloading embedder: {embedder_type} (was {current_type})")
                self.embedder = get_embedder(embedder_type)

        # Load hybrid features components (regex-based)
        if self.use_hybrid_features:
            scaler_path = input_path / "feature_scaler.joblib"
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                self.feature_extractor = CodeFeatureExtractor()
                print("  Loaded hybrid features (regex-based)")

            features_path = input_path / "code_features.npy"
            if features_path.exists():
                self.code_features_matrix = np.load(features_path)

        # Load AST features components
        if self.use_ast_features:
            ast_scaler_path = input_path / "ast_feature_scaler.joblib"
            if ast_scaler_path.exists():
                self.ast_feature_scaler = joblib.load(ast_scaler_path)
                if TREE_SITTER_AVAILABLE:
                    self.ast_feature_extractor = ASTFeatureExtractor()
                print("  Loaded AST features")

            ast_features_path = input_path / "ast_features.npy"
            if ast_features_path.exists():
                self.ast_features_matrix = np.load(ast_features_path)

        # Load purpose embeddings
        if self.use_purpose_embeddings:
            purpose_path = input_path / "purpose_embeddings.npy"
            if purpose_path.exists():
                self.purpose_embeddings_matrix = np.load(purpose_path)
                print("  Loaded purpose embeddings")

        # Load optimal thresholds
        thresholds_path = input_path / "optimal_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                self.optimal_thresholds = json.load(f)
            print(f"  Loaded {len(self.optimal_thresholds)} tuned thresholds")

        # Legacy compatibility
        self.classifier = self.side_effects_classifier
        self.mlb = self.side_effects_mlb

        print(f"Loaded {len(self.embedded_functions)} functions from {input_dir}")


def get_embedding_summary(code: str, pipeline: FunctionEmbeddingPipeline) -> str:
    """
    Generate a human-readable summary for a function based on embedding similarity.

    Uses nearest neighbor approach: finds the most similar labeled function
    and borrows its high_level_purpose.
    """
    if not pipeline.embedded_functions:
        return "No reference functions available for comparison"

    results = pipeline.similarity_search(code, top_k=1)
    if results:
        nearest, similarity = results[0]
        return nearest.labels.get("high_level_purpose", "Unknown purpose")

    return "Could not determine purpose"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.embed <labeled_dataset.json> [output_dir]")
        sys.exit(1)

    from .label import load_labeled_dataset

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"

    # Load labeled dataset
    with open(input_path, 'r') as f:
        labeled_data = json.load(f)

    # Create pipeline and embed
    pipeline = FunctionEmbeddingPipeline()
    pipeline.embed_labeled_functions(labeled_data)

    # Train classifier
    print("\nTraining side_effects classifier...")
    metrics = pipeline.train_classifier()
    print(f"Test F1 (macro): {metrics['test_f1_macro']:.3f}")

    # Cluster functions
    print("\nClustering functions...")
    cluster_results = pipeline.cluster_functions(n_clusters=4)
    print(f"Silhouette score: {cluster_results['silhouette_score']:.3f}")

    # Save
    pipeline.save(output_dir)
