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

    Features are organized into groups:
    - Original (20 features): Memory/IO/HW/Global/Error APIs + code metrics + keywords
    - Embedded/Bare-Metal (20 features): MMIO, regmap, interrupts, spinlocks,
      device model, peripheral protocols, bitwise ops, inline asm, compiler attrs
    """

    # =========================================================================
    # ORIGINAL API patterns (unchanged)
    # =========================================================================

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

    # =========================================================================
    # NEW: Embedded / Bare-Metal / Hardware-Level patterns
    # =========================================================================

    # Group 1: MMIO (Memory-Mapped I/O) register access
    MMIO_APIS = [
        r'\breadl\s*\(', r'\bwritel\s*\(', r'\breadw\s*\(', r'\bwritew\s*\(',
        r'\breadb\s*\(', r'\bwriteb\s*\(', r'\breadq\s*\(',  r'\bwriteq\s*\(',
        r'\bioread32\s*\(', r'\biowrite32\s*\(', r'\bioread16\s*\(', r'\biowrite16\s*\(',
        r'\bioread8\s*\(', r'\biowrite8\s*\(',
        r'\b__iomem\b',
    ]

    # Group 2: Linux regmap abstraction (very common in GPIO/WDT/SPI drivers)
    REGMAP_APIS = [
        r'\bregmap_read\s*\(', r'\bregmap_write\s*\(', r'\bregmap_update_bits\s*\(',
        r'\bregmap_bulk_read\s*\(', r'\bregmap_bulk_write\s*\(',
        r'\bregmap_set_bits\s*\(', r'\bregmap_clear_bits\s*\(',
        r'\bregmap_test_bits\s*\(',
    ]

    # Group 3: Interrupt handling
    IRQ_APIS = [
        r'\brequest_irq\s*\(', r'\bfree_irq\s*\(', r'\bdevm_request_irq\s*\(',
        r'\bdevm_request_threaded_irq\s*\(',
        r'\benable_irq\s*\(', r'\bdisable_irq\s*\(', r'\bdisable_irq_nosync\s*\(',
        r'\birq_set_irq_type\s*\(', r'\birq_to_desc\s*\(',
        r'\bhandle_level_irq\b', r'\bhandle_edge_irq\b',
    ]

    # Group 4: Spinlocks / mutexes / synchronization
    SYNC_APIS = [
        r'\bspin_lock\s*\(', r'\bspin_unlock\s*\(', r'\bspin_lock_irqsave\s*\(',
        r'\bspin_unlock_irqrestore\s*\(', r'\bspin_lock_irq\s*\(', r'\bspin_unlock_irq\s*\(',
        r'\bmutex_lock\s*\(', r'\bmutex_unlock\s*\(',
        r'\batomic_read\s*\(', r'\batomic_set\s*\(', r'\batomic_inc\b', r'\batomic_dec\b',
        r'\bbarrier\s*\(', r'\bmb\s*\(', r'\bwmb\s*\(', r'\brmb\s*\(',
        r'\bsmp_mb\s*\(', r'\bsmp_wmb\s*\(', r'\bsmp_rmb\s*\(',
    ]

    # Group 5: Device managed resources (devm_*)
    DEVM_APIS = [
        r'\bdevm_kzalloc\s*\(', r'\bdevm_kmalloc\s*\(', r'\bdevm_kcalloc\s*\(',
        r'\bdevm_ioremap\s*\(', r'\bdevm_ioremap_resource\s*\(',
        r'\bdevm_clk_get\s*\(', r'\bdevm_clk_get_enabled\s*\(',
        r'\bdevm_gpiod_get\s*\(', r'\bdevm_gpio_request\s*\(',
        r'\bdevm_regulator_get\s*\(', r'\bdevm_reset_control_get\s*\(',
        r'\bdevm_pinctrl_get\s*\(',
    ]

    # Group 6: SPI peripheral protocol
    SPI_APIS = [
        r'\bspi_transfer\b', r'\bspi_message\b', r'\bspi_sync\s*\(',
        r'\bspi_write\s*\(', r'\bspi_read\s*\(', r'\bspi_write_then_read\s*\(',
        r'\bspi_register_driver\s*\(', r'\bspi_unregister_driver\s*\(',
    ]

    # Group 7: I2C peripheral protocol
    I2C_APIS = [
        r'\bi2c_transfer\s*\(', r'\bi2c_master_send\s*\(', r'\bi2c_master_recv\s*\(',
        r'\bi2c_smbus_read_byte\s*\(', r'\bi2c_smbus_write_byte\s*\(',
        r'\bi2c_smbus_read_byte_data\s*\(', r'\bi2c_smbus_write_byte_data\s*\(',
        r'\bi2c_smbus_read_word_data\s*\(', r'\bi2c_smbus_write_word_data\s*\(',
        r'\bi2c_add_driver\s*\(', r'\bi2c_del_driver\s*\(',
    ]

    # Group 8: GPIO subsystem
    GPIO_APIS = [
        r'\bgpio_get_value\s*\(', r'\bgpio_set_value\s*\(',
        r'\bgpio_direction_input\s*\(', r'\bgpio_direction_output\s*\(',
        r'\bgpiod_get_value\s*\(', r'\bgpiod_set_value\s*\(',
        r'\bgpiod_direction_input\s*\(', r'\bgpiod_direction_output\s*\(',
        r'\bgpiochip_get_data\s*\(', r'\bgpiochip_add_data\s*\(',
        r'\bdevm_gpiochip_add_data\s*\(',
    ]

    # Group 9: Watchdog / timer subsystem
    WDT_TIMER_APIS = [
        r'\bwatchdog_init_timeout\s*\(', r'\bwatchdog_register_device\s*\(',
        r'\bdevm_watchdog_register_device\s*\(', r'\bwatchdog_set_drvdata\s*\(',
        r'\bwatchdog_get_drvdata\s*\(', r'\bwatchdog_stop_on_reboot\s*\(',
        r'\bmod_timer\s*\(', r'\bsetup_timer\s*\(', r'\btimer_setup\s*\(',
        r'\bhrtimer_start\s*\(', r'\bhrtimer_init\s*\(',
    ]

    def __init__(self, use_hardware_features: bool = False):
        self.use_hardware_features = use_hardware_features

        # Compile original patterns
        self.memory_patterns = [re.compile(p) for p in self.MEMORY_APIS]
        self.io_patterns = [re.compile(p) for p in self.IO_APIS]
        self.hardware_patterns = [re.compile(p) for p in self.HARDWARE_APIS]
        self.global_patterns = [re.compile(p) for p in self.GLOBAL_STATE_PATTERNS]
        self.error_patterns = [re.compile(p) for p in self.ERROR_HANDLING_PATTERNS]

        # Compile embedded/bare-metal patterns only when enabled
        if not self.use_hardware_features:
            return
        self.mmio_patterns = [re.compile(p) for p in self.MMIO_APIS]
        self.regmap_patterns = [re.compile(p) for p in self.REGMAP_APIS]
        self.irq_patterns = [re.compile(p) for p in self.IRQ_APIS]
        self.sync_patterns = [re.compile(p) for p in self.SYNC_APIS]
        self.devm_patterns = [re.compile(p) for p in self.DEVM_APIS]
        self.spi_patterns = [re.compile(p) for p in self.SPI_APIS]
        self.i2c_patterns = [re.compile(p) for p in self.I2C_APIS]
        self.gpio_patterns = [re.compile(p) for p in self.GPIO_APIS]
        self.wdt_timer_patterns = [re.compile(p) for p in self.WDT_TIMER_APIS]

    def extract_features(self, code: str) -> np.ndarray:
        """
        Extract feature vector from C code.

        Returns a numpy array of 40 features:
        - Original features (20): API presence, code metrics, keywords
        - Embedded/bare-metal features (20): MMIO, regmap, IRQ, sync,
          devm, SPI, I2C, GPIO, WDT/timer, bitwise, asm, compiler attrs,
          driver lifecycle
        """
        features = []

        # =================================================================
        # ORIGINAL FEATURES (20 features, unchanged)
        # =================================================================

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
        features.append(min(lines / 50.0, 1.0))

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

        # =================================================================
        # EMBEDDED / BARE-METAL FEATURES (20 features, only with --hardware-features)
        # =================================================================
        if not self.use_hardware_features:
            return np.array(features, dtype=np.float32)

        # --- Feature 21-22: MMIO register access (readl/writel/ioread32/etc) ---
        mmio_count = sum(1 for p in self.mmio_patterns if p.search(code))
        features.append(min(mmio_count / 4.0, 1.0))   # normalized count
        features.append(1.0 if mmio_count > 0 else 0.0)  # binary presence

        # --- Feature 23-24: Regmap abstraction (regmap_read/write/update_bits) ---
        regmap_count = sum(1 for p in self.regmap_patterns if p.search(code))
        features.append(min(regmap_count / 3.0, 1.0))
        features.append(1.0 if regmap_count > 0 else 0.0)

        # --- Feature 25-26: Interrupt handling (request_irq/enable_irq/etc) ---
        irq_count = sum(1 for p in self.irq_patterns if p.search(code))
        features.append(min(irq_count / 3.0, 1.0))
        features.append(1.0 if irq_count > 0 else 0.0)

        # --- Feature 27-28: Synchronization (spinlock/mutex/atomic/barriers) ---
        sync_count = sum(1 for p in self.sync_patterns if p.search(code))
        features.append(min(sync_count / 4.0, 1.0))
        features.append(1.0 if sync_count > 0 else 0.0)

        # --- Feature 29-30: Device managed resources (devm_*) ---
        devm_count = sum(1 for p in self.devm_patterns if p.search(code))
        features.append(min(devm_count / 3.0, 1.0))
        features.append(1.0 if devm_count > 0 else 0.0)

        # --- Feature 31: SPI peripheral protocol ---
        spi_count = sum(1 for p in self.spi_patterns if p.search(code))
        features.append(1.0 if spi_count > 0 else 0.0)

        # --- Feature 32: I2C peripheral protocol ---
        i2c_count = sum(1 for p in self.i2c_patterns if p.search(code))
        features.append(1.0 if i2c_count > 0 else 0.0)

        # --- Feature 33: GPIO subsystem ---
        gpio_count = sum(1 for p in self.gpio_patterns if p.search(code))
        features.append(1.0 if gpio_count > 0 else 0.0)

        # --- Feature 34: Watchdog / timer subsystem ---
        wdt_count = sum(1 for p in self.wdt_timer_patterns if p.search(code))
        features.append(1.0 if wdt_count > 0 else 0.0)

        # --- Feature 35: Bitwise register manipulation density ---
        # Counts bit shifts and hex mask operations — core pattern in bare-metal code
        bitmask_ops = len(re.findall(r'(?:<<|>>)\s*\d+', code))
        hex_masks = len(re.findall(r'[&|]\s*0x[0-9a-fA-F]+', code))
        bit_macro = len(re.findall(r'\bBIT\s*\(', code)) + len(re.findall(r'\bGENMASK\s*\(', code))
        bitwise_total = bitmask_ops + hex_masks + bit_macro
        features.append(min(bitwise_total / 8.0, 1.0))

        # --- Feature 36: Driver lifecycle function (probe/remove/suspend/resume) ---
        # Check if the function NAME itself indicates a driver lifecycle callback
        func_sig = code.split('{')[0] if '{' in code else code[:200]
        is_lifecycle = 1.0 if re.search(
            r'\b(?:probe|remove|suspend|resume|shutdown|init|exit|attach|detach)\s*\(', func_sig
        ) else 0.0
        features.append(is_lifecycle)

        # --- Feature 37: Platform/bus device struct usage ---
        has_device_struct = 1.0 if re.search(
            r'\b(?:platform_device|pci_dev|i2c_client|spi_device|usb_device|'
            r'platform_driver|i2c_driver|spi_driver)\b', code
        ) else 0.0
        features.append(has_device_struct)

        # --- Feature 38: Kernel module macros ---
        has_module_macro = 1.0 if re.search(
            r'\b(?:module_init|module_exit|MODULE_LICENSE|MODULE_AUTHOR|'
            r'MODULE_DESCRIPTION|module_platform_driver|module_i2c_driver|'
            r'module_spi_driver)\s*\(', code
        ) else 0.0
        features.append(has_module_macro)

        # --- Feature 39: Inline assembly ---
        has_inline_asm = 1.0 if re.search(
            r'\b(?:__asm__|asm\s+volatile|__asm)\b', code
        ) else 0.0
        features.append(has_inline_asm)

        # --- Feature 40: Compiler attributes / builtins ---
        has_compiler_attr = 1.0 if re.search(
            r'(?:__attribute__\s*\(\(|__packed|__aligned|__section|'
            r'\blikely\s*\(|\bunlikely\s*\(|__force\b|__user\b|__kernel\b)', code
        ) else 0.0
        features.append(has_compiler_attr)

        return np.array(features, dtype=np.float32)

    def extract_batch(self, codes: List[str]) -> np.ndarray:
        """Extract features for multiple code snippets."""
        return np.array([self.extract_features(code) for code in codes])

    @property
    def feature_names(self) -> List[str]:
        """Return human-readable feature names."""
        return [
            # Original 20 features
            'memory_count_norm', 'memory_present',
            'io_count_norm', 'io_present',
            'hardware_count_norm', 'hardware_present',
            'global_count_norm', 'global_present',
            'error_count_norm', 'error_present',
            'lines_norm', 'func_calls_norm', 'control_flow_norm',
            'pointer_ops_norm', 'param_count_norm', 'return_count_norm',
            'has_null', 'has_sizeof', 'has_struct_access', 'has_array_access',
            # New 20 embedded/bare-metal features (only with --hardware-features)
            *([ 'mmio_count_norm', 'mmio_present',
            'regmap_count_norm', 'regmap_present',
            'irq_count_norm', 'irq_present',
            'sync_count_norm', 'sync_present',
            'devm_count_norm', 'devm_present',
            'spi_present', 'i2c_present', 'gpio_present', 'wdt_timer_present',
            'bitwise_density', 'is_driver_lifecycle',
            'has_device_struct', 'has_module_macro',
            'has_inline_asm', 'has_compiler_attr',
            ] if self.use_hardware_features else []),
        ]

    @property
    def n_features(self) -> int:
        """Return number of features."""
        return 40 if self.use_hardware_features else 20


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


# =========================================================================
# Everything below this line is IDENTICAL to the original embed.py
# (EmbeddedFunction, all Embedder classes, FunctionEmbeddingPipeline, etc.)
# =========================================================================

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

    Why Qwen3-Embedding:
    - State-of-the-art embedding quality
    - Better semantic understanding than CodeBERT
    - Trained on diverse data including code
    - Strong general-purpose code embedding model
    """

    MODELS = {
        "0.6B": "Qwen/Qwen3-Embedding-0.6B",
        "4B": "Qwen/Qwen3-Embedding-4B",
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

    def embed(self, code: str, max_length: int = 512) -> np.ndarray:
        """Generate embedding for a code snippet using mean pooling. max_length=512 for MPS compat."""
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

    def embed_batch(self, codes: List[str], batch_size: int = 4, max_length: int = 512) -> np.ndarray:
        """Embed multiple code snippets in batches. max_length=512 for MPS compat.
        On MPS, uses batch_size=1 and clears cache between batches to avoid OOM."""
        all_embeddings = []

        # On MPS (Apple Silicon), force batch_size=1 to avoid OOM
        effective_batch_size = batch_size
        if self.device == "mps":
            effective_batch_size = 1

        for i in range(0, len(codes), effective_batch_size):
            batch = codes[i:i + effective_batch_size]

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

            # Clear MPS cache to prevent memory buildup
            if self.device == "mps":
                torch.mps.empty_cache()

        return np.vstack(all_embeddings)


class JinaCodeEmbedder:
    MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
    def __init__(self, device=None):
        if not TRANSFORMERS_AVAILABLE: raise ImportError("transformers and torch not installed")
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading Jina Code Embeddings on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True, use_safetensors=True).to(self.device)
        self.model.eval()
        print("Jina Code Embeddings loaded successfully")

    def embed(self, code, max_length=4096):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return embedding.squeeze().cpu().numpy()

    def embed_batch(self, codes, batch_size=4, max_length=4096):
        all_embeddings = []
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


class CodeSageEmbedder:
    MODEL_NAME = "codesage/codesage-large-v2"
    def __init__(self, device=None):
        if not TRANSFORMERS_AVAILABLE: raise ImportError("transformers and torch not installed")
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading CodeSage on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True, add_eos_token=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to(self.device)
        self.model.eval()
        print("CodeSage loaded successfully")

    def embed(self, code, max_length=1024):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return embedding.squeeze().cpu().numpy()

    def embed_batch(self, codes, batch_size=4, max_length=1024):
        all_embeddings = []
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


class NomicCodeEmbedder:
    MODEL_NAME = "nomic-ai/nomic-embed-code"
    def __init__(self, device=None):
        if not TRANSFORMERS_AVAILABLE: raise ImportError("transformers and torch not installed")
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading nomic-embed-code on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True, use_safetensors=True).to(self.device)
        self.model.eval()
        print("nomic-embed-code loaded successfully")

    def embed(self, code, max_length=4096):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return embedding.squeeze().cpu().numpy()

    def embed_batch(self, codes, batch_size=4, max_length=4096):
        all_embeddings = []
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(self.device)
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
        model_type: "codebert", "qwen3", "qwen3-4b", "nomic", "jina", or "codesage"
        device: "cuda", "mps", or "cpu"

    Available models:
    - codebert: microsoft/codebert-base (768 dim, 512 max tokens)
    - qwen3: Qwen/Qwen3-Embedding-0.6B (1024 dim, 32k context)
    - qwen3-4b: Qwen/Qwen3-Embedding-4B (better quality, 32k context)
    - nomic: nomic-ai/nomic-embed-code (768 dim, 32k context, code-specific)
    - jina: jinaai/jina-embeddings-v2-base-code (768 dim, 8k context, code-specific)
    - codesage: codesage/codesage-large-v2 (2048 dim, 1024 max tokens, code-specific)
    """
    model_type = model_type.lower()

    if model_type == "qwen3":
        return Qwen3Embedder(device=device, model_size="0.6B")
    elif model_type == "qwen3-4b":
        return Qwen3Embedder(device=device, model_size="4B")
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
        self.use_hardware_features: bool = False
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
        self.feature_extractor = CodeFeatureExtractor(use_hardware_features=self.use_hardware_features)
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
        if self.embeddings_matrix is None:
            raise ValueError("No embeddings loaded. Call embed_labeled_functions first.")
        query_embedding = self.embedder.embed(query_code).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.embedded_functions[idx], similarities[idx]))
        return results

    def find_similar_to_function(self, function_name: str, top_k: int = 5) -> List[Tuple[EmbeddedFunction, float]]:
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
                         side_effects_clf: str = "logistic_regression",
                         complexity_clf: str = "logistic_regression",
                         error_handling_clf: str = "logistic_regression",
                         tune_hyperparams: bool = False,
                         lr_c: float = 1.0, lr_max_iter: int = 1000, lr_solver: str = "lbfgs",
                         rf_n_estimators: int = 100, rf_max_depth: Optional[int] = 20,
                         rf_min_samples_split: int = 5, rf_min_samples_leaf: int = 1,
                         svm_c: float = 1.0, svm_kernel: str = "rbf", svm_gamma: str = "scale",
                         mlp_hidden_layers: tuple = (256, 128), mlp_activation: str = "relu",
                         mlp_learning_rate: float = 0.001, mlp_max_iter: int = 500,
                         mlp_early_stopping: bool = True,
                         use_class_weight: bool = True, embed_dim: Optional[int] = None,
                         use_binary_classifiers: bool = False, use_smote: bool = False,
                         use_hybrid_features: bool = False, use_ast_features: bool = False,
                         use_purpose_embeddings: bool = False, use_threshold_tuning: bool = False) -> Dict:
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn not installed")
        if self.embeddings_matrix is None: raise ValueError("No embeddings loaded")

        from sklearn.metrics import f1_score, accuracy_score, classification_report
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import GridSearchCV

        X_embed = self.embeddings_matrix
        original_dim = X_embed.shape[1]

        n_samples = X_embed.shape[0]
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        if embed_dim is not None and embed_dim < original_dim:
            print(f"  Reducing embeddings from {original_dim} to {embed_dim} dimensions using PCA...")
            self.pca = PCA(n_components=embed_dim, random_state=random_state)
            X_train = self.pca.fit_transform(X_embed[train_idx])
            X_test = self.pca.transform(X_embed[test_idx])
            variance_retained = sum(self.pca.explained_variance_ratio_) * 100
            print(f"  Variance retained: {variance_retained:.1f}%")
        else:
            self.pca = None
            X_train = X_embed[train_idx].copy()
            X_test = X_embed[test_idx].copy()

        self.use_hybrid_features = use_hybrid_features
        if use_hybrid_features:
            if self.code_features_matrix is None:
                raise ValueError("Code features not extracted. Re-run embed_labeled_functions.")
            self.feature_scaler = StandardScaler()
            cf_train = self.feature_scaler.fit_transform(self.code_features_matrix[train_idx])
            cf_test = self.feature_scaler.transform(self.code_features_matrix[test_idx])
            X_train = np.hstack([X_train, cf_train])
            X_test = np.hstack([X_test, cf_test])
            print(f"  Hybrid features: {X_train.shape[1]} dims (+{self.code_features_matrix.shape[1]} regex features)")

        self.use_ast_features = use_ast_features
        if use_ast_features:
            if self.ast_features_matrix is None:
                if TREE_SITTER_AVAILABLE:
                    raise ValueError("AST features not extracted. Re-run embed_labeled_functions.")
                else:
                    print("  Warning: tree-sitter not available, skipping AST features")
            else:
                self.ast_feature_scaler = StandardScaler()
                ast_train = self.ast_feature_scaler.fit_transform(self.ast_features_matrix[train_idx])
                ast_test = self.ast_feature_scaler.transform(self.ast_features_matrix[test_idx])
                X_train = np.hstack([X_train, ast_train])
                X_test = np.hstack([X_test, ast_test])
                print(f"  AST features: {X_train.shape[1]} dims (+{self.ast_features_matrix.shape[1]} AST features)")

        self.use_purpose_embeddings = use_purpose_embeddings
        if use_purpose_embeddings:
            if self.purpose_embeddings_matrix is None:
                print("  Warning: No purpose embeddings available, skipping")
            else:
                X_train = np.hstack([X_train, self.purpose_embeddings_matrix[train_idx]])
                X_test = np.hstack([X_test, self.purpose_embeddings_matrix[test_idx]])
                print(f"  Purpose embeddings: {X_train.shape[1]} dims (+{self.purpose_embeddings_matrix.shape[1]} purpose dims)")

        self.use_threshold_tuning = use_threshold_tuning

        metrics = {"train_size": 0, "test_size": 0, "classifiers": {"side_effects": side_effects_clf, "complexity": complexity_clf, "error_handling": error_handling_clf}, "hyperparams": {"lr": {"C": lr_c, "max_iter": lr_max_iter, "solver": lr_solver}, "rf": {"n_estimators": rf_n_estimators, "max_depth": rf_max_depth, "min_samples_split": rf_min_samples_split, "min_samples_leaf": rf_min_samples_leaf}, "svm": {"C": svm_c, "kernel": svm_kernel, "gamma": svm_gamma}, "mlp": {"hidden_layers": mlp_hidden_layers, "activation": mlp_activation, "learning_rate": mlp_learning_rate, "max_iter": mlp_max_iter, "early_stopping": mlp_early_stopping}, "class_weight": "balanced" if use_class_weight else None}, "original_dim": original_dim, "reduced_dim": embed_dim if embed_dim else original_dim}

        if use_class_weight:
            print(f"  Using class_weight='balanced' to handle class imbalance")

        def get_configured_classifier(clf_type, use_weights=True, is_multilabel=False):
            class_weight = 'balanced' if (use_class_weight and use_weights) else None
            if clf_type == "logistic_regression":
                return LogisticRegression(C=lr_c, max_iter=lr_max_iter, solver=lr_solver, random_state=random_state, class_weight=class_weight)
            elif clf_type == "svm":
                return SVC(C=svm_c, kernel=svm_kernel, gamma=svm_gamma, random_state=random_state, class_weight=class_weight)
            elif clf_type == "mlp":
                use_early_stop = mlp_early_stopping and not is_multilabel
                return MLPClassifier(hidden_layer_sizes=mlp_hidden_layers, activation=mlp_activation, learning_rate_init=mlp_learning_rate, max_iter=mlp_max_iter, early_stopping=use_early_stop, validation_fraction=0.1 if use_early_stop else 0.0, random_state=random_state, verbose=False)
            else:
                return RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_split=rf_min_samples_split, min_samples_leaf=rf_min_samples_leaf, random_state=random_state, n_jobs=-1, class_weight=class_weight)

        if use_smote and not IMBLEARN_AVAILABLE:
            print("  Warning: imbalanced-learn not installed. Continuing without SMOTE...")
            use_smote = False

        # ===== Side Effects (Multi-label) =====
        mode_str = "binary classifiers" if use_binary_classifiers else "multi-label"
        print(f"  Training side_effects classifier ({side_effects_clf}, {mode_str})...")

        side_effects_list = [ef.labels.get("side_effects", ["none"]) for ef in self.embedded_functions]
        self.side_effects_mlb = MultiLabelBinarizer()
        Y_se = self.side_effects_mlb.fit_transform(side_effects_list)
        Y_se_train = Y_se[train_idx]
        Y_se_test = Y_se[test_idx]

        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)
        metrics["use_binary_classifiers"] = use_binary_classifiers
        metrics["use_smote"] = use_smote

        X_train_fit = X_train
        Y_se_train_fit = Y_se_train

        if use_threshold_tuning:
            _fit_idx, _val_idx = train_test_split(np.arange(len(X_train)), test_size=0.2, random_state=random_state)
            X_train_fit = X_train[_fit_idx]
            X_val = X_train[_val_idx]
            Y_se_train_fit = Y_se_train[_fit_idx]
            Y_se_val = Y_se_train[_val_idx]

        if not use_binary_classifiers:
            base_clf = get_configured_classifier(side_effects_clf, use_weights=True, is_multilabel=True)
            self.side_effects_classifier = MultiOutputClassifier(base_clf)
            self.side_effects_classifier.fit(X_train_fit, Y_se_train_fit)
            Y_se_pred_train = self.side_effects_classifier.predict(X_train)
            Y_se_pred = self.side_effects_classifier.predict(X_test)
        else:
            self.side_effects_binary_classifiers = {}
            self.side_effects_classifier = None
            Y_se_pred_train = np.zeros_like(Y_se_train)
            Y_se_pred = np.zeros_like(Y_se_test)
            for i, class_name in enumerate(self.side_effects_mlb.classes_):
                y_fit_binary = Y_se_train_fit[:, i]
                pos_count = y_fit_binary.sum()
                if pos_count < 2:
                    print(f"    Skipping '{class_name}' (only {pos_count} positive samples)")
                    continue
                clf = get_configured_classifier(side_effects_clf, use_weights=True, is_multilabel=False)
                clf.fit(X_train_fit, y_fit_binary)
                self.side_effects_binary_classifiers[class_name] = clf
                Y_se_pred_train[:, i] = clf.predict(X_train)
                Y_se_pred[:, i] = clf.predict(X_test)
                from sklearn.metrics import f1_score as f1
                class_f1 = f1(Y_se_test[:, i], Y_se_pred[:, i], zero_division=0)
                print(f"    '{class_name}': F1={class_f1:.3f} (pos={pos_count})")

        train_f1 = f1_score(Y_se_train, Y_se_pred_train, average='macro', zero_division=0)
        test_f1 = f1_score(Y_se_test, Y_se_pred, average='macro', zero_division=0)
        train_acc = accuracy_score(Y_se_train, Y_se_pred_train)
        test_acc = accuracy_score(Y_se_test, Y_se_pred)

        overfit_gap = train_f1 - test_f1
        if overfit_gap > 0.15: overfit_status = "OVERFITTING (train >> test)"
        elif overfit_gap < -0.05: overfit_status = "UNDERFITTING (train < test, unusual)"
        else: overfit_status = "OK"

        metrics["side_effects"] = {"train_f1_macro": train_f1, "test_f1_macro": test_f1, "train_accuracy": train_acc, "test_accuracy": test_acc, "overfit_gap": overfit_gap, "overfit_status": overfit_status, "classes": self.side_effects_mlb.classes_.tolist(), "classification_report": classification_report(Y_se_test, Y_se_pred, target_names=self.side_effects_mlb.classes_, output_dict=True, zero_division=0)}
        print(f"  Side Effects - Train F1: {train_f1:.3f}, Test F1: {test_f1:.3f} ({overfit_status})")

        # ===== Complexity (Single-label) =====
        print(f"  Training complexity classifier ({complexity_clf})...")
        complexity_list = [ef.labels.get("complexity", "medium") for ef in self.embedded_functions]
        complexity_encoder = LabelEncoder()
        Y_cx = complexity_encoder.fit_transform(complexity_list)
        self.complexity_classes = complexity_encoder.classes_.tolist()
        Y_cx_train = Y_cx[train_idx]
        Y_cx_test = Y_cx[test_idx]

        self.complexity_classifier = get_configured_classifier(complexity_clf, use_weights=True, is_multilabel=False)
        self.complexity_classifier.fit(X_train, Y_cx_train)
        Y_cx_pred_train = self.complexity_classifier.predict(X_train)
        Y_cx_pred = self.complexity_classifier.predict(X_test)

        cx_train_f1 = f1_score(Y_cx_train, Y_cx_pred_train, average='macro', zero_division=0)
        cx_test_f1 = f1_score(Y_cx_test, Y_cx_pred, average='macro', zero_division=0)

        metrics["complexity"] = {"train_f1_macro": cx_train_f1, "test_f1_macro": cx_test_f1, "train_accuracy": accuracy_score(Y_cx_train, Y_cx_pred_train), "test_accuracy": accuracy_score(Y_cx_test, Y_cx_pred), "classes": self.complexity_classes}
        print(f"  Complexity - Train F1: {cx_train_f1:.3f}, Test F1: {cx_test_f1:.3f}")

        # ===== Error Handling (Single-label) =====
        print(f"  Training error_handling classifier ({error_handling_clf})...")
        error_handling_list = [ef.labels.get("error_handling", "none") for ef in self.embedded_functions]
        error_handling_encoder = LabelEncoder()
        Y_eh = error_handling_encoder.fit_transform(error_handling_list)
        self.error_handling_classes = error_handling_encoder.classes_.tolist()
        Y_eh_train = Y_eh[train_idx]
        Y_eh_test = Y_eh[test_idx]

        self.error_handling_classifier = get_configured_classifier(error_handling_clf, use_weights=True, is_multilabel=False)
        self.error_handling_classifier.fit(X_train, Y_eh_train)
        Y_eh_pred_train = self.error_handling_classifier.predict(X_train)
        Y_eh_pred = self.error_handling_classifier.predict(X_test)

        eh_train_f1 = f1_score(Y_eh_train, Y_eh_pred_train, average='macro', zero_division=0)
        eh_test_f1 = f1_score(Y_eh_test, Y_eh_pred, average='macro', zero_division=0)

        metrics["error_handling"] = {"train_f1_macro": eh_train_f1, "test_f1_macro": eh_test_f1, "train_accuracy": accuracy_score(Y_eh_train, Y_eh_pred_train), "test_accuracy": accuracy_score(Y_eh_test, Y_eh_pred), "classes": self.error_handling_classes}
        print(f"  Error Handling - Train F1: {eh_train_f1:.3f}, Test F1: {eh_test_f1:.3f}")

        # Legacy compatibility
        self.classifier = self.side_effects_classifier
        self.mlb = self.side_effects_mlb

        # Aggregate metrics
        metrics["test_f1_macro"] = metrics["side_effects"]["test_f1_macro"]
        metrics["test_accuracy"] = metrics["side_effects"]["test_accuracy"]
        metrics["train_f1_macro"] = metrics["side_effects"]["train_f1_macro"]

        return metrics

    def _get_embedding(self, code: str, purpose: str = None) -> np.ndarray:
        embedding = self.embedder.embed(code).reshape(1, -1)
        if self.pca is not None:
            embedding = self.pca.transform(embedding)
        if self.use_hybrid_features and self.feature_extractor is not None:
            code_features = self.feature_extractor.extract_features(code).reshape(1, -1)
            if self.feature_scaler is not None:
                code_features = self.feature_scaler.transform(code_features)
            embedding = np.hstack([embedding, code_features])
        if self.use_ast_features and self.ast_feature_extractor is not None:
            ast_features = self.ast_feature_extractor.extract_features(code).reshape(1, -1)
            if self.ast_feature_scaler is not None:
                ast_features = self.ast_feature_scaler.transform(ast_features)
            embedding = np.hstack([embedding, ast_features])
        if self.use_purpose_embeddings:
            purpose_text = purpose if purpose else "function implementation"
            purpose_emb = self.embedder.embed(purpose_text).reshape(1, -1)
            embedding = np.hstack([embedding, purpose_emb])
        return embedding

    def _tune_threshold(self, y_true, y_proba, thresholds=None):
        from sklearn.metrics import f1_score
        if thresholds is None: thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_threshold = 0.5
        best_f1 = 0.0
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1: best_f1 = f1; best_threshold = thresh
        return best_threshold, best_f1

    def predict_side_effects(self, code: str) -> List[str]:
        if self.side_effects_mlb is None: raise ValueError("Classifier not trained.")
        embedding = self._get_embedding(code)
        if self.side_effects_binary_classifiers:
            result = []
            for class_name, clf in self.side_effects_binary_classifiers.items():
                if self.use_threshold_tuning and class_name in self.optimal_thresholds and hasattr(clf, 'predict_proba'):
                    thresh = self.optimal_thresholds[class_name]
                    proba = clf.predict_proba(embedding)[0, 1]
                    pred = 1 if proba >= thresh else 0
                else:
                    pred = clf.predict(embedding)[0]
                if pred == 1: result.append(class_name)
            return result if result else ["none"]
        elif self.side_effects_classifier is not None:
            prediction = self.side_effects_classifier.predict(embedding)
            result = list(self.side_effects_mlb.inverse_transform(prediction)[0])
            return result if result else ["none"]
        else:
            raise ValueError("No classifier available.")

    def predict_complexity(self, code: str) -> str:
        if self.complexity_classifier is None: raise ValueError("Classifier not trained.")
        embedding = self._get_embedding(code)
        prediction = self.complexity_classifier.predict(embedding)
        return self.complexity_classes[prediction[0]]

    def predict_error_handling(self, code: str) -> str:
        if self.error_handling_classifier is None: raise ValueError("Classifier not trained.")
        embedding = self._get_embedding(code)
        prediction = self.error_handling_classifier.predict(embedding)
        return self.error_handling_classes[prediction[0]]

    def predict_all_labels(self, code: str) -> Dict:
        return {"side_effects": self.predict_side_effects(code), "complexity": self.predict_complexity(code), "error_handling": self.predict_error_handling(code)}

    # =========================================================================
    # Bonus: Clustering by semantic purpose
    # =========================================================================

    def cluster_functions(self, n_clusters: int = 5, use_pca: bool = True) -> Dict:
        if self.embeddings_matrix is None: raise ValueError("No embeddings loaded")
        if use_pca and self.pca is not None:
            X = self.pca.transform(self.embeddings_matrix)
            print(f"  Clustering on PCA-reduced embeddings ({X.shape[1]} dims)")
        else:
            X = self.embeddings_matrix
            print(f"  Clustering on raw embeddings ({X.shape[1]} dims)")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        clusters = {i: [] for i in range(n_clusters)}
        for i, ef in enumerate(self.embedded_functions):
            clusters[cluster_labels[i]].append(ef.function_name)
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(X, cluster_labels)
        cluster_side_effects = {i: {} for i in range(n_clusters)}
        for i, ef in enumerate(self.embedded_functions):
            cluster_id = cluster_labels[i]
            for se in ef.labels.get("side_effects", ["none"]):
                cluster_side_effects[cluster_id][se] = cluster_side_effects[cluster_id].get(se, 0) + 1
        return {"n_clusters": n_clusters, "silhouette_score": silhouette, "cluster_sizes": {i: len(clusters[i]) for i in range(n_clusters)}, "cluster_functions": clusters, "cluster_side_effects": cluster_side_effects, "cluster_assignments": cluster_labels.tolist()}

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stale_files = []
        if self.pca is None: stale_files.append("pca.joblib")
        if not self.side_effects_binary_classifiers: stale_files.append("side_effects_binary_classifiers.joblib")
        if not self.use_hybrid_features: stale_files.extend(["feature_scaler.joblib", "code_features.npy"])
        if not self.use_ast_features: stale_files.extend(["ast_feature_scaler.joblib", "ast_features.npy"])
        if not self.use_purpose_embeddings: stale_files.append("purpose_embeddings.npy")
        if not self.optimal_thresholds: stale_files.append("optimal_thresholds.json")
        for f in stale_files:
            fpath = output_path / f
            if fpath.exists(): fpath.unlink(); print(f"  Removed stale file: {f}")

        data = [ef.to_dict() for ef in self.embedded_functions]
        with open(output_path / "embedded_functions.json", 'w') as f: json.dump(data, f)
        np.save(output_path / "embeddings.npy", self.embeddings_matrix)
        if self.side_effects_classifier is not None:
            joblib.dump(self.side_effects_classifier, output_path / "side_effects_classifier.joblib")
            joblib.dump(self.side_effects_mlb, output_path / "side_effects_mlb.joblib")
        if self.complexity_classifier is not None:
            joblib.dump(self.complexity_classifier, output_path / "complexity_classifier.joblib")
            joblib.dump(self.complexity_classes, output_path / "complexity_classes.joblib")
        if self.error_handling_classifier is not None:
            joblib.dump(self.error_handling_classifier, output_path / "error_handling_classifier.joblib")
            joblib.dump(self.error_handling_classes, output_path / "error_handling_classes.joblib")
        if self.pca is not None: joblib.dump(self.pca, output_path / "pca.joblib")
        if self.side_effects_binary_classifiers: joblib.dump(self.side_effects_binary_classifiers, output_path / "side_effects_binary_classifiers.joblib")
        if self.use_hybrid_features and self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, output_path / "feature_scaler.joblib")
            if self.code_features_matrix is not None: np.save(output_path / "code_features.npy", self.code_features_matrix)
        if self.use_ast_features and self.ast_feature_scaler is not None:
            joblib.dump(self.ast_feature_scaler, output_path / "ast_feature_scaler.joblib")
            if self.ast_features_matrix is not None: np.save(output_path / "ast_features.npy", self.ast_features_matrix)
        if self.use_purpose_embeddings and self.purpose_embeddings_matrix is not None:
            np.save(output_path / "purpose_embeddings.npy", self.purpose_embeddings_matrix)
        if self.optimal_thresholds:
            with open(output_path / "optimal_thresholds.json", 'w') as f: json.dump(self.optimal_thresholds, f)

        embedder_type = "codebert"
        if hasattr(self.embedder, 'MODEL_NAME'):
            model_name = self.embedder.MODEL_NAME
            if "qwen" in model_name.lower():
                if "4B" in model_name or "4b" in model_name: embedder_type = "qwen3-4b"
                else: embedder_type = "qwen3"
            elif "nomic" in model_name.lower(): embedder_type = "nomic"
            elif "jina" in model_name.lower(): embedder_type = "jina"
            elif "codesage" in model_name.lower(): embedder_type = "codesage"

        config = {"use_hybrid_features": self.use_hybrid_features, "use_hardware_features": self.use_hardware_features, "use_ast_features": self.use_ast_features, "use_purpose_embeddings": self.use_purpose_embeddings, "use_threshold_tuning": self.use_threshold_tuning, "embedder_type": embedder_type}
        with open(output_path / "pipeline_config.json", 'w') as f: json.dump(config, f)

        if self.side_effects_classifier is not None:
            joblib.dump(self.side_effects_classifier, output_path / "classifier.joblib")
            joblib.dump(self.side_effects_mlb, output_path / "mlb.joblib")

        print(f"Saved pipeline to {output_dir}")

    def load(self, input_dir: str):
        input_path = Path(input_dir)
        with open(input_path / "embedded_functions.json", 'r') as f: data = json.load(f)
        self.embedded_functions = []
        for item in data:
            item["embedding"] = np.array(item["embedding"])
            self.embedded_functions.append(EmbeddedFunction(**item))
        self.embeddings_matrix = np.load(input_path / "embeddings.npy")

        se_classifier_path = input_path / "side_effects_classifier.joblib"
        if se_classifier_path.exists():
            self.side_effects_classifier = joblib.load(se_classifier_path)
            self.side_effects_mlb = joblib.load(input_path / "side_effects_mlb.joblib")
        elif (input_path / "classifier.joblib").exists():
            self.side_effects_classifier = joblib.load(input_path / "classifier.joblib")
            self.side_effects_mlb = joblib.load(input_path / "mlb.joblib")

        cx_classifier_path = input_path / "complexity_classifier.joblib"
        if cx_classifier_path.exists():
            self.complexity_classifier = joblib.load(cx_classifier_path)
            self.complexity_classes = joblib.load(input_path / "complexity_classes.joblib")

        eh_classifier_path = input_path / "error_handling_classifier.joblib"
        if eh_classifier_path.exists():
            self.error_handling_classifier = joblib.load(eh_classifier_path)
            self.error_handling_classes = joblib.load(input_path / "error_handling_classes.joblib")

        pca_path = input_path / "pca.joblib"
        if pca_path.exists(): self.pca = joblib.load(pca_path); print(f"  Loaded PCA (reducing to {self.pca.n_components} dimensions)")
        binary_clf_path = input_path / "side_effects_binary_classifiers.joblib"
        if binary_clf_path.exists(): self.side_effects_binary_classifiers = joblib.load(binary_clf_path); print(f"  Loaded {len(self.side_effects_binary_classifiers)} binary classifiers")

        config_path = input_path / "pipeline_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f: config = json.load(f)
            self.use_hybrid_features = config.get("use_hybrid_features", False)
            self.use_hardware_features = config.get("use_hardware_features", False)
            self.use_ast_features = config.get("use_ast_features", False)
            self.use_purpose_embeddings = config.get("use_purpose_embeddings", False)
            self.use_threshold_tuning = config.get("use_threshold_tuning", False)
            embedder_type = config.get("embedder_type", "codebert")
            current_type = "codebert"
            if self.embedder is not None and hasattr(self.embedder, 'MODEL_NAME'):
                model_name = self.embedder.MODEL_NAME.lower()
                if "qwen" in model_name:
                    if "4b" in model_name: current_type = "qwen3-4b"
                    else: current_type = "qwen3"
                elif "nomic" in model_name: current_type = "nomic"
                elif "jina" in model_name: current_type = "jina"
            if current_type != embedder_type:
                print(f"  Reloading embedder: {embedder_type} (was {current_type})")
                self.embedder = get_embedder(embedder_type)

        if self.use_hybrid_features:
            scaler_path = input_path / "feature_scaler.joblib"
            if scaler_path.exists(): self.feature_scaler = joblib.load(scaler_path); self.feature_extractor = CodeFeatureExtractor(use_hardware_features=self.use_hardware_features); print(f"  Loaded hybrid features ({'40 HW' if self.use_hardware_features else '20 base'} regex features)")
            features_path = input_path / "code_features.npy"
            if features_path.exists(): self.code_features_matrix = np.load(features_path)

        if self.use_ast_features:
            ast_scaler_path = input_path / "ast_feature_scaler.joblib"
            if ast_scaler_path.exists(): self.ast_feature_scaler = joblib.load(ast_scaler_path); self.ast_feature_extractor = ASTFeatureExtractor() if TREE_SITTER_AVAILABLE else None; print("  Loaded AST features")
            ast_features_path = input_path / "ast_features.npy"
            if ast_features_path.exists(): self.ast_features_matrix = np.load(ast_features_path)

        if self.use_purpose_embeddings:
            purpose_path = input_path / "purpose_embeddings.npy"
            if purpose_path.exists(): self.purpose_embeddings_matrix = np.load(purpose_path); print("  Loaded purpose embeddings")

        thresholds_path = input_path / "optimal_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f: self.optimal_thresholds = json.load(f)
            print(f"  Loaded {len(self.optimal_thresholds)} tuned thresholds")

        self.classifier = self.side_effects_classifier
        self.mlb = self.side_effects_mlb
        print(f"Loaded {len(self.embedded_functions)} functions from {input_dir}")


def get_embedding_summary(code: str, pipeline: FunctionEmbeddingPipeline) -> str:
    if not pipeline.embedded_functions: return "No reference functions available for comparison"
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
    with open(input_path, 'r') as f: labeled_data = json.load(f)
    pipeline = FunctionEmbeddingPipeline()
    pipeline.embed_labeled_functions(labeled_data)
    print("\nTraining side_effects classifier...")
    metrics = pipeline.train_classifier()
    print(f"Test F1 (macro): {metrics['test_f1_macro']:.3f}")
    print("\nClustering functions...")
    cluster_results = pipeline.cluster_functions(n_clusters=4)
    print(f"Silhouette score: {cluster_results['silhouette_score']:.3f}")
    pipeline.save(output_dir)