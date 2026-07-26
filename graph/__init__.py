"""Just-in-time retrieval over an `ast`-derived code graph.

Load identifiers into context; fetch bodies only when the model asks.

    from graph import CodeIndex, GraphTools, run_jit_agent

    index = CodeIndex.build(".")          # walk the repo
    index.save()                          # cheap JSON index
    graph = index.graph()                 # resolved nodes + edges
    tools = GraphTools(graph)             # the four retrieval tools

The call graph is a lower bound, not a complete picture -- see `graph/README.md`
and the module docstring of `graph.builder` for exactly what `ast` cannot see.
"""

from .agent import (
    AgentRun,
    build_outline,
    build_system_prompt,
    run_jit_agent,
)
from .baseline import (
    BaselineRun,
    Chunk,
    LexicalRetriever,
    build_baseline_prompt,
    chunk_file,
    chunk_repo,
    recall_at_k,
    run_baseline,
    sweep_k,
)
from .builder import (
    EDGE_KINDS,
    NODE_KINDS,
    CallRef,
    CodeGraph,
    FileAnalysis,
    ImportRef,
    Node,
    analyze_file,
    analyze_source,
    module_name_for,
)
from .compare import ArmResult, Comparison, build_graph, format_report, run_comparison
from .index import CodeIndex, RefreshReport, discover_python_files, load_graph
from .questions import FIXTURE_QUESTIONS, QUESTION_SETS, REPO_QUESTIONS, Question
from .tokens import (
    LiveTokenCounter,
    OfflineTokenCounter,
    TokenLedger,
    default_counter,
)
from .tools import TOOL_DEFINITIONS, TOOL_NAMES, GraphTools, tokenize

__all__ = [
    # graph construction
    "CodeGraph",
    "CodeIndex",
    "FileAnalysis",
    "Node",
    "ImportRef",
    "CallRef",
    "RefreshReport",
    "NODE_KINDS",
    "EDGE_KINDS",
    "analyze_file",
    "analyze_source",
    "module_name_for",
    "discover_python_files",
    "load_graph",
    "build_graph",
    # retrieval tools
    "GraphTools",
    "TOOL_DEFINITIONS",
    "TOOL_NAMES",
    "tokenize",
    # jit agent
    "run_jit_agent",
    "AgentRun",
    "build_outline",
    "build_system_prompt",
    # baseline
    "run_baseline",
    "BaselineRun",
    "LexicalRetriever",
    "Chunk",
    "chunk_file",
    "chunk_repo",
    "build_baseline_prompt",
    "recall_at_k",
    "sweep_k",
    # comparison
    "run_comparison",
    "Comparison",
    "ArmResult",
    "format_report",
    "Question",
    "REPO_QUESTIONS",
    "FIXTURE_QUESTIONS",
    "QUESTION_SETS",
    # accounting
    "TokenLedger",
    "OfflineTokenCounter",
    "LiveTokenCounter",
    "default_counter",
]
