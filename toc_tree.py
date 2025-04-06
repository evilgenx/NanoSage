# toc_tree.py
import uuid

##############################################
# TOC Node: Represents a branch in the search tree
##############################################

class TOCNode:
    # Define possible statuses
    STATUS_PENDING = "Pending"
    STATUS_SKIPPED = "Skipped"
    STATUS_SEARCHING = "Searching"
    STATUS_SUMMARIZING = "Summarizing"
    STATUS_EXPANDING = "Expanding"
    STATUS_DONE = "Done"
    STATUS_ERROR = "Error" # Added for potential error states

    def __init__(self, query_text, depth=1, parent_id=None): # Added parent_id for tree structure tracking
        self.node_id = str(uuid.uuid4())  # Unique identifier for this node
        self.parent_id = parent_id        # ID of the parent node (None for root nodes)
        self.query_text = query_text      # The subquery text for this branch
        self.depth = depth                # Depth level in the tree
        self.summary = ""                 # Summary of findings for this branch
        self.web_results = []             # Web search results for this branch
        # self.corpus_entries = []        # Removed: Corpus entries are added directly to KB, not stored per node
        self.children = []                # Child TOCNode objects for further subqueries
        self.relevance_score = 0.0        # Relevance score relative to the overall query
        self.content_relevance_score = None # Relevance score based on summarized content
        self.anchor_id = ""               # Unique ID for HTML anchors (assigned later)
        self.status = self.STATUS_PENDING # Initial status for GUI tracking

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        # Updated repr to include new fields
        return (f"TOCNode(id='{self.node_id}', text='{self.query_text}', depth={self.depth}, "
                f"status='{self.status}', relevance={self.relevance_score:.2f}, children={len(self.children)})")

    def to_dict(self):
        """Convert node data to a dictionary suitable for sending via signals."""
        return {
            "id": self.node_id,
            "parent_id": self.parent_id,
            "text": self.query_text,
            "depth": self.depth,
            "status": self.status,
            "relevance": f"{self.relevance_score:.2f}" if self.relevance_score is not None else "N/A",
            "content_relevance": f"{self.content_relevance_score:.2f}" if self.content_relevance_score is not None else "N/A",
            "summary_snippet": (self.summary[:50] + "...") if self.summary else ""
        }

def build_toc_string(toc_nodes, indent=0):
    """
    Recursively build a string representation of the TOC tree.
    """
    toc_str = ""
    for node in toc_nodes:
        prefix = "  " * indent + "- "
        summary_snippet = (node.summary[:150] + "...") if node.summary else "No summary"
        toc_str += f"{prefix}{node.query_text} (Relevance: {node.relevance_score:.2f}, Summary: {summary_snippet})\n"
        if node.children:
            toc_str += build_toc_string(node.children, indent=indent+1)
    return toc_str

def assign_anchor_ids(nodes, prefix="toc"):
    """
    Recursively assigns unique, hierarchical anchor IDs to each node in the tree.
    Call this *after* the tree structure is finalized.
    Example IDs: toc-0, toc-1, toc-1-0, toc-1-1, toc-2
    """
    for i, node in enumerate(nodes):
        node.anchor_id = f"{prefix}-{i}"
        if node.children:
            assign_anchor_ids(node.children, prefix=node.anchor_id)
