# toc_tree.py

##############################################
# TOC Node: Represents a branch in the search tree
##############################################

class TOCNode:
    def __init__(self, query_text, depth=1):
        self.query_text = query_text      # The subquery text for this branch
        self.depth = depth                # Depth level in the tree
        self.summary = ""                 # Summary of findings for this branch
        self.web_results = []             # Web search results for this branch
        self.corpus_entries = []          # Corpus entries generated from this branch
        self.children = []                # Child TOCNode objects for further subqueries
        self.relevance_score = 0.0        # Relevance score relative to the overall query
        self.anchor_id = ""               # Unique ID for HTML anchors

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TOCNode(query_text='{self.query_text}', depth={self.depth}, relevance_score={self.relevance_score:.2f}, children={len(self.children)})"

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
