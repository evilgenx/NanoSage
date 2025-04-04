import re # Added for anchor insertion
from llm_providers.tasks import rag_final_answer
import toc_tree
from aggregator import aggregate_results

def build_final_answer(
    enhanced_query,
    toc_tree_nodes, # Renamed from toc_tree for clarity
    summarized_web,
    summarized_local,
    reference_links, # Passed in from summarize_web_results
    resolved_settings,
    progress_callback,
    previous_results_content="", # Keep optional args if needed
    follow_up_convo=""
):
    """Builds the final RAG answer."""
    toc_str = toc_tree.build_toc_string(toc_tree_nodes) if toc_tree_nodes else "No Table of Contents generated (Web search might be disabled or yielded no relevant branches)." # Use toc_tree
    # Build a reference links string
    reference_links_str = ""
    if reference_links:
        reference_links_str = "\n".join(f"- {link}" for link in reference_links)
    else:
        reference_links_str = "No web reference links found."

    # Retrieve the prompt template content loaded by the worker
    prompt_template_content = resolved_settings.get('rag_prompt_template_content')

    if not prompt_template_content:
        # This should ideally not happen due to the checks in the worker, but handle defensively
        progress_callback("[ERROR] Prompt template content missing in resolved_settings. Cannot generate final answer.")
        return "Error: Could not load prompt template for final report generation."

    # Format the loaded template with the gathered data
    progress_callback("Formatting final RAG prompt...")
    try:
        # Ensure all expected keys are present, even if empty, to avoid KeyError on format
        format_data = {
            'enhanced_query': enhanced_query or "N/A",
            'toc_str': toc_str or "N/A",
            'summarized_web': summarized_web or "N/A",
            'summarized_local': summarized_local or "N/A",
            'reference_links_str': reference_links_str or "N/A",
            # Add any other placeholders your templates might use
        }
        final_prompt = prompt_template_content.format(**format_data)
    except KeyError as e:
        progress_callback(f"[ERROR] Missing key in prompt template formatting: {e}. Check template file placeholders.")
        return f"Error: Prompt template formatting failed due to missing key: {e}"
    except Exception as e:
        progress_callback(f"[ERROR] An unexpected error occurred during prompt formatting: {e}")
        return f"Error: Failed to format prompt template: {e}"

    # Use resolved RAG model
    progress_callback(f"Calling final RAG model ({resolved_settings['rag_model']})...")
    print("[DEBUG] Final RAG prompt formatted. Passing to rag_final_answer()...")
    # Assemble llm_config for the final RAG task
    provider = resolved_settings.get('rag_model', 'gemma')
    model_id = resolved_settings.get(f"{provider}_model_id") # e.g., gemini_model_id
    api_key = resolved_settings.get(f"{provider}_api_key") # e.g., gemini_api_key
    llm_config_for_rag = {
        "provider": provider,
        "model_id": model_id,
        "api_key": api_key,
        "personality": resolved_settings.get('personality')
    }
    final_answer = rag_final_answer(
        final_prompt, # Pass the formatted prompt content
        llm_config=llm_config_for_rag
        # Consider adding progress_callback to rag_final_answer if it's long
    )
    progress_callback("Final RAG generation complete.")

    # --- Insert Anchors ---
    if toc_tree_nodes:
        progress_callback("Inserting anchors into final report...")
        final_answer = _insert_anchors_into_report(final_answer, toc_tree_nodes)
        progress_callback("Anchors inserted.")
    # --- End Insert Anchors ---

    return final_answer

def _insert_anchors_into_report(report_content, toc_nodes):
    """
    Post-processes the report content (Markdown string) to insert HTML anchors
    based on the TOC nodes. Tries to find Markdown headings matching the node's query_text.
    """
    modified_content = report_content
    nodes_to_process = list(toc_nodes) # Flatten the tree for easier iteration? No, need hierarchy for matching order.

    processed_anchors = set() # Keep track of inserted anchors to avoid duplicates if query_text repeats

    def find_and_insert(node, current_content):
        nonlocal processed_anchors
        if not node or not node.query_text or not node.anchor_id or node.anchor_id in processed_anchors:
            return current_content

        # Try to find the heading corresponding to the node's query_text
        # Escape query text for regex and be flexible with surrounding whitespace/newlines
        # Look for lines starting with # and containing the query text
        # This is imperfect as LLM might rephrase.
        escaped_query = re.escape(node.query_text.strip())
        # Pattern: Start of line, 1+ hash marks, optional space, the query text, optional space, end of line.
        # Using re.MULTILINE flag
        pattern = re.compile(r"^(#+)\s*" + escaped_query + r"\s*$", re.MULTILINE | re.IGNORECASE)

        match = pattern.search(current_content)
        if match:
            # Insert anchor before the matched heading line
            # Use re.sub with count=1 to replace only the first occurrence found *after* previous insertions
            # This is still tricky. Let's try replacing the first match globally for this anchor.
            anchor_tag = f'<a name="{node.anchor_id}"></a>\n'
            replacement = anchor_tag + match.group(0) # Prepend anchor tag with a newline

            # Replace only the first occurrence of this specific heading match
            try:
                # Use a temporary placeholder to avoid re-matching issues if query text appears multiple times
                placeholder = f"__ANCHOR_PLACEHOLDER_{node.anchor_id}__"
                temp_content, num_replacements = pattern.subn(placeholder, current_content, count=1)
                if num_replacements > 0:
                    current_content = temp_content.replace(placeholder, replacement, 1)
                    processed_anchors.add(node.anchor_id)
                    print(f"[DEBUG] Inserted anchor '{node.anchor_id}' for heading: {node.query_text}")
                else:
                     print(f"[DEBUG] Could not insert anchor '{node.anchor_id}'. Heading not found for: {node.query_text}")

            except Exception as e:
                 print(f"[ERROR] Error inserting anchor {node.anchor_id}: {e}")
                 # Continue with original content if replacement fails
                 pass # Keep original current_content

        else:
             print(f"[DEBUG] Heading pattern not found for anchor '{node.anchor_id}': {node.query_text}")


        # Process children recursively
        for child in node.children:
            # Pass the potentially modified content down
            current_content = find_and_insert(child, current_content)

        return current_content

    # Process top-level nodes
    for node in toc_nodes:
        modified_content = find_and_insert(node, modified_content)

    return modified_content


def save_report(
    query_id,
    enhanced_query,
    web_results,
    local_results,
    final_answer,
    config,
    grouped_web_results,
    progress_callback,
    previous_results=None,
    follow_up_convo=None
):
    """Saves the final aggregated report."""
    progress_callback("Aggregating results and saving report...")
    print("[INFO] Saving final report to disk...")
    output_path = aggregate_results(
        query_id,
        enhanced_query,
        web_results, # Pass original web results list
        local_results, # Pass original local results list
        final_answer,
        config,
        grouped_web_results=grouped_web_results, # Pass grouped results
        previous_results=previous_results,
        follow_up_conversation=follow_up_convo
        # Removed toc_nodes=self.toc_tree as it's not an expected argument
    )
    progress_callback(f"Report saved to: {output_path}")
    return output_path
