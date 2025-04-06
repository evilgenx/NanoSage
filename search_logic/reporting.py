import re # Added for anchor insertion
from llm_providers.tasks import rag_final_answer
import toc_tree
from aggregator import aggregate_results
from typing import Optional, Dict, Any # Added import
# Import send_progress helper
from .web_recursive import send_progress # Use relative import

def build_final_answer(
    enhanced_query,
    toc_tree_nodes, # Renamed from toc_tree for clarity
    summarized_web,
    summarized_local,
    reference_links, # Passed in from summarize_web_results
    resolved_settings,
    progress_callback,
    include_visuals=False, # Add new parameter
    previous_results_content="", # Keep optional args if needed
    follow_up_convo=""
):
    """Builds the final RAG answer, optionally including visuals."""
    send_progress(progress_callback, "phase_start", {"phase": "build_report", "message": "Building final report content..."})

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
        err_msg = "Prompt template content missing in resolved_settings. Cannot generate final answer."
        send_progress(progress_callback, "error", {"message": err_msg})
        send_progress(progress_callback, "phase_end", {"phase": "build_report", "message": "Report building failed (missing template)."})
        return f"Error: {err_msg}"

    # --- Conditionally add visual instructions ---
    if include_visuals:
        vis_msg = "Adding visual instructions to prompt..."
        send_progress(progress_callback, "log", {"level": "info", "message": vis_msg})
        visual_instructions = """
# Visual Content Guidelines (Apply ONLY if relevant and adds significant value)
*   **Images:** Where appropriate (e.g., illustrating a concept, showing a specific item mentioned), embed relevant images using Markdown: `![Descriptive Alt Text](URL)`. Prioritize using image URLs found in the {reference_links_str} if suitable.
*   **Maps:** If a specific geographic location (city, region, country) is a key subject, embed a static map using OpenStreetMap: `![Map of LOCATION](https://render.openstreetmap.org/cgi-bin/export?bbox=MINLON,MINLAT,MAXLON,MAXLAT&scale=10000&format=png)`. You MUST determine appropriate MINLON, MINLAT, MAXLON, MAXLAT bounding box values for the LOCATION. Replace LOCATION in the alt text. Use a reasonable scale.
*   **Placement:** Insert visuals logically within the relevant sections of the report body. Do not clutter the report unnecessarily.
---
"""
        # Prepend instructions to the main template content
        prompt_template_content = visual_instructions + "\n" + prompt_template_content
    # --- End conditional visual instructions ---

    # Format the loaded template with the gathered data
    format_msg = "Formatting final RAG prompt..."
    send_progress(progress_callback, "status", {"message": format_msg})
    send_progress(progress_callback, "log", {"level": "info", "message": format_msg})
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
        err_msg = f"Missing key in prompt template formatting: {e}. Check template file placeholders."
        send_progress(progress_callback, "error", {"message": err_msg})
        send_progress(progress_callback, "phase_end", {"phase": "build_report", "message": "Report building failed (template format error)."})
        return f"Error: Prompt template formatting failed due to missing key: {e}"
    except Exception as e:
        err_msg = f"An unexpected error occurred during prompt formatting: {e}"
        send_progress(progress_callback, "error", {"message": err_msg, "details": traceback.format_exc()})
        send_progress(progress_callback, "phase_end", {"phase": "build_report", "message": "Report building failed (prompt format error)."})
        return f"Error: Failed to format prompt template: {e}"

    # Use resolved RAG model
    rag_call_msg = f"Calling final RAG model ({resolved_settings['rag_model']})..."
    send_progress(progress_callback, "status", {"message": rag_call_msg})
    send_progress(progress_callback, "log", {"level": "info", "message": rag_call_msg})
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
    send_progress(progress_callback, "log", {"level": "info", "message": "Final RAG generation complete."})


    # --- Insert Anchors ---
    if toc_tree_nodes:
        anchor_msg = "Inserting anchors into final report..."
        send_progress(progress_callback, "status", {"message": anchor_msg})
        send_progress(progress_callback, "log", {"level": "info", "message": anchor_msg})
        final_answer = _insert_anchors_into_report(final_answer, toc_tree_nodes)
        send_progress(progress_callback, "log", {"level": "info", "message": "Anchors inserted."})
    # --- End Insert Anchors ---

    send_progress(progress_callback, "phase_end", {"phase": "build_report", "message": "Finished building report content."})
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
    save_msg = "Aggregating results and saving report..."
    send_progress(progress_callback, "phase_start", {"phase": "save_report", "message": save_msg})
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
    end_msg = f"Report saved to: {output_path}"
    send_progress(progress_callback, "phase_end", {"phase": "save_report", "message": end_msg})
    send_progress(progress_callback, "complete", {"report_path": output_path, "message": end_msg}) # Send overall completion
    return output_path
