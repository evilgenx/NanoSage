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

    # Construct final prompt
    progress_callback("Constructing final RAG prompt...")
    aggregation_prompt = f"""
You are an expert research analyst. Using all of the data provided below, produce a comprehensive, advanced report of at least 3000 words on the topic.
The report should include:
1) A detailed Table of Contents (based on the search branches, if available),
2) Multiple sections,
3) In-depth analysis with citations (referencing URLs or local file paths where applicable),
4) A final reference section listing all relevant URLs.

User Query: {enhanced_query}

Table of Contents:
{toc_str}

Summarized Web Results:
{summarized_web}

Summarized Local Document Results:
{summarized_local}

Reference Links (unique URLs found):
{reference_links_str}

Additionally, incorporate any previously gathered information if available.
Provide a thorough discussion covering background, current findings, challenges, and future directions.
Write the report in clear Markdown with section headings, subheadings, and references.

Report:
"""
    # Use resolved RAG model
    progress_callback(f"Calling final RAG model ({resolved_settings['rag_model']})...")
    print("[DEBUG] Final RAG prompt constructed. Passing to rag_final_answer()...")
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
        aggregation_prompt,
        llm_config=llm_config_for_rag
        # Consider adding progress_callback to rag_final_answer if it's long
    )
    progress_callback("Final RAG generation complete.")
    return final_answer

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
