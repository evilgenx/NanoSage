import random
from llm_providers.utils import clean_search_query, split_query
from knowledge_base import late_interaction_score # Removed embed_text
from embeddings.base import BaseEmbedder # Added for type hinting

def generate_initial_subqueries(enhanced_query, config):
    """Generates initial subqueries by splitting the enhanced query."""
    plain_enhanced_query = clean_search_query(enhanced_query)
    # Use resolved max_query_length from config (assuming it might be under 'advanced')
    max_query_length = config.get('advanced', {}).get("max_query_length", 200)
    initial_subqueries = split_query(plain_enhanced_query, max_len=max_query_length) # Use split_query from utils
    print(f"[INFO] Generated {len(initial_subqueries)} initial subqueries from the enhanced query.")
    return initial_subqueries

def perform_monte_carlo_subqueries(
    parent_query, # Not used currently, but kept for potential future use
    subqueries,
    config,
    resolved_settings,
    enhanced_query_embedding,
    progress_callback,
    embedder: BaseEmbedder # Added embedder parameter
    # Removed model, processor, model_type
):
    """
    Simple Monte Carlo approach:
      1) Embed each subquery and compute a relevance score against the main query embedding.
      2) Weighted random selection of a subset based on relevance scores.
     """
    max_subqs = config.get("monte_carlo_samples", 3)
    progress_callback(f"Monte Carlo: Scoring {len(subqueries)} subqueries...")
    print(f"[DEBUG] Monte Carlo: randomly picking up to {max_subqs} subqueries from {len(subqueries)} total.")
    scored_subqs = []
    for i, sq in enumerate(subqueries):
        sq_clean = clean_search_query(sq) # Use clean_search_query from utils
        if not sq_clean:
            continue
        # Embed the subquery using the provided embedder instance
        node_emb = embedder.embed(text=sq_clean) # Use embedder.embed()
        if node_emb is None:
            print(f"[WARN] MC: Failed to embed subquery '{sq_clean[:30]}...'. Skipping.")
            continue
        # Score against the pre-computed enhanced query embedding
        score = late_interaction_score(enhanced_query_embedding, node_emb)
        scored_subqs.append((sq_clean, score))

    if not scored_subqs:
        progress_callback("Monte Carlo: No valid subqueries found or embedded.")
        print("[WARN] No valid subqueries found/embedded for Monte Carlo. Returning original list.")
        return subqueries # Return original if scoring failed

    # Weighted random choice
    progress_callback(f"Monte Carlo: Selecting up to {max_subqs} subqueries...")
    # Ensure weights are non-negative
    weights = [max(0, s) for (_, s) in scored_subqs]
    # Avoid division by zero if all weights are zero
    if sum(weights) == 0:
         weights = [1] * len(scored_subqs) # Equal probability if all scores <= 0

    chosen = random.choices(
        population=scored_subqs,
        weights=weights,
        k=min(max_subqs, len(scored_subqs))
    )
    # Return just the chosen subqueries
    chosen_sqs = [ch[0] for ch in chosen]
    progress_callback(f"Monte Carlo: Selected {len(chosen_sqs)} subqueries.")
    print(f"[DEBUG] Monte Carlo selected: {chosen_sqs}")
    return chosen_sqs
