# llm_providers/utils.py
import re

def extract_final_query(text):
    """Extracts the query following the 'Final Enhanced Query:' marker."""
    marker = "Final Enhanced Query:"
    if marker in text:
        return text.split(marker)[-1].strip()
    # If marker not found, return the whole text stripped,
    # assuming it might be the direct output or an error message.
    return text.strip()

def clean_search_query(query):
    """Removes markdown characters and extra whitespace from a query."""
    # Remove markdown emphasis characters: *, _, `
    query = re.sub(r'[\*\_`]', '', query)
    # Replace multiple whitespace characters with a single space
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

def split_query(query, max_len=200):
    """Splits a long query into smaller sentences, respecting a max length."""
    # Remove quotes which might interfere with sentence splitting or downstream use
    query = query.replace('"', '').replace("'", "")
    # Split primarily by periods, assuming they mark sentence ends
    sentences = query.split('.')
    subqueries = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Basic check to avoid adding sentences that are just punctuation or whitespace
        if not any(c.isalnum() for c in sentence):
            continue

        # Check if adding the next sentence exceeds the max length
        if len(current) + len(sentence) + 1 <= max_len: # +1 for the potential ". "
            current += (". " if current else "") + sentence
        else:
            # If the current subquery isn't empty, add it
            if current:
                subqueries.append(current)
            # Start the new subquery with the current sentence,
            # but only if it doesn't exceed max_len itself.
            if len(sentence) <= max_len:
                current = sentence
            else:
                # If the sentence itself is too long, we might need a more robust
                # splitting strategy (e.g., by words), but for now, we'll just add it
                # as is, potentially exceeding max_len for this single chunk.
                # Or, we could truncate it, but that might lose meaning.
                # Let's add it as is for now.
                # Consider adding a warning if a single sentence exceeds max_len.
                print(f"[WARN] Single sentence exceeds max_len ({max_len}): '{sentence[:50]}...'")
                subqueries.append(sentence)
                current = "" # Reset current as this long sentence forms its own chunk

    # Add the last accumulated subquery if it's not empty
    if current:
        subqueries.append(current)

    # Final filter to remove any potentially empty strings that slipped through
    return [sq for sq in subqueries if sq.strip()]
