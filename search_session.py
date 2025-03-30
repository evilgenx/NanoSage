# search_session.py

import os
import uuid
import asyncio
import random

# Import necessary components from updated knowledge_base
from knowledge_base import KnowledgeBase, late_interaction_score, embed_text
from web_search import download_webpages_ddg, parse_html_to_text, group_web_results_by_domain, sanitize_filename
from aggregator import aggregate_results
import llm_utils # Added import
import toc_tree # Added import

#########################################################
# The "SearchSession" class: orchestrate the entire pipeline,
# including optional Monte Carlo subquery sampling, recursive web search,
# TOC tracking, and relevance scoring.
#########################################################

class SearchSession:
    # Changed retrieval_model to embedding_model_name
    # Added selected_gemini_model, selected_openrouter_model, and progress_callback parameters
    def __init__(self, query, config, corpus_dir=None, device="cpu",
                 embedding_model_name="colpali", top_k=3, web_search_enabled=False,
                 personality=None, rag_model="gemma", selected_gemini_model=None,
                 selected_openrouter_model=None, max_depth=1, progress_callback=None):
        """
        :param device: Device for embedding ('cpu', 'cuda', 'Gemini', 'OpenRouter').
        :param embedding_model_name: Specific embedding model ID.
        :param selected_gemini_model: The specific Gemini *generative* model chosen by the user.
        :param selected_openrouter_model: The specific OpenRouter *generative* model chosen by the user.
        :param max_depth: Maximum recursion depth for subquery expansion.
        :param progress_callback: Optional function to call with status updates.
        """
        self.query = query
        self.config = config
        self.corpus_dir = corpus_dir
        self.progress_callback = progress_callback or (lambda msg: None) # Store callback or a dummy lambda
        self.device = device # For embeddings
        self.embedding_model_name = embedding_model_name # Store embedding model name
        self.top_k = top_k
        self.web_search_enabled = web_search_enabled
        self.personality = personality
        self.rag_model = rag_model
        self.selected_gemini_model = selected_gemini_model # Store the selected Gemini model
        self.selected_openrouter_model = selected_openrouter_model # Store the selected OpenRouter model
        self.max_depth = max_depth

        self.query_id = str(uuid.uuid4())[:8]
        self.base_result_dir = os.path.join(self.config.get("results_base_dir", "results"), self.query_id)
        os.makedirs(self.base_result_dir, exist_ok=True)

        self.progress_callback(f"Initializing SearchSession (ID: {self.query_id})...")
        print(f"[INFO] Initializing SearchSession for query_id={self.query_id}")

        # Enhance the query via chain-of-thought
        self.progress_callback("Enhancing query using chain-of-thought...")
        self.enhanced_query = llm_utils.chain_of_thought_query_enhancement(
            self.query,
            personality=self.personality,
            rag_model=self.rag_model,
            selected_gemini_model=self.selected_gemini_model, # Pass Gemini model
            selected_openrouter_model=self.selected_openrouter_model # Pass OpenRouter model
        )
        self.progress_callback(f"Enhanced Query: {self.enhanced_query}")

        # Initialize KnowledgeBase (this now handles model loading/API prep internally)
        self.progress_callback(f"Initializing KnowledgeBase with device='{self.device}', model='{self.embedding_model_name}'...")
        print(f"[INFO] Initializing KnowledgeBase with device='{self.device}', model='{self.embedding_model_name}'")
        self.kb = KnowledgeBase(
            device=self.device,
            embedding_model_name=self.embedding_model_name,
            progress_callback=self.progress_callback
        )
        # Store model_type from KB for convenience in embedding calls later
        self.model_type = self.kb.model_type
        self.model = self.kb.model # Might be None for API
        self.processor = self.kb.processor # Might be None for API

        # Compute the overall enhanced query embedding once *after* KB is initialized.
        self.progress_callback("Computing embedding for enhanced query...")
        print("[INFO] Computing embedding for enhanced query...")
        self.enhanced_query_embedding = embed_text(
            text=self.enhanced_query,
            model=self.model,
            processor=self.processor,
            model_type=self.model_type,
            embedding_model_name=self.embedding_model_name,
            device=self.device
        )
        if self.enhanced_query_embedding is None:
             # Handle error - maybe raise an exception or log a critical error
             self.progress_callback("[CRITICAL] Failed to compute initial query embedding. Aborting.")
             raise ValueError("Failed to compute initial query embedding.")


        # Build local corpus if directory is provided
        if self.corpus_dir:
            self.kb.build_from_directory(self.corpus_dir)
        else:
            self.progress_callback("No local corpus directory specified.")

        # Placeholders for web search results and TOC tree.
        self.web_results = []
        self.grouped_web_results = {}
        self.local_results = []
        self.toc_tree = []  # List of toc_tree.TOCNode objects for the initial subqueries

    async def run_session(self):
        """
        Main entry point: perform recursive web search (if enabled) and then local retrieval.
        """
        self.progress_callback(f"Starting search session (Max Depth: {self.max_depth})...")
        print(f"[INFO] Starting session with query_id={self.query_id}, max_depth={self.max_depth}")
        plain_enhanced_query = llm_utils.clean_search_query(self.enhanced_query) # Use llm_utils

        # 1) Generate subqueries from the enhanced query
        self.progress_callback("Generating initial subqueries...")
        initial_subqueries = llm_utils.split_query(plain_enhanced_query, max_len=self.config.get("max_query_length", 200)) # Use llm_utils
        self.progress_callback(f"Generated {len(initial_subqueries)} initial subqueries.")
        print(f"[INFO] Generated {len(initial_subqueries)} initial subqueries from the enhanced query.")

        # 2) Optionally do a Monte Carlo approach to sample subqueries
        if self.config.get("monte_carlo_search", True):
            self.progress_callback("Performing Monte Carlo subquery sampling...")
            print("[INFO] Using Monte Carlo approach to sample subqueries.")
            initial_subqueries = self.perform_monte_carlo_subqueries(plain_enhanced_query, initial_subqueries)
            self.progress_callback(f"Selected {len(initial_subqueries)} subqueries via Monte Carlo.")

        # 3) If web search is enabled and max_depth >= 1, do the recursive expansion
        if self.web_search_enabled and self.max_depth >= 1:
            self.progress_callback("Starting recursive web search...")
            web_results, web_entries, grouped, toc_nodes = await self.perform_recursive_web_searches(initial_subqueries, current_depth=1)
            self.web_results = web_results
            self.grouped_web_results = grouped
            self.toc_tree = toc_nodes
            # Add new entries to the knowledge base
            self.progress_callback(f"Adding {len(web_entries)} web entries to knowledge base...")
            # KB's add_documents handles ensuring embeddings are tensors on CPU
            self.kb.add_documents(web_entries)
            self.progress_callback("Finished recursive web search.")
        else:
            self.progress_callback("Web search disabled or max_depth < 1, skipping.")
            print("[INFO] Web search is disabled or max_depth < 1, skipping web expansion.")

        # 4) Local retrieval (uses the KB's search method)
        self.progress_callback(f"Retrieving top {self.top_k} documents from knowledge base...")
        print(f"[INFO] Retrieving top {self.top_k} documents for final answer.")
        # KB search method now handles embedding the query internally
        self.local_results = self.kb.search(self.enhanced_query, top_k=self.top_k)
        self.progress_callback(f"Retrieved {len(self.local_results)} documents from knowledge base.")

        # 5) Summaries and final RAG generation
        self.progress_callback("Summarizing web results...")
        summarized_web = self._summarize_web_results(self.web_results)
        self.progress_callback("Summarizing local results...")
        summarized_local = self._summarize_local_results(self.local_results)
        self.progress_callback("Building final report using RAG...")
        final_answer = self._build_final_answer(summarized_web, summarized_local)
        self.progress_callback("Finished building final report.")
        print("[INFO] Finished building final advanced report.")
        return final_answer

    def perform_monte_carlo_subqueries(self, parent_query, subqueries):
        """
        Simple Monte Carlo approach:
          1) Embed each subquery and compute a relevance score against the main query embedding.
          2) Weighted random selection of a subset based on relevance scores.
         """
        max_subqs = self.config.get("monte_carlo_samples", 3)
        self.progress_callback(f"Monte Carlo: Scoring {len(subqueries)} subqueries...")
        print(f"[DEBUG] Monte Carlo: randomly picking up to {max_subqs} subqueries from {len(subqueries)} total.")
        scored_subqs = []
        for i, sq in enumerate(subqueries):
            sq_clean = llm_utils.clean_search_query(sq) # Use llm_utils
            if not sq_clean:
                continue
            # Embed the subquery using the unified embed_text function
            node_emb = embed_text(
                text=sq_clean,
                model=self.model,
                processor=self.processor,
                model_type=self.model_type,
                embedding_model_name=self.embedding_model_name,
                device=self.device
            )
            if node_emb is None:
                 print(f"[WARN] MC: Failed to embed subquery '{sq_clean[:30]}...'. Skipping.")
                 continue
            # Score against the pre-computed enhanced query embedding
            score = late_interaction_score(self.enhanced_query_embedding, node_emb)
            scored_subqs.append((sq_clean, score))

        if not scored_subqs:
            self.progress_callback("Monte Carlo: No valid subqueries found or embedded.")
            print("[WARN] No valid subqueries found/embedded for Monte Carlo. Returning original list.")
            return subqueries # Return original if scoring failed

        # Weighted random choice
        self.progress_callback(f"Monte Carlo: Selecting up to {max_subqs} subqueries...")
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
        self.progress_callback(f"Monte Carlo: Selected {len(chosen_sqs)} subqueries.")
        print(f"[DEBUG] Monte Carlo selected: {chosen_sqs}")
        return chosen_sqs

    async def perform_recursive_web_searches(self, subqueries, current_depth=1):
        """
        Recursively perform web searches for each subquery up to self.max_depth.
        Returns:
          aggregated_web_results, aggregated_corpus_entries, grouped_results, toc_nodes
        """
        aggregated_web_results = []
        aggregated_corpus_entries = []
        toc_nodes = []
        min_relevance = self.config.get("min_relevance", 0.5)

        for sq in subqueries:
            sq_clean = llm_utils.clean_search_query(sq) # Use llm_utils
            if not sq_clean:
                continue

            # Create a TOC node
            toc_node = toc_tree.TOCNode(query_text=sq_clean, depth=current_depth) # Use toc_tree

            # Embed subquery for relevance check
            node_embedding = embed_text(
                text=sq_clean,
                model=self.model,
                processor=self.processor,
                model_type=self.model_type,
                embedding_model_name=self.embedding_model_name,
                device=self.device
            )
            if node_embedding is None:
                 print(f"[WARN] Failed to embed subquery '{sq_clean[:50]}...' for relevance check. Skipping branch.")
                 continue

            relevance = late_interaction_score(self.enhanced_query_embedding, node_embedding)
            toc_node.relevance_score = relevance

            if relevance < min_relevance:
                self.progress_callback(f"Skipping branch (low relevance {relevance:.2f}): '{sq_clean[:50]}...'")
                print(f"[INFO] Skipping branch '{sq_clean}' due to low relevance ({relevance:.2f} < {min_relevance}).")
                continue

            # Create subdirectory
            safe_subquery = sanitize_filename(sq_clean)[:30]
            subquery_dir = os.path.join(self.base_result_dir, f"web_{safe_subquery}")
            os.makedirs(subquery_dir, exist_ok=True)
            self.progress_callback(f"Searching web (Depth {current_depth}, Rel: {relevance:.2f}): '{sq_clean[:50]}...'")
            print(f"[DEBUG] Searching web for subquery '{sq_clean}' at depth={current_depth}...")

            # Pass the progress_callback to the download function
            pages = await download_webpages_ddg(
                sq_clean,
                limit=self.config.get("web_search_limit", 5),
                output_dir=subquery_dir,
                progress_callback=self.progress_callback # Pass the callback here
            )
            self.progress_callback(f"Downloaded {len(pages)} pages for '{sq_clean[:50]}...'")
            branch_web_results = []
            branch_corpus_entries = []
            for i, page in enumerate(pages):
                if not page:
                    continue
                file_path = page.get("file_path")
                url = page.get("url")
                if not file_path or not url:
                    continue

                raw_text = parse_html_to_text(file_path)
                if not raw_text or not raw_text.strip():
                    print(f"[INFO] Skipping empty page content from {url}")
                    continue

                snippet = raw_text[:100].replace('\n', ' ') + "..."
                limited_text = raw_text[:2048] # Limit text length for embedding if needed

                # Embed the web page content
                emb = embed_text(
                    text=limited_text,
                    model=self.model,
                    processor=self.processor,
                    model_type=self.model_type,
                    embedding_model_name=self.embedding_model_name,
                    device=self.device
                )

                if emb is not None:
                    entry = {
                        "embedding": emb, # embed_text now returns tensor on CPU
                        "metadata": {
                            "file_path": file_path,
                            "type": "webhtml",
                            "snippet": snippet,
                            "url": url
                        }
                    }
                    branch_corpus_entries.append(entry)
                    branch_web_results.append({"url": url, "snippet": snippet})
                else:
                    self.progress_callback(f"[WARN] Failed to embed page content from '{url}'. Skipping.")
                    print(f"[WARN] Failed to embed page content from '{url}'. Skipping.")


            # Summarize
            self.progress_callback(f"Summarizing {len(branch_web_results)} web results for '{sq_clean[:50]}...'")
            branch_snippets = " ".join([r.get("snippet", "") for r in branch_web_results])
            # Pass all model params to summarize_text
            toc_node.summary = llm_utils.summarize_text(
                branch_snippets,
                personality=self.personality,
                rag_model=self.rag_model,
                selected_gemini_model=self.selected_gemini_model,
                selected_openrouter_model=self.selected_openrouter_model
            )
            toc_node.web_results = branch_web_results
            # Note: We don't store corpus entries directly in TOC node anymore, they go straight to KB

            additional_subqueries = []
            if current_depth < self.max_depth:
                self.progress_callback(f"Generating potential sub-subqueries for '{sq_clean[:50]}...'")
                # Use the llm_utils function directly, passing all model params
                additional_query = llm_utils.chain_of_thought_query_enhancement(
                    sq_clean,
                    personality=self.personality,
                    rag_model=self.rag_model,
                    selected_gemini_model=self.selected_gemini_model,
                    selected_openrouter_model=self.selected_openrouter_model
                )
                if additional_query and additional_query != sq_clean:
                    additional_subqueries = llm_utils.split_query(additional_query, max_len=self.config.get("max_query_length", 200)) # Use llm_utils
                    self.progress_callback(f"Generated {len(additional_subqueries)} sub-subqueries.")

            if additional_subqueries:
                # Recursive call returns results, entries, grouped, and child nodes
                deeper_web_results, deeper_corpus_entries, _, deeper_toc_nodes = await self.perform_recursive_web_searches(additional_subqueries, current_depth=current_depth+1)
                # Extend the current branch's results and entries
                branch_web_results.extend(deeper_web_results)
                branch_corpus_entries.extend(deeper_corpus_entries)
                # Add child nodes to the current TOC node
                for child_node in deeper_toc_nodes:
                    toc_node.add_child(child_node)

            # Aggregate results from this branch
            aggregated_web_results.extend(branch_web_results)
            aggregated_corpus_entries.extend(branch_corpus_entries)
            toc_nodes.append(toc_node)

        # Grouping happens once after all recursion is done
        grouped = group_web_results_by_domain(
            [{"url": r["url"], "file_path": e["metadata"]["file_path"], "content_type": e["metadata"].get("type", "")} # Use type from metadata
             for r, e in zip(aggregated_web_results, aggregated_corpus_entries) if "metadata" in e and "file_path" in e["metadata"]]
        )
        return aggregated_web_results, aggregated_corpus_entries, grouped, toc_nodes

    def _summarize_web_results(self, web_results):
        lines = []
        reference_urls = []
        self.progress_callback(f"Preparing {len(web_results)} web results for summarization...")
        for w in web_results:
            url = w.get('url')
            snippet = w.get('snippet')
            lines.append(f"URL: {url} - snippet: {snippet}")
            if url: # Only add valid URLs
                 reference_urls.append(url)
        text = "\n".join(lines)
        # We'll store reference URLs in self._reference_links for final prompt
        self._reference_links = sorted(list(set(reference_urls)))  # unique and sorted
        if not text.strip():
             self.progress_callback("No web results to summarize.")
             return "No web results found or summarized."
        self.progress_callback("Calling LLM to summarize web results...")
        # Pass all model params to summarize_text
        summary = llm_utils.summarize_text(
            text,
            personality=self.personality,
            rag_model=self.rag_model,
            selected_gemini_model=self.selected_gemini_model,
            selected_openrouter_model=self.selected_openrouter_model
        )
        self.progress_callback("Finished summarizing web results.")
        return summary

    def _summarize_local_results(self, local_results):
        lines = []
        self.progress_callback(f"Preparing {len(local_results)} local results for summarization...")
        for doc in local_results:
            meta = doc.get('metadata', {})
            file_path = meta.get('file_path')
            snippet = meta.get('snippet', '')
            lines.append(f"File: {file_path} snippet: {snippet}")
        text = "\n".join(lines)
        if not text.strip():
             self.progress_callback("No local results to summarize.")
             return "No local documents found or summarized."
        self.progress_callback("Calling LLM to summarize local results...")
        # Pass all model params to summarize_text
        summary = llm_utils.summarize_text(
            text,
            personality=self.personality,
            rag_model=self.rag_model,
            selected_gemini_model=self.selected_gemini_model,
            selected_openrouter_model=self.selected_openrouter_model
        )
        self.progress_callback("Finished summarizing local results.")
        return summary

    def _build_final_answer(self, summarized_web, summarized_local, previous_results_content="", follow_up_convo=""):
        toc_str = toc_tree.build_toc_string(self.toc_tree) if self.toc_tree else "No Table of Contents generated (Web search might be disabled or yielded no relevant branches)." # Use toc_tree
        # Build a reference links string from _reference_links, if available
        reference_links = ""
        if hasattr(self, "_reference_links") and self._reference_links:
            reference_links = "\n".join(f"- {link}" for link in self._reference_links)
        else:
            reference_links = "No web reference links found."

        # Construct final prompt
        self.progress_callback("Constructing final RAG prompt...")
        aggregation_prompt = f"""
You are an expert research analyst. Using all of the data provided below, produce a comprehensive, advanced report of at least 3000 words on the topic.
The report should include:
1) A detailed Table of Contents (based on the search branches, if available),
2) Multiple sections,
3) In-depth analysis with citations (referencing URLs or local file paths where applicable),
4) A final reference section listing all relevant URLs.

User Query: {self.enhanced_query}

Table of Contents:
{toc_str}

Summarized Web Results:
{summarized_web}

Summarized Local Document Results:
{summarized_local}

Reference Links (unique URLs found):
{reference_links}

Additionally, incorporate any previously gathered information if available.
Provide a thorough discussion covering background, current findings, challenges, and future directions.
Write the report in clear Markdown with section headings, subheadings, and references.

Report:
"""
        self.progress_callback(f"Calling final RAG model ({self.rag_model})...")
        print("[DEBUG] Final RAG prompt constructed. Passing to llm_utils.rag_final_answer()...")
        # Pass all model params to rag_final_answer
        final_answer = llm_utils.rag_final_answer(
            aggregation_prompt,
            rag_model=self.rag_model,
            personality=self.personality,
            selected_gemini_model=self.selected_gemini_model,
            selected_openrouter_model=self.selected_openrouter_model
            # Consider adding progress_callback to rag_final_answer if it's long
        )
        self.progress_callback("Final RAG generation complete.")
        return final_answer

    def save_report(self, final_answer, previous_results=None, follow_up_convo=None):
        self.progress_callback("Aggregating results and saving report...")
        print("[INFO] Saving final report to disk...")
        output_path = aggregate_results(
            self.query_id,
            self.enhanced_query,
            self.web_results, # Pass original web results list
            self.local_results, # Pass original local results list
            final_answer,
            self.config,
            grouped_web_results=self.grouped_web_results, # Pass grouped results
            previous_results=previous_results,
            follow_up_conversation=follow_up_convo
            # Removed toc_nodes=self.toc_tree as it's not an expected argument
        )
        self.progress_callback(f"Report saved to: {output_path}")
        return output_path
