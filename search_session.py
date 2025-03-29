# search_session.py

import os
import uuid
import asyncio
import random

from knowledge_base import KnowledgeBase, late_interaction_score, load_corpus_from_dir, load_retrieval_model, embed_text
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
    # Added selected_gemini_model and progress_callback parameters
    def __init__(self, query, config, corpus_dir=None, device="cpu",
                 retrieval_model="colpali", top_k=3, web_search_enabled=False,
                 personality=None, rag_model="gemma", selected_gemini_model=None, max_depth=1,
                 progress_callback=None): # Added progress_callback
        """
        :param selected_gemini_model: The specific Gemini model chosen by the user.
        :param max_depth: Maximum recursion depth for subquery expansion.
        :param progress_callback: Optional function to call with status updates.
        """
        self.query = query
        self.config = config
        self.corpus_dir = corpus_dir
        self.progress_callback = progress_callback or (lambda msg: None) # Store callback or a dummy lambda
        self.device = device
        self.retrieval_model = retrieval_model
        self.top_k = top_k
        self.web_search_enabled = web_search_enabled
        self.personality = personality
        self.rag_model = rag_model
        self.selected_gemini_model = selected_gemini_model # Store the selected model
        self.max_depth = max_depth

        self.query_id = str(uuid.uuid4())[:8]
        self.base_result_dir = os.path.join(self.config.get("results_base_dir", "results"), self.query_id)
        os.makedirs(self.base_result_dir, exist_ok=True)

        self.progress_callback(f"Initializing SearchSession (ID: {self.query_id})...")
        print(f"[INFO] Initializing SearchSession for query_id={self.query_id}")

        # Enhance the query via chain-of-thought
        self.progress_callback("Enhancing query using chain-of-thought...")
        # Use llm_utils function directly here, passing self.selected_gemini_model
        self.enhanced_query = llm_utils.chain_of_thought_query_enhancement(
            self.query,
            personality=self.personality,
            rag_model=self.rag_model,
            selected_gemini_model=self.selected_gemini_model # Pass the model
        )
        self.progress_callback(f"Enhanced Query: {self.enhanced_query}")
        # Fallback handled inside the enhancement function now
        # if not self.enhanced_query:
        #     self.enhanced_query = self.query

        # Load retrieval model.
        self.progress_callback(f"Loading retrieval model ({self.retrieval_model}) on {self.device}...")
        self.model, self.processor, self.model_type = load_retrieval_model(
            model_choice=self.retrieval_model,
            device=self.device
        )

        # Compute the overall enhanced query embedding once.
        self.progress_callback("Computing embedding for enhanced query...")
        print("[INFO] Computing embedding for enhanced query...")
        self.enhanced_query_embedding = embed_text(self.enhanced_query, self.model, self.processor, self.model_type, self.device)

        # Create a knowledge base.
        self.progress_callback("Creating KnowledgeBase...")
        print("[INFO] Creating KnowledgeBase...")
        self.kb = KnowledgeBase(self.model, self.processor, model_type=self.model_type, device=self.device)

        # Load local corpus if available.
        self.corpus = []
        if self.corpus_dir:
            self.progress_callback(f"Loading local documents from {self.corpus_dir}...")
            print(f"[INFO] Loading local documents from {self.corpus_dir}")
            # Pass callback to load_corpus_from_dir if modified to accept it
            local_docs = load_corpus_from_dir(self.corpus_dir, self.model, self.processor, self.device, self.model_type, progress_callback=self.progress_callback)
            self.corpus.extend(local_docs)
            self.progress_callback(f"Loaded {len(local_docs)} local documents.")
        self.kb.add_documents(self.corpus)

        # Placeholders for web search results and TOC tree.
        self.web_results = []
        self.grouped_web_results = {}
        self.local_results = []
        self.toc_tree = []  # List of toc_tree.TOCNode objects for the initial subqueries

    # Removed the instance method _chain_of_thought_query_enhancement as it's now in llm_utils

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
            self.corpus.extend(web_entries)
            self.kb.add_documents(web_entries)
            self.progress_callback("Finished recursive web search.")
        else:
            self.progress_callback("Web search disabled or max_depth < 1, skipping.")
            print("[INFO] Web search is disabled or max_depth < 1, skipping web expansion.")

        # 4) Local retrieval
        self.progress_callback(f"Retrieving top {self.top_k} local documents...")
        print(f"[INFO] Retrieving top {self.top_k} local documents for final answer.")
        self.local_results = self.kb.search(self.enhanced_query, top_k=self.top_k)
        self.progress_callback(f"Retrieved {len(self.local_results)} local documents.")

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
          1) Embed each subquery and compute a relevance score.
          2) Weighted random selection of a subset (k=3) based on relevance scores.
         """
        max_subqs = self.config.get("monte_carlo_samples", 3)
        self.progress_callback(f"Monte Carlo: Scoring {len(subqueries)} subqueries...")
        print(f"[DEBUG] Monte Carlo: randomly picking up to {max_subqs} subqueries from {len(subqueries)} total.")
        scored_subqs = []
        for i, sq in enumerate(subqueries):
            sq_clean = llm_utils.clean_search_query(sq) # Use llm_utils
            if not sq_clean:
                continue
            # self.progress_callback(f"MC Scoring {i+1}/{len(subqueries)}: '{sq_clean[:30]}...'") # Can be too verbose
            node_emb = embed_text(sq_clean, self.model, self.processor, self.model_type, self.device)
            score = late_interaction_score(self.enhanced_query_embedding, node_emb)
            scored_subqs.append((sq_clean, score))

        if not scored_subqs:
            self.progress_callback("Monte Carlo: No valid subqueries found.")
            print("[WARN] No valid subqueries found for Monte Carlo. Returning original list.")
            return subqueries

        # Weighted random choice
        self.progress_callback(f"Monte Carlo: Selecting up to {max_subqs} subqueries...")
        chosen = random.choices(
            population=scored_subqs,
            weights=[s for (_, s) in scored_subqs],
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
            # Relevance
            node_embedding = embed_text(sq_clean, self.model, self.processor, self.model_type, self.device)
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

            pages = await download_webpages_ddg(sq_clean, limit=self.config.get("web_search_limit", 5), output_dir=subquery_dir)
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
                # self.progress_callback(f"Processing page {i+1}/{len(pages)}: {url}") # Can be too verbose
                raw_text = parse_html_to_text(file_path)
                if not raw_text.strip():
                    continue
                snippet = raw_text[:100].replace('\n', ' ') + "..."
                limited_text = raw_text[:2048]
                try:
                    # self.progress_callback(f"Embedding page {i+1}...") # Can be too verbose
                    if self.model_type == "colpali":
                        inputs = self.processor(text=[limited_text], truncation=True, max_length=512, return_tensors="pt").to(self.device)
                        outputs = self.model(**inputs)
                        emb = outputs.embeddings.mean(dim=1).squeeze(0)
                    else:
                        emb = self.model.encode(limited_text, convert_to_tensor=True)
                    entry = {
                        "embedding": emb,
                        "metadata": {
                            "file_path": file_path,
                            "type": "webhtml",
                            "snippet": snippet,
                            "url": url
                        }
                    }
                    branch_corpus_entries.append(entry)
                    branch_web_results.append({"url": url, "snippet": snippet})
                except Exception as e:
                    self.progress_callback(f"[WARN] Error embedding page '{url}': {e}")
                    print(f"[WARN] Error embedding page '{url}': {e}")

            # Summarize
            self.progress_callback(f"Summarizing {len(branch_web_results)} web results for '{sq_clean[:50]}...'")
            branch_snippets = " ".join([r.get("snippet", "") for r in branch_web_results])
            toc_node.summary = llm_utils.summarize_text(branch_snippets, personality=self.personality) # Use llm_utils
            toc_node.web_results = branch_web_results
            toc_node.corpus_entries = branch_corpus_entries

            additional_subqueries = []
            if current_depth < self.max_depth:
                self.progress_callback(f"Generating potential sub-subqueries for '{sq_clean[:50]}...'")
                # Use the llm_utils function directly
                additional_query = llm_utils.chain_of_thought_query_enhancement(
                    sq_clean,
                    personality=self.personality,
                    rag_model=self.rag_model,
                    selected_gemini_model=self.selected_gemini_model # Pass model
                )
                if additional_query and additional_query != sq_clean:
                    additional_subqueries = llm_utils.split_query(additional_query, max_len=self.config.get("max_query_length", 200)) # Use llm_utils
                    self.progress_callback(f"Generated {len(additional_subqueries)} sub-subqueries.")

            if additional_subqueries:
                deeper_web_results, deeper_corpus_entries, _, deeper_toc_nodes = await self.perform_recursive_web_searches(additional_subqueries, current_depth=current_depth+1)
                branch_web_results.extend(deeper_web_results)
                branch_corpus_entries.extend(deeper_corpus_entries)
                for child_node in deeper_toc_nodes:
                    toc_node.add_child(child_node)

            aggregated_web_results.extend(branch_web_results)
            aggregated_corpus_entries.extend(branch_corpus_entries)
            toc_nodes.append(toc_node)

        grouped = group_web_results_by_domain(
            [{"url": r["url"], "file_path": e["metadata"]["file_path"], "content_type": e["metadata"].get("content_type", "")}
             for r, e in zip(aggregated_web_results, aggregated_corpus_entries)]
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
            reference_urls.append(url)
        text = "\n".join(lines)
        # We'll store reference URLs in self._reference_links for final prompt
        self._reference_links = list(set(reference_urls))  # unique
        self.progress_callback("Calling LLM to summarize web results...")
        summary = llm_utils.summarize_text(text, personality=self.personality) # Use llm_utils
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
        self.progress_callback("Calling LLM to summarize local results...")
        summary = llm_utils.summarize_text(text, personality=self.personality) # Use llm_utils
        self.progress_callback("Finished summarizing local results.")
        return summary

    def _build_final_answer(self, summarized_web, summarized_local, previous_results_content="", follow_up_convo=""):
        toc_str = toc_tree.build_toc_string(self.toc_tree) if self.toc_tree else "No TOC available." # Use toc_tree
        # Build a reference links string from _reference_links, if available
        reference_links = ""
        if hasattr(self, "_reference_links"):
            reference_links = "\n".join(f"- {link}" for link in self._reference_links)

        # Construct final prompt
        self.progress_callback("Constructing final RAG prompt...")
        aggregation_prompt = f"""
You are an expert research analyst. Using all of the data provided below, produce a comprehensive, advanced report of at least 3000 words on the topic.
The report should include:
1) A detailed Table of Contents (based on the search branches),
2) Multiple sections,
3) In-depth analysis with citations,
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
        # Pass selected_gemini_model
        final_answer = llm_utils.rag_final_answer( # Use llm_utils
            aggregation_prompt,
            rag_model=self.rag_model,
            personality=self.personality,
            selected_gemini_model=self.selected_gemini_model
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
            self.web_results,
            self.local_results,
            final_answer,
            self.config,
            grouped_web_results=self.grouped_web_results,
            previous_results=previous_results,
            follow_up_conversation=follow_up_convo
        )
        self.progress_callback(f"Report saved to: {output_path}")
        return output_path
