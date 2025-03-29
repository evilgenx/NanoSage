# main.py

import argparse
import asyncio
import yaml
import os
import sys # Import sys for exiting

# Import SearchSession and related functions
from search_session import SearchSession, list_gemini_models

# Conditional GUI imports
GUI_ENABLED = True
try:
    from PyQt6.QtWidgets import QApplication
    from gui import MainWindow
except ImportError:
    GUI_ENABLED = False
    print("[WARN] PyQt6 or gui.py not found. GUI mode disabled.")

def load_config(config_path):
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Multi-step RAG pipeline with depth-limited searching.")
    # Make query not strictly required initially, will check later for CLI mode
    parser.add_argument("--query", type=str, help="Initial user query (required for CLI mode)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--corpus_dir", type=str, default=None, help="Path to local corpus folder")
    parser.add_argument("--device", type=str, default="cpu", help="Device for retrieval model (cpu or cuda)")
    parser.add_argument("--retrieval_model", type=str, choices=["colpali", "all-minilm"], default="colpali")
    parser.add_argument("--top_k", type=int, default=3, help="Number of local docs to retrieve")
    parser.add_argument("--web_search", action="store_true", default=False, help="Enable web search")
    parser.add_argument("--personality", type=str, default=None, help="Optional personality for Gemma (e.g. cheerful)")
    parser.add_argument("--rag_model", type=str, choices=["gemma", "pali", "gemini"], default="gemma", help="Which model to use for final RAG steps (gemma, pali, gemini)")
    parser.add_argument("--max_depth", type=int, default=1, help="Depth limit for subquery expansions")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical user interface") # Add GUI flag
    args = parser.parse_args()

    # --- GUI Mode ---
    if args.gui:
        if not GUI_ENABLED:
             print("[ERROR] GUI dependencies (PyQt6) are not installed or gui.py is missing. Cannot launch GUI.")
             sys.exit(1)
        print("[INFO] Launching NanoSage GUI...")
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())

    # --- CLI Mode (Original Logic) ---
    print("[INFO] Running in Command-Line Interface mode.")

    # Check if query is provided for CLI mode
    if not args.query:
        parser.error("--query is required when not using --gui")

    config = load_config(args.config)

    selected_gemini_model = None
    if args.rag_model == "gemini":
        print("[INFO] Fetching available Gemini models...")
        available_models = list_gemini_models()

        if available_models is None:
            # Error occurred (e.g., API key missing or API call failed)
            print("[ERROR] Could not retrieve Gemini models. Please check API key and network connection.")
            sys.exit(1) # Exit if models can't be listed
        elif not available_models:
            # API call succeeded but no suitable models found
            print("[ERROR] No Gemini models supporting 'generateContent' found.")
            # Optionally, default to a known model or exit
            # For now, let's default to gemini-1.5-flash-latest as a fallback
            print("[WARN] Defaulting to 'models/gemini-1.5-flash-latest'. This might not work.")
            selected_gemini_model = "models/gemini-1.5-flash-latest"
            # Alternatively, exit: sys.exit(1)
        else:
            print("Available Gemini Models:")
            for i, model_name in enumerate(available_models):
                print(f"  {i + 1}: {model_name}")

            while True:
                try:
                    choice = input("Please select a Gemini model number: ")
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_models):
                        selected_gemini_model = available_models[choice_idx]
                        print(f"[INFO] Using Gemini model: {selected_gemini_model}")
                        break
                    else:
                        print(f"[ERROR] Invalid choice. Please enter a number between 1 and {len(available_models)}.")
                except ValueError:
                    print("[ERROR] Invalid input. Please enter a number.")
                except EOFError: # Handle Ctrl+D or unexpected end of input
                     print("\n[ERROR] Input cancelled. Exiting.")
                     sys.exit(1)


    session = SearchSession(
        query=args.query,
        config=config,
        corpus_dir=args.corpus_dir,
        device=args.device,
        retrieval_model=args.retrieval_model,
        top_k=args.top_k,
        web_search_enabled=args.web_search,
        personality=args.personality,
        rag_model=args.rag_model,
        selected_gemini_model=selected_gemini_model, # Pass the selected model
        max_depth=args.max_depth
    )

    loop = asyncio.get_event_loop()
    final_answer = loop.run_until_complete(session.run_session())

    # Save final report
    output_path = session.save_report(final_answer)
    print(f"[INFO] Final report saved to: {output_path}")


if __name__ == "__main__":
    main()
