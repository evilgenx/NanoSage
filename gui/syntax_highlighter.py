#!/usr/bin/env python3
# gui/syntax_highlighter.py

import logging
from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt6.QtCore import QRegularExpression

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import Formatter
    from pygments.styles import get_style_by_name
    from pygments.token import Token, STANDARD_TYPES
    PYGMENTS_INSTALLED = True
except ImportError:
    PYGMENTS_INSTALLED = False
    logging.warning("Pygments library not found. Syntax highlighting will be disabled.")

# --- Pygments Formatter for QSyntaxHighlighter ---
# This custom formatter translates Pygments tokens into QTextCharFormats

if PYGMENTS_INSTALLED:
    class QtFormatter(Formatter):
        def __init__(self, style_name='default', **options):
            super().__init__(**options)
            self.style_name = style_name
            self.formats = {}
            try:
                style = get_style_by_name(style_name)
                # Create QTextCharFormat for each token type in the style
                for token, style_def in style:
                    fmt = QTextCharFormat()
                    if style_def['color']:
                        fmt.setForeground(QColor(f"#{style_def['color']}"))
                    if style_def['bgcolor']:
                        fmt.setBackground(QColor(f"#{style_def['bgcolor']}"))
                    if style_def['bold']:
                        fmt.setFontWeight(QFont.Weight.Bold)
                    if style_def['italic']:
                        fmt.setFontItalic(True)
                    if style_def['underline']:
                        fmt.setFontUnderline(True)
                    self.formats[token] = fmt
            except Exception as e:
                logging.error(f"Error initializing Pygments style '{style_name}': {e}")
                # If style fails, we might want to reconsider disabling Pygments here too,
                # but the outer check should handle the main import error case.
                # For now, keep the class defined but formats might be empty/incomplete.

        def format(self, tokensource, outfile):
            # This method is required by Formatter, but we use self.formats directly
            # We'll iterate through tokens and apply formats in highlightBlock
            pass
else:
    # Define a dummy placeholder class if Pygments is missing
    # This prevents NameError during MarkdownSyntaxHighlighter initialization
    class QtFormatter:
        def __init__(self, *args, **kwargs):
            self.formats = {} # Provide the expected attribute

# --- Markdown Syntax Highlighter ---

class MarkdownSyntaxHighlighter(QSyntaxHighlighter):
    # States for block state machine
    STATE_NORMAL = 0
    STATE_CODE_BLOCK = 1

    def __init__(self, parent, style_name='default'):
        super().__init__(parent)
        self.formatter = None
        self.current_lexer = None
        self.code_block_regex_start = QRegularExpression(r"^```(\w+)?\s*$") # Start: ```lang or ```
        self.code_block_regex_end = QRegularExpression(r"^```\s*$")         # End: ```

        if PYGMENTS_INSTALLED:
            try:
                self.formatter = QtFormatter(style_name=style_name)
            except Exception as e:
                logging.error(f"Failed to initialize QtFormatter: {e}")
                # Pygments might be installed but style failed, formatter remains None

    def highlightBlock(self, text):
        if not self.formatter: # If Pygments failed or formatter init failed
            return

        # --- State Machine Logic ---
        previous_state = self.previousBlockState()
        current_state = self.STATE_NORMAL

        match_start = self.code_block_regex_start.match(text)
        match_end = self.code_block_regex_end.match(text)

        if previous_state == self.STATE_CODE_BLOCK:
            if match_end.hasMatch():
                # End of code block found
                current_state = self.STATE_NORMAL
                self.current_lexer = None # Reset lexer
                # Format the closing ``` line itself (optional, could leave default)
                # self.setFormat(0, len(text), self.formatter.formats.get(Token.Comment, QTextCharFormat()))
            else:
                # Still inside a code block
                current_state = self.STATE_CODE_BLOCK
                # Apply highlighting using the current lexer
                if self.current_lexer:
                    try:
                        # Highlight the current line (block)
                        tokens = self.current_lexer.get_tokens_unprocessed(text)
                        start_col = 0
                        for index, token_type, token_text in tokens:
                            length = len(token_text)
                            if token_type in self.formatter.formats:
                                self.setFormat(start_col, length, self.formatter.formats[token_type])
                            start_col += length
                    except Exception as e:
                        logging.debug(f"Pygments highlighting error in block: {e}") # Debug level
                        # Fallback: Apply a default format to avoid crashing
                        # self.setFormat(0, len(text), QTextCharFormat())
                # else: apply default format?

        elif match_start.hasMatch():
            # Start of a new code block found
            current_state = self.STATE_CODE_BLOCK
            lang = match_start.captured(1) # Get the captured language name (or None)
            try:
                if lang:
                    self.current_lexer = get_lexer_by_name(lang)
                else:
                    # Try to guess if no language specified (can be slow/inaccurate)
                    # self.current_lexer = guess_lexer(text) # Avoid guessing for now
                    self.current_lexer = get_lexer_by_name("text") # Default to text lexer
            except Exception:
                self.current_lexer = get_lexer_by_name("text") # Fallback to plain text lexer
            # Format the opening ``` line itself (optional)
            # self.setFormat(0, len(text), self.formatter.formats.get(Token.Comment, QTextCharFormat()))
        else:
            # Normal text block, no highlighting applied by this highlighter
            # (Could add basic Markdown rules here later if desired)
            current_state = self.STATE_NORMAL
            self.current_lexer = None

        # Set the state for the *next* block
        self.setCurrentBlockState(current_state)

        # Debugging state transitions
        # print(f"Block: '{text[:20]}...' Prev State: {previous_state}, Curr State: {current_state}")
