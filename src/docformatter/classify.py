#!/usr/bin/env python
#
#       docformatter.classify.py is part of the docformatter project
#
# Copyright (C) 2012-2023 Steven Myint
# Copyright (C) 2023-2025 Doyle "weibullguy" Rowland
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""This module provides docformatter's classification functions."""

# Standard Library Imports
import re
import sys
import tokenize
from dataclasses import dataclass
from tokenize import TokenInfo

# docformatter Package Imports
from docformatter.constants import MAX_PYTHON_VERSION, QUOTE_TYPES

PY312 = (sys.version_info[0], sys.version_info[1]) > MAX_PYTHON_VERSION

_BLOCKING_KEYWORDS = frozenset({"assert", "return", "raise", "yield", "del"})

_DEFINITION_KEYWORDS = frozenset({"def", "async", "class"})


@dataclass
class _DocstringContext:
    """Track the parsing state for docstring detection.

    Attributes
    ----------
    scope : str
        Current scope: "module", "class", or "function".
    first_statement_seen : bool
        False until the first statement is encountered in the current scope.
    last_was_assignment : bool
        True if the last processed token was an assignment operator.
    anchor_index : int | None
        Index of the class/def/async/assignment token that anchors this scope.
    in_decorator : bool
        True if currently processing a decorator (between @ and def/class).
    assignment_anchor_index : int | None
        Index of the NAME token before the most recent `=` for attribute docstrings.
    in_signature : bool
        True while inside a function/class definition signature (before body colon).
    """

    scope: str = "module"
    first_statement_seen: bool = False
    last_was_assignment: bool = False
    anchor_index: int | None = None
    in_decorator: bool = False
    assignment_anchor_index: int | None = None
    in_signature: bool = False


def _is_docstring_prefix(token_string: str) -> bool:
    """Return True if the token string starts with a valid docstring quote.

    Parameters
    ----------
    token_string : str
        The string token to check.

    Returns
    -------
    bool
        True if the token starts with a valid triple-quote prefix.
    """
    return any(token_string.startswith(quote) for quote in QUOTE_TYPES)


def _is_scope_token(token: TokenInfo) -> bool:
    """Return True if token is a scope-defining keyword (class, def, async)."""
    return token.type == tokenize.NAME and token.string in _DEFINITION_KEYWORDS


def _is_assignment_token(token: TokenInfo) -> bool:
    """Return True if token is an assignment operator (= or :)."""
    return token.type == tokenize.OP and token.string in ("=", ":")


def _classify_docstring_single_pass(  # noqa: PLR0911
    token: TokenInfo,
    token_index: int,
    context_stack: list[_DocstringContext],
) -> tuple[str | None, int | None]:
    """Classify a STRING token as a docstring in a single pass.

    Parameters
    ----------
    token : TokenInfo
        The STRING token to classify.
    token_index : int
        Index of the token in the token list.
    context_stack : list[_DocstringContext]
        Current context stack tracking scope and state.

    Returns
    -------
    tuple[str | None, int | None]
        (docstring_type, anchor_index) or (None, None) if not a docstring.
    """
    current_ctx = context_stack[-1]

    if not _is_docstring_prefix(token.string):
        return None, None

    if current_ctx.last_was_assignment:
        current_ctx.last_was_assignment = False
        if " = " in token.line:
            return None, None
        anchor_idx = current_ctx.assignment_anchor_index
        if anchor_idx is not None:
            return "attribute", anchor_idx
        return None, None

    if current_ctx.scope == "module" and not current_ctx.first_statement_seen:
        current_ctx.first_statement_seen = True
        current_ctx.last_was_assignment = False
        return "module", 0

    if (
        current_ctx.scope in ("class", "function")
        and not current_ctx.first_statement_seen
        and current_ctx.anchor_index is not None
    ):
        current_ctx.first_statement_seen = True
        return current_ctx.scope, current_ctx.anchor_index

    return None, None


def _handle_definition_token(
    token: TokenInfo,
    token_index: int,
    context_stack: list[_DocstringContext],
) -> None:
    """Handle a class/def/async definition token.

    Parameters
    ----------
    token : TokenInfo
        The definition keyword token.
    token_index : int
        Index of the token in the token list.
    context_stack : list[_DocstringContext]
        Context stack to update.
    """
    current_ctx = context_stack[-1]
    current_ctx.in_decorator = False

    if token.string == "async":
        return

    new_ctx = _DocstringContext(
        scope="class" if token.string == "class" else "function",
        anchor_index=token_index,
        in_decorator=False,
        in_signature=True,
    )
    context_stack.append(new_ctx)


def _handle_assignment_token(
    token: TokenInfo,
    prev_name_index: int | None,
    paren_depth: int,
    just_saw_definition: bool,
    context_stack: list[_DocstringContext],
) -> tuple[bool, bool]:
    """Handle an assignment operator (= or :) token.

    Parameters
    ----------
    token : TokenInfo
        The assignment operator token.
    prev_name_index : int | None
        Index of the previous NAME token.
    paren_depth : int
        Current parenthesis nesting depth.
    just_saw_definition : bool
        Whether we just saw a definition keyword.
    context_stack : list[_DocstringContext]
        Context stack to update.

    Returns
    -------
    tuple[bool, bool]
        Updated (just_saw_definition, continue_flag).
    """
    current_ctx = context_stack[-1]

    if token.string == "=":
        current_ctx.last_was_assignment = True
        current_ctx.assignment_anchor_index = prev_name_index
        return False, True

    if token.string == ":":
        if current_ctx.in_signature:
            if paren_depth == 0:
                current_ctx.in_signature = False
            return False, True
        if just_saw_definition:
            return False, True
        if (
            prev_name_index is not None
            and paren_depth == 0
            and not current_ctx.in_decorator
        ):
            current_ctx.last_was_assignment = True
            current_ctx.assignment_anchor_index = prev_name_index
        return False, True

    return just_saw_definition, False


def _handle_scope_exit(
    token: TokenInfo,
    context_stack: list[_DocstringContext],
) -> None:
    """Handle DEDENT token for scope exit.

    Parameters
    ----------
    token : TokenInfo
        The DEDENT token.
    context_stack : list[_DocstringContext]
        Context stack to update.
    """
    if token.type != tokenize.DEDENT:
        return
    if len(context_stack) > 1:
        context_stack.pop()
        if context_stack:
            context_stack[-1].last_was_assignment = False
            context_stack[-1].first_statement_seen = True


def _deduplicate_docstrings(
    docstring_blocks: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Remove adjacent docstrings with the same anchor index.

    Parameters
    ----------
    docstring_blocks : list[tuple[int, int, str]]
        List of docstring blocks to deduplicate.

    Returns
    -------
    list[tuple[int, int, str]]
        Deduplicated list of docstring blocks.
    """
    i = 1
    while i < len(docstring_blocks):
        if docstring_blocks[i][0] == docstring_blocks[i - 1][0]:
            docstring_blocks.pop(i)
        i += 1
    return docstring_blocks


def _do_find_docstring_blocks_single_pass(  # noqa: PLR0912, PLR0915
    tokens: list[TokenInfo],
) -> list[tuple[int, int, str]]:
    """Identify all docstring blocks using a single-pass forward approach.

    This function uses a context stack to track nesting levels and determine
    docstring types without backward scanning.

    Parameters
    ----------
    tokens : list[TokenInfo]
        A list of tokenized Python source code.

    Returns
    -------
    list[tuple[int, int, str]]
        A list of tuples representing each docstring block. Each tuple contains:
            - anchor_index (int): Index of the anchor token.
            - string_index (int): Index of the docstring token.
            - docstring_type (str): One of "module", "class", "function", or
              "attribute".
    """
    docstring_blocks: list[tuple[int, int, str]] = []
    context_stack: list[_DocstringContext] = [_DocstringContext(scope="module")]
    prev_name_index: int | None = None
    paren_depth: int = 0
    bracket_depth: int = 0
    brace_depth: int = 0
    just_saw_definition: bool = False

    for i, token in enumerate(tokens):
        current_ctx = context_stack[-1]

        if token.type in (tokenize.ENCODING, tokenize.COMMENT):
            continue

        if token.type == tokenize.OP and token.string == "@":
            current_ctx.in_decorator = True
            prev_name_index = None
            continue

        if _is_scope_token(token):
            current_ctx.in_decorator = False
            _handle_definition_token(token, i, context_stack)
            prev_name_index = None
            just_saw_definition = True
            continue

        if token.type == tokenize.NAME and token.string in ("import", "from"):
            current_ctx.first_statement_seen = True
            current_ctx.last_was_assignment = False
            just_saw_definition = False
            continue

        if token.type == tokenize.NAME:
            prev_name_index = i

        if _is_assignment_token(token):
            just_saw_definition, cont = _handle_assignment_token(
                token, prev_name_index, paren_depth, just_saw_definition, context_stack
            )
            if cont:
                continue

        if token.type == tokenize.OP and token.string == "(":
            paren_depth += 1
            continue

        if token.type == tokenize.OP and token.string == ")":
            paren_depth -= 1
            continue

        if token.type == tokenize.OP and token.string == "[":
            bracket_depth += 1
            continue

        if token.type == tokenize.OP and token.string == "]":
            bracket_depth -= 1
            continue

        if token.type == tokenize.OP and token.string == "{":
            brace_depth += 1
            continue

        if token.type == tokenize.OP and token.string == "}":
            brace_depth -= 1
            continue

        if token.type == tokenize.DEDENT:
            _handle_scope_exit(token, context_stack)
            continue

        if token.type == tokenize.NAME and token.string in _BLOCKING_KEYWORDS:
            current_ctx.first_statement_seen = True
            current_ctx.last_was_assignment = False
            prev_name_index = None
            continue

        if token.type == tokenize.STRING:
            result = _classify_docstring_single_pass(token, i, context_stack)
            if result[0] is not None:
                docstring_blocks.append((result[1], i, result[0]))

        if token.type in (tokenize.NEWLINE, tokenize.NL):
            if current_ctx.last_was_assignment:
                current_ctx.first_statement_seen = True

    return _deduplicate_docstrings(docstring_blocks)


def do_find_docstring_blocks(tokens: list[TokenInfo]) -> list[tuple[int, int, str]]:
    """Identify all docstring blocks and their anchor points.

    Parameters
    ----------
    tokens : list[TokenInfo]
        A list of tokenized Python source code.

    Returns
    -------
    list[tuple[int, int, str]]
        A list of tuples representing each docstring block.  Each tuple contains:
            - anchor_index (int): Index of the anchor (class, def, async def, or
              assignment).
            - string_index (int): Index of the docstring token.
            - docstring_type (str): One of "module", "class", "function", or
              "attribute".
    """
    return _do_find_docstring_blocks_single_pass(tokens)


def is_closing_quotes(token: TokenInfo, prev_token: TokenInfo) -> bool:
    """Determine if token is a closing quote for a docstring.

    Parameters
    ----------
    token : TokenInfo
        The token to check.
    prev_token : TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is a closing quote for a docstring, False otherwise.
    """
    _offset = prev_token.line.split("\n")[-1]
    if prev_token.line.endswith("\n"):
        _offset = prev_token.line.split("\n")[-2]

    return (
        token.line.strip() == '"""'
        and token.type == tokenize.NEWLINE
        or token.line == _offset
    )


def is_code_line(token: TokenInfo) -> bool:
    """Determine if token is a line of code.

    Parameters
    ----------
    token : TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is a code line, False otherwise.
    """
    return (token.type == tokenize.NAME or token.string == "...") and not (
        token.line.strip().startswith("def ")
        or token.line.strip().startswith("async ")
        or token.line.strip().startswith("class ")
    )


def is_definition_line(token: TokenInfo) -> bool:
    """Determine if token is a class or function/method definition line.

    Parameters
    ----------
    token : TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is a definition line, False otherwise.
    """
    return token.type == tokenize.NAME and (
        token.line.startswith("def ")
        or token.line.startswith("async ")
        or token.line.startswith("class ")
    )


def is_f_string(token: TokenInfo, prev_token: TokenInfo) -> bool:
    """Determine if token is an f-string.

    Parameters
    ----------
    token : TokenInfo
        The token to check.
    prev_token : TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is an f-string, False otherwise.
    """
    if PY312:
        return tokenize.FSTRING_MIDDLE in [token.type, prev_token.type]
    return any(
        [
            token.string.startswith('f"""'),
            prev_token.string.startswith('f"""'),
            token.string.startswith("f'''"),
            prev_token.string.startswith("f'''"),
        ]
    )


def is_inline_comment(token: TokenInfo) -> bool:
    """Determine if token is an inline comment.

    Parameters
    ----------
    token : TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is an inline comment, False otherwise.
    """
    return token.line.strip().startswith('"""') and token.string.startswith("#")


def is_line_following_indent(
    token: TokenInfo,
    prev_token: TokenInfo,
) -> bool:
    """Determine if token is a line that follows an indent.

    Parameters
    ----------
    token : TokenInfo
        The token to check.
    prev_token : TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is a line that follows an indent, False otherwise.
    """
    return prev_token.type == tokenize.INDENT and prev_token.line in token.line


def is_nested_definition_line(token: TokenInfo) -> bool:
    """Determine if token is a nested class or function/method definition line.

    Parameters
    ----------
    token : TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is a nested definition line, False otherwise.
    """
    return re.match(r"^ {4,}(async|class|def) ", token.line) is not None


def is_newline_continuation(
    token: TokenInfo,
    prev_token: TokenInfo,
) -> bool:
    """Determine if token is a continuation of a previous line.

    Parameters
    ----------
    token : TokenInfo
        The token to check.
    prev_token : TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is a continuation of a previous line, False otherwise.
    """
    return (
        token.type in (tokenize.NEWLINE, tokenize.NL)
        and token.line.strip() in prev_token.line.strip()
        and token.line not in {"\n", "\r\n"}
    )


def is_string_variable(
    token: TokenInfo,
    prev_token: TokenInfo,
) -> bool:
    """Determine if token is a string variable assignment.

    Parameters
    ----------
    token : TokenInfo
        The token to check.
    prev_token : TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is a string variable assignment, False otherwise.
    """
    # TODO: The AWAIT token is removed in Python 3.13 and later.  Only Python 3.9
    # seems to generate the AWAIT token, so we can safely remove the check for it when
    # support for Python 3.9 is dropped in April 2026.
    if sys.version_info <= (3, 12):
        _token_types = (tokenize.AWAIT, tokenize.OP)
    else:
        _token_types = (tokenize.OP,)

    return prev_token.type in _token_types and (
        '= """' in token.line or token.line in prev_token.line
    )


def is_docstring_at_end_of_file(tokens: list[TokenInfo], index: int) -> bool:
    """Determine if the docstring is at the end of the file."""
    for i in range(index + 1, len(tokens)):
        tok = tokens[i]
        if tok.type not in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        ):
            return False

    return True
