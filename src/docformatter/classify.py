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
from tokenize import TokenInfo
from typing import Union

# docformatter Package Imports
from docformatter.constants import MAX_PYTHON_VERSION

PY312 = (sys.version_info[0], sys.version_info[1]) > MAX_PYTHON_VERSION

_SEARCHING = 0
_SAW_COLON = 1

_SKIP_TOKEN_TYPES = frozenset(
    {
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.NUMBER,
    }
)

_BLOCKING_KEYWORDS = frozenset({"assert", "return", "raise", "yield", "del"})

_DEFINITION_KEYWORDS = frozenset({"def", "async", "class"})


def _is_name_after_definition(
    tokens: list[tokenize.TokenInfo],
    name_index: int,
) -> bool:
    """Return True if the NAME at name_index is preceded by a definition keyword."""
    j = name_index - 1
    while j >= 0 and tokens[j].type in (
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.OP,
        tokenize.NUMBER,
        tokenize.STRING,
    ):
        j -= 1
    if j >= 0 and tokens[j].type == tokenize.NAME:
        return tokens[j].string in _DEFINITION_KEYWORDS
    return False


def do_find_docstring_blocks(tokens: list[TokenInfo]) -> list[tuple[int, int, str]]:
    """Identify all docstring blocks and their anchor points.

    Parameters
    ----------
    tokens (list[TokenInfo]):
        A list of tokenized Python source code.

    Returns
    -------
    list[tuple[int, int, str]]:
        A list of tuples representing each docstring block.  Each tuple contains:
            - anchor_index (int): Index of the anchor (class, def, async def, or
              assignment).
            - string_index (int): Index of the docstring token.
            - docstring_type (str): One of "module", "class", "function", or
              "attribute".
    """
    docstring_blocks = []

    for i, token in enumerate(tokens):
        if (
            token.type != tokenize.STRING
            or not (
                token.string.startswith('"""')
                or token.string.startswith('r"""')
                or token.string.startswith('R"""')
                or token.string.startswith('u"""')
                or token.string.startswith('U"""')
                or token.string.startswith("'''")
                or token.string.startswith("r'''")
                or token.string.startswith("R'''")
                or token.string.startswith("u'''")
                or token.string.startswith("U'''")
            )
            or " = " in token.line
        ):
            continue

        if is_module_docstring(tokens, i):
            docstring_blocks.append((0, i, "module"))
            continue

        if is_attribute_docstring(tokens, i):
            anchor_idx = _do_find_anchor_index(tokens, i, target="attribute")
            if anchor_idx is not None:
                docstring_blocks.append((anchor_idx, i, "attribute"))
            continue

        if is_class_docstring(tokens, i):
            anchor_idx = _do_find_anchor_index(tokens, i, target="class")
            if anchor_idx is not None:
                docstring_blocks.append((anchor_idx, i, "class"))
            continue

        if is_function_or_method_docstring(tokens, i):
            anchor_idx = _do_find_anchor_index(tokens, i, target="def")
            if anchor_idx is not None:
                docstring_blocks.append((anchor_idx, i, "function"))
            continue

    # If adjacent docstrings have the same anchor index, remove the second one as
    # there can only be one docstring per anchor.
    i = 1
    while i < len(docstring_blocks):
        if docstring_blocks[i][0] == docstring_blocks[i - 1][0]:
            docstring_blocks.pop(i)
        i += 1

    return docstring_blocks


def _do_find_anchor_index(
    tokens: list[TokenInfo],
    docstring_index: int,
    target: str,
) -> Union[int, None]:
    """Walk backward from a docstring to find the matching anchor.

    The matching anchor would be one of `class`, `def`, `async def`, or an assignment.

    Parameters
    ----------
    tokens (list[TokenInfo]):
        A list of tokenized Python source code.
    docstring_index (int):
        Index of the STRING token representing the docstring.
    target (str):
        One of "class", "def", or "attribute" indicating what to search for.

    Returns
    -------
    int | None:
        Index of the anchor token if found, otherwise None.
    """
    i = docstring_index - 1
    saw_decorator = False

    while i >= 0:
        tok = tokens[i]

        if tok.type == tokenize.OP and tok.string == "@":
            saw_decorator = True

        if target == "class" and tok.type == tokenize.NAME and tok.string == "class":
            return i

        if target == "def" and tok.type == tokenize.NAME and tok.string == "def":
            # Handle @decorator above def
            if saw_decorator:
                while i > 0 and tokens[i - 1].type != tokenize.NEWLINE:
                    i -= 1
            return i

        if target == "attribute":
            if tok.type == tokenize.NAME:
                return i

        i -= 1

    return None


def is_attribute_docstring(  # noqa: PLR0911, PLR0912
    tokens: list[tokenize.TokenInfo],
    index: int,
) -> bool:
    """Return True if the string token is an attribute docstring.

    An attribute docstring is a string that immediately follows an attribute
    assignment or type annotation within a class body.  Valid patterns include:

    - Simple assignment: ``x = 1`` followed by a docstring
    - Annotated assignment: ``x: int = 1`` followed by a docstring
    - Annotation only: ``x: int`` followed by a docstring

    Parameters
    ----------
    tokens : list[TokenInfo]
        A list of tokenized Python source code.
    index : int
        Index of the STRING token to check.

    Returns
    -------
    bool
        True if attribute docstring, False otherwise.
    """
    if index < 2:  # noqa: PLR2004
        return False

    # State machine walks backward from the string token.
    # _SEARCHING: Look for "=" (attribute assignment) or ":" (type annotation).
    # _SAW_COLON: After seeing ":", determine if it's a variable annotation
    #             or part of a function/class definition.
    state = _SEARCHING

    # Tracks whether we've seen a closing paren while in SAW_COLON state.
    # A ")" indicates we're walking through a function signature (e.g.,
    # def foo(x: int):), so a NAME before it would be a parameter, not
    # an attribute.
    saw_paren = False

    i = index - 1
    while i >= 0:
        tok = tokens[i]
        tok_type = tok.type
        tok_string = tok.string

        if state == _SEARCHING:
            if tok_type == tokenize.OP:
                if tok_string == "=":
                    # Found assignment: x = """docstring"""
                    # But if we're inside parentheses, it's a keyword argument
                    _balance = 0
                    for j in range(i):
                        if tokens[j].type == tokenize.OP:
                            if tokens[j].string == "(":
                                _balance += 1
                            elif tokens[j].string == ")":
                                _balance -= 1
                    if _balance > 0:
                        return False
                    return True
                if tok_string == ":":
                    # Possible type annotation: x: int or x: int = value
                    state = _SAW_COLON
            elif tok_type == tokenize.STRING:
                # Another string precedes this one, not an attribute docstring
                return False
            elif tok_type == tokenize.NAME:
                if tok_string in _BLOCKING_KEYWORDS:
                    # String is part of a statement (return, assert, etc.)
                    return False
            elif tok_type not in _SKIP_TOKEN_TYPES:
                pass

        elif state == _SAW_COLON:
            if tok_type == tokenize.OP:
                if tok_string == "=":
                    # Annotated assignment: x: int = """docstring"""
                    # But if we're inside parentheses, it's a keyword argument
                    _balance = 0
                    for j in range(i):
                        if tokens[j].type == tokenize.OP:
                            if tokens[j].string == "(":
                                _balance += 1
                            elif tokens[j].string == ")":
                                _balance -= 1
                    if _balance > 0:
                        return False
                    return True
                if tok_string == ")":
                    # Function signature context: def foo(x: int):
                    # The colon belongs to the function, not an attribute.
                    saw_paren = True
            elif tok_type == tokenize.STRING:
                # Another string precedes this one
                return False
            elif tok_type == tokenize.NAME:
                if tok_string in _DEFINITION_KEYWORDS:
                    # Colon belongs to def/class, not an attribute
                    return False
                if tok_string in _BLOCKING_KEYWORDS:
                    # String is part of a statement (return, assert, etc.)
                    return False
                if _is_name_after_definition(tokens, i):
                    # NAME is the function/class name (e.g., "foo" in "def foo:")
                    return False
                if (
                    i > 0
                    and tokens[i - 1].type == tokenize.OP
                    and tokens[i - 1].string == "->"
                ):
                    # Return type annotation: def foo() -> int:
                    return False
                if not saw_paren:
                    # Variable annotation: x: int (no function signature context)
                    return True
                # If saw_paren is True, this is a parameter annotation
                # like def foo(x: int):, which is not an attribute.
            elif tok_type not in _SKIP_TOKEN_TYPES:
                pass

        i -= 1

    return False


def is_class_docstring(
    tokens: list[tokenize.TokenInfo],
    index: int,
) -> bool:
    """Determine if docstring is a class docstring."""
    # Walk backward to find the most recent `class` keyword before the string,
    # without crossing over a `def`, `async`, or another block
    for i in range(index - 1, -1, -1):
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string == "class":
            return True
        if tok.type == tokenize.NAME and tok.string in ("def", "async"):
            return False  # Hit enclosing function or method first.
        if tok.type == tokenize.OP and tok.string == "=":
            return False  # Hit assignment, not a class docstring.

    return False


def is_closing_quotes(
    token: tokenize.TokenInfo, prev_token: tokenize.TokenInfo
) -> bool:
    """Determine if token is a closing quote for a docstring.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.
    prev_token : tokenize.TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is a closing quote for a docstring, False otherwise.
    """
    _offset = prev_token.line.split("\n")[-1]
    if prev_token.line.endswith("\n"):
        _offset = prev_token.line.split("\n")[-2]

    if (
        token.line.strip() == '"""'
        and token.type == tokenize.NEWLINE
        or token.line == _offset
    ):
        return True

    return False


def is_code_line(token: tokenize.TokenInfo) -> bool:
    """Determine if token is a line of code.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is a code line, False otherwise.
    """
    if (token.type == tokenize.NAME or token.string == "...") and not (
        token.line.strip().startswith("def ")
        or token.line.strip().startswith("async ")
        or token.line.strip().startswith("class ")
    ):
        return True

    return False


def is_definition_line(token: tokenize.TokenInfo) -> bool:
    """Determine if token is a class or function/method definition line.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is a definition line, False otherwise.
    """
    if token.type == tokenize.NAME and (
        token.line.startswith("def ")
        or token.line.startswith("async ")
        or token.line.startswith("class ")
    ):
        return True

    return False


def is_f_string(token: tokenize.TokenInfo, prev_token: tokenize.TokenInfo) -> bool:
    """Determine if token is an f-string.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.
    prev_token : tokenize.TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is an f-string, False otherwise.
    """
    if PY312:
        if tokenize.FSTRING_MIDDLE in [token.type, prev_token.type]:
            return True
    elif any(
        [
            token.string.startswith('f"""'),
            prev_token.string.startswith('f"""'),
            token.string.startswith("f'''"),
            prev_token.string.startswith("f'''"),
        ]
    ):
        return True

    return False


def is_function_or_method_docstring(
    tokens: list[tokenize.TokenInfo],
    index: int,
) -> bool:
    """Determine if docstring is a function or method docstring."""
    for i in range(index - 1, -1, -1):
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string in ("def", "async"):
            return True
        if tok.type == tokenize.NAME and tok.string == "class":
            return False  # hit enclosing class first
        if tok.type == tokenize.NAME and tok.string in (
            "assert",
            "return",
            "raise",
            "yield",
            "del",
        ):
            return False  # string is part of an expression, not a docstring

    return False


def is_inline_comment(token: tokenize.TokenInfo) -> bool:
    """Determine if token is an inline comment.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is an inline comment, False otherwise.
    """
    if token.line.strip().startswith('"""') and token.string.startswith("#"):
        return True
    return False


def is_line_following_indent(
    token: tokenize.TokenInfo,
    prev_token: tokenize.TokenInfo,
) -> bool:
    """Determine if token is a line that follows an indent.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.
    prev_token : tokenize.TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is a line that follows an indent, False otherwise.
    """
    if prev_token.type == tokenize.INDENT and prev_token.line in token.line:
        return True

    return False


def is_module_docstring(
    tokens: list[tokenize.TokenInfo],
    index: int,
) -> bool:
    """Determine if docstring is a module docstring."""
    # No code tokens before the string
    for k in range(index):
        if tokens[k][0] not in (
            tokenize.ENCODING,
            tokenize.COMMENT,
            tokenize.NEWLINE,
            tokenize.NL,
        ):
            return False
    return True


def is_nested_definition_line(token: tokenize.TokenInfo) -> bool:
    """Determine if token is a nested class or function/method definition line.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.

    Returns
    -------
    bool
        True if the token is a nested definition line, False otherwise.
    """
    return re.match(r"^ {4,}(async|class|def) ", token.line) is not None


def is_newline_continuation(
    token: tokenize.TokenInfo,
    prev_token: tokenize.TokenInfo,
) -> bool:
    """Determine if token is a continuation of a previous line.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.
    prev_token : tokenize.TokenInfo
        The previous token in the stream.

    Returns
    -------
    bool
        True if the token is a continuation of a previous line, False otherwise.
    """
    if (
        token.type in (tokenize.NEWLINE, tokenize.NL)
        and token.line.strip() in prev_token.line.strip()
        and token.line not in {"\n", "\r\n"}
    ):
        return True

    return False


def is_string_variable(
    token: tokenize.TokenInfo,
    prev_token: tokenize.TokenInfo,
) -> bool:
    """Determine if token is a string variable assignment.

    Parameters
    ----------
    token : tokenize.TokenInfo
        The token to check.
    prev_token : tokenize.TokenInfo
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

    if prev_token.type in _token_types and (
        '= """' in token.line or token.line in prev_token.line
    ):
        return True

    return False


def is_docstring_at_end_of_file(tokens: list[tokenize.TokenInfo], index: int) -> bool:
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
