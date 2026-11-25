"""Python lexing for the cleanup tools.

stdin: JSON list of {"key": str, "src": str}
stdout: JSON object key -> {"comments": [[line, col, end_line, end_col], ...] | null,
                            "ast": sha256 of ast.dump | null}
Columns are code-point offsets within the line (tokenize convention).
"""
import ast
import hashlib
import io
import json
import sys
import tokenize
import warnings

def comments(src):
    try:
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                (l1, c1), (l2, c2) = tok.start, tok.end
                if l1 == 1 and c1 == 0 and tok.string.startswith("#!"):
                    continue
                out.append([l1, c1, l2, c2])
        return out
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None

def ast_hash(src):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
        return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    except (SyntaxError, ValueError):
        return None

def main():
    items = json.load(sys.stdin)
    json.dump({i["key"]: {"comments": comments(i["src"]), "ast": ast_hash(i["src"])} for i in items}, sys.stdout)

main()
