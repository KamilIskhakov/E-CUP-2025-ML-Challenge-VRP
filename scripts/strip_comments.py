import io
import os
import sys
import tokenize


EXCLUDE_DIRS = {
    ".git",
    "notebooks",
    "__pycache__",
    ".venv",
    "venv",
}


def strip_comments_from_code(code: str) -> str:
    output_tokens = []
    try:
        reader = io.StringIO(code).readline
        prev_toktype = None
        for tok in tokenize.generate_tokens(reader):
            tok_type, tok_str, start, end, line = tok
            if tok_type == tokenize.COMMENT:
                continue
                                                             
            if tok_type == tokenize.NL and prev_toktype == tokenize.COMMENT:
                prev_toktype = tok_type
                continue
            output_tokens.append(tok)
            prev_toktype = tok_type
        return tokenize.untokenize(output_tokens)
    except tokenize.TokenError:
        return code


def process_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
        stripped = strip_comments_from_code(original)
        if stripped != original:
            with open(path, "w", encoding="utf-8") as f:
                f.write(stripped)
    except Exception:
        pass


def main(root: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            process_file(os.path.join(dirpath, fname))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")


