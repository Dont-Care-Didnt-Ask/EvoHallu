import inspect
import textwrap


def function_to_string(func):
    """
    Takes a function and returns its source code as a string.

    args:
    func (function): The function whose source code is to be extracted.

    Returns:
    str: The source code of the function.
    """
    try:
        source = inspect.getsource(func)
        dedented_source = textwrap.dedent(source)
        return dedented_source
    except TypeError:
        return "The provided input is not a function."
    except OSError:
        return "The source code for the function could not be retrieved."


def string_to_function(cfg, func_str, imports):
    """
    Convert a string representation of a Python function into an actual Python function.

    Parameters:
    func_str (str): The string representation of the Python function.
    func_name (str): The name of the function to retrieve after executing the string.

    Returns:
    function: The actual Python function.
    """
    # Define a local dictionary to execute the function string
    local_dict = {}

    # Here I want to execute the imports
    exec(imports, globals(), local_dict)

    # Execute the function string
    exec(func_str, globals(), local_dict)

    # Retrieve the function from the local dictionary
    func = local_dict.get(cfg.function_str_to_extract)

    # Ensure the function exists in the local dictionary
    if func is None:
        raise ValueError(f"Function '{cfg.function_str_to_extract}' not found in the provided string.")

    return func


def extract_imports(code_output):
    """
    Extracts the import statements from the given code output.

    Parameters:
    code_output (str): The output string containing the code.

    Returns:
    str: The extracted import statements, or an empty string if no imports are found.
    """
    lines = code_output.split("\n")
    import_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Check if the line is an import statement
        if stripped_line.startswith("import ") or stripped_line.startswith("from "):
            import_lines.append(line)

    if import_lines:
        return "\n".join(import_lines)
    else:
        return ""


def extract_functions(code_output):
    """
    Extract the first top-level Python function definition from a model output.

    Robust behavior:
    - If fenced code blocks (``` ... ```) are present, prefer the first code block
      that contains a function definition. Reasoning outside code fences is ignored.
    - If no fences are present, scan the whole text.
    - Capture the full function body until the next top-level dedent (or end of block).
    - Does not stop on the first 'return' line (supports multi-line returns).

    Parameters:
    code_output (str): The raw assistant output.

    Returns:
    str: The extracted function definition text, or an empty string if not found.
    """

    def _split_code_blocks(text):
        blocks = []
        lines_local = text.split("\n")
        in_block = False
        current = []
        for ln in lines_local:
            if ln.startswith("```"):
                if not in_block:
                    in_block = True
                    current = []
                    # Skip the opening fence line (may include a language tag)
                    continue
                else:
                    # Closing fence: end current block
                    blocks.append("\n".join(current))
                    in_block = False
                    current = []
                    continue
            if in_block:
                current.append(ln)
        return blocks

    def _extract_first_function(text_segment):
        seg_lines = text_segment.split("\n")
        start_idx = -1
        for i, ln in enumerate(seg_lines):
            if ln.lstrip().startswith("def "):
                start_idx = i
                break
        if start_idx == -1:
            return ""

        def_line = seg_lines[start_idx]
        def_indent = len(def_line) - len(def_line.lstrip())

        collected = [def_line]
        # Collect subsequent lines that remain indented beyond the def indentation
        for j in range(start_idx + 1, len(seg_lines)):
            lnj = seg_lines[j]
            stripped = lnj.strip()
            if stripped == "":
                collected.append(lnj)
                continue
            indent = len(lnj) - len(lnj.lstrip())
            if indent <= def_indent and not lnj.lstrip().startswith("@") and not lnj.lstrip().startswith("def "):
                # Dedent to top-level or lower: end of function body
                break
            if lnj.lstrip().startswith("def ") and indent <= def_indent:
                break
            collected.append(lnj)
        return "\n".join(collected)

    # 1) Prefer code-fenced blocks
    if "```" in code_output:
        blocks = _split_code_blocks(code_output)
        for blk in blocks:
            if "def " in blk:
                extracted = _extract_first_function(blk)
                if extracted:
                    return extracted

    # 2) Fallback: scan the entire text
    return _extract_first_function(code_output)


def separate_imports_from_func(func_str):
    lines = func_str.splitlines()
    import_lines = []
    func_lines = []
    found_def = False
    for line in lines:
        if line.strip().startswith("def "):
            found_def = True
            func_lines.append(line)
        elif not found_def and (line.strip().startswith("import ") or line.strip().startswith("from ")):
            import_lines.append(line)
        else:
            if found_def:
                func_lines.append(line)
    return "\n".join(import_lines), "\n".join(func_lines)
