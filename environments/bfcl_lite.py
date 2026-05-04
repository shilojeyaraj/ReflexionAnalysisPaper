"""
Lightweight BFCL-compatible function-call evaluator.

Replaces the bfcl-eval package entirely. Uses only Python's built-in ast
module — zero external dependencies.

Why we dropped bfcl-eval:
  - Hard-pins numpy==1.26.4, which has no wheel for Python 3.13
  - Requires tree-sitter==0.21.3 (no Python 3.13 wheel, needs MSVC compiler)
  - Installs 25+ unrelated provider SDKs (anthropic, cohere, mistralai, boto3…)

This module provides the same things we need:
  1. decode_function_call()  — parse "func(a=1, b=2)" → [{func: {a:1, b:2}}]
  2. check_function_call()   — validate parsed call against ground truth spec
  3. TASKS                   — 40 bundled simple_python tasks
  4. MULTIPLE_TASKS          — 20 bundled multiple_function tasks (disambiguation)

Error type vocabulary:
  success | wrong_func_name | missing_required_param |
  wrong_arg_type | wrong_arg_value | no_function_call | bad_arguments |
  ambiguous_selection  (chose a candidate function, but the wrong one)
"""

import ast
import json
from typing import Any

# ---------------------------------------------------------------------------
# AST-based function call parser
# ---------------------------------------------------------------------------

def decode_function_call(response: str) -> list[dict]:
    """
    Parse a Python function call string into list[{func_name: {kwarg: value}}].

    Accepts:
        "get_weather(city='London', unit='celsius')"
        "  calculate_area(base=10, height=5)  "
    Returns:
        [{"get_weather": {"city": "London", "unit": "celsius"}}]
    Returns [] if nothing parseable is found.
    """
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()

    # Try direct parse
    parsed = _try_parse(text)
    if parsed:
        return parsed

    # Last-resort: pull the first line that looks like a function call
    for line in text.splitlines():
        line = line.strip()
        if "(" in line and line.endswith(")"):
            result = _try_parse(line)
            if result:
                return result

    return []


def _try_parse(text: str) -> list[dict]:
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return []
    if not isinstance(tree.body, ast.Call):
        return []
    try:
        return [_call_node_to_dict(tree.body)]
    except Exception:
        return []


def _call_node_to_dict(call: ast.Call) -> dict:
    if isinstance(call.func, ast.Name):
        func_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        func_name = call.func.attr
    else:
        raise ValueError("Cannot extract function name from AST node")

    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue  # **kwargs splat — skip
        kwargs[kw.arg] = ast.literal_eval(kw.value)

    return {func_name: kwargs}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def check_function_call(
    parsed: list[dict],
    ground_truth: list[dict],
    func_description: list[dict],
) -> dict:
    """
    Validate a parsed function call against ground-truth specifications.

    Args:
        parsed:          Output of decode_function_call()
        ground_truth:    List of acceptable calls, e.g.
                         [{"calc_area": {"base": [10], "height": [5]}}]
                         Each value is a list of acceptable values.
        func_description: List of function schemas (from the task dict).

    Returns:
        {"valid": bool, "error": list[str], "error_type": str}
    """
    if not parsed:
        return {
            "valid": False,
            "error": ["No parseable function call in response."],
            "error_type": "no_function_call",
        }

    call = parsed[0]
    actual_func = next(iter(call))
    actual_args: dict = call[actual_func]

    # ── Check function name ─────────────────────────────────────────────
    expected_funcs = [next(iter(gt)) for gt in ground_truth]
    if actual_func not in expected_funcs:
        # Distinguish ambiguous_selection (called a real candidate, wrong one)
        # from wrong_func_name (called something not in the candidate set at all)
        candidate_names = [f.get("name") for f in func_description if f.get("name")]
        if actual_func in candidate_names:
            error_type = "ambiguous_selection"
            error_msg = (
                f"Selected wrong function from candidates: got '{actual_func}', "
                f"correct function is one of {expected_funcs}"
            )
        else:
            error_type = "wrong_func_name"
            error_msg = (
                f"Wrong function name: got '{actual_func}', "
                f"expected one of {expected_funcs}"
            )
        return {
            "valid": False,
            "error": [error_msg],
            "error_type": error_type,
        }

    # Use the matching ground-truth entry
    gt_entry = next(gt for gt in ground_truth if next(iter(gt)) == actual_func)
    gt_args: dict = gt_entry[actual_func]

    # ── Check required parameters ───────────────────────────────────────
    schema = next(
        (f for f in func_description if f.get("name") == actual_func), None
    )
    if schema:
        required = schema.get("parameters", {}).get("required", [])
        for req in required:
            if req not in actual_args:
                return {
                    "valid": False,
                    "error": [f"Missing required parameter: '{req}'"],
                    "error_type": "missing_required_param",
                }

    # ── Check argument types ────────────────────────────────────────────
    if schema:
        props = schema.get("parameters", {}).get("properties", {})
        for arg_name, arg_val in actual_args.items():
            if arg_name not in props:
                continue
            expected_type = props[arg_name].get("type", "")
            type_ok = _check_type(arg_val, expected_type)
            if not type_ok:
                return {
                    "valid": False,
                    "error": [
                        f"Wrong type for '{arg_name}': "
                        f"got {type(arg_val).__name__}, expected {expected_type}"
                    ],
                    "error_type": "wrong_arg_type",
                }

    # ── Check argument values ───────────────────────────────────────────
    for arg_name, acceptable in gt_args.items():
        if arg_name not in actual_args:
            continue
        actual_val = actual_args[arg_name]
        # ground_truth values are lists of acceptable options
        if isinstance(acceptable, list):
            if actual_val not in acceptable:
                return {
                    "valid": False,
                    "error": [
                        f"Wrong value for '{arg_name}': "
                        f"got {actual_val!r}, acceptable: {acceptable}"
                    ],
                    "error_type": "wrong_arg_value",
                }
        else:
            if actual_val != acceptable:
                return {
                    "valid": False,
                    "error": [
                        f"Wrong value for '{arg_name}': "
                        f"got {actual_val!r}, expected {acceptable!r}"
                    ],
                    "error_type": "wrong_arg_value",
                }

    return {"valid": True, "error": [], "error_type": "success"}


def _check_type(value: Any, expected: str) -> bool:
    mapping = {
        "integer": int,
        "int": int,
        "number": (int, float),
        "float": float,
        "string": str,
        "str": str,
        "boolean": bool,
        "bool": bool,
    }
    if expected not in mapping:
        return True  # unknown type — don't penalise
    return isinstance(value, mapping[expected])


# ---------------------------------------------------------------------------
# Bundled task dataset (40 simple_python-style tasks)
# Format mirrors BFCL: task_id, description, functions, ground_truth
# ---------------------------------------------------------------------------

TASKS: list[dict] = [
    # ── Math / calculation ──────────────────────────────────────────────
    {
        "task_id": "simple_0",
        "question": "Calculate the area of a triangle with base 10 and height 5.",
        "function": [{"name": "calculate_triangle_area",
                      "description": "Calculate the area of a triangle.",
                      "parameters": {"type": "dict", "properties": {
                          "base":   {"type": "integer", "description": "Base length"},
                          "height": {"type": "integer", "description": "Height"}},
                          "required": ["base", "height"]}}],
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
    },
    {
        "task_id": "simple_1",
        "question": "What is 15 percent of 200?",
        "function": [{"name": "calculate_percentage",
                      "description": "Calculate a percentage of a number.",
                      "parameters": {"type": "dict", "properties": {
                          "value":      {"type": "number", "description": "The base number"},
                          "percentage": {"type": "number", "description": "The percentage"}},
                          "required": ["value", "percentage"]}}],
        "ground_truth": [{"calculate_percentage": {"value": [200], "percentage": [15]}}],
    },
    {
        "task_id": "simple_2",
        "question": "Find the hypotenuse of a right triangle with legs 3 and 4.",
        "function": [{"name": "calculate_hypotenuse",
                      "description": "Calculate hypotenuse using Pythagorean theorem.",
                      "parameters": {"type": "dict", "properties": {
                          "leg_a": {"type": "number"},
                          "leg_b": {"type": "number"}},
                          "required": ["leg_a", "leg_b"]}}],
        "ground_truth": [{"calculate_hypotenuse": {"leg_a": [3], "leg_b": [4]}}],
    },
    {
        "task_id": "simple_3",
        "question": "Calculate the compound interest for principal 1000, rate 5%, time 3 years.",
        "function": [{"name": "compound_interest",
                      "description": "Calculate compound interest.",
                      "parameters": {"type": "dict", "properties": {
                          "principal": {"type": "number"},
                          "rate":      {"type": "number", "description": "Annual rate as decimal, e.g. 0.05"},
                          "years":     {"type": "integer"}},
                          "required": ["principal", "rate", "years"]}}],
        "ground_truth": [{"compound_interest": {"principal": [1000], "rate": [0.05], "years": [3]}}],
    },
    {
        "task_id": "simple_4",
        "question": "What is the factorial of 7?",
        "function": [{"name": "factorial",
                      "description": "Compute the factorial of a non-negative integer.",
                      "parameters": {"type": "dict", "properties": {
                          "n": {"type": "integer", "description": "Non-negative integer"}},
                          "required": ["n"]}}],
        "ground_truth": [{"factorial": {"n": [7]}}],
    },
    # ── Unit conversion ─────────────────────────────────────────────────
    {
        "task_id": "simple_5",
        "question": "Convert 100 Fahrenheit to Celsius.",
        "function": [{"name": "fahrenheit_to_celsius",
                      "description": "Convert Fahrenheit to Celsius.",
                      "parameters": {"type": "dict", "properties": {
                          "fahrenheit": {"type": "number"}},
                          "required": ["fahrenheit"]}}],
        "ground_truth": [{"fahrenheit_to_celsius": {"fahrenheit": [100]}}],
    },
    {
        "task_id": "simple_6",
        "question": "How many kilometers is 26.2 miles?",
        "function": [{"name": "miles_to_kilometers",
                      "description": "Convert miles to kilometers.",
                      "parameters": {"type": "dict", "properties": {
                          "miles": {"type": "number"}},
                          "required": ["miles"]}}],
        "ground_truth": [{"miles_to_kilometers": {"miles": [26.2]}}],
    },
    {
        "task_id": "simple_7",
        "question": "Convert 5 kilograms to pounds.",
        "function": [{"name": "kilograms_to_pounds",
                      "description": "Convert kilograms to pounds.",
                      "parameters": {"type": "dict", "properties": {
                          "kilograms": {"type": "number"}},
                          "required": ["kilograms"]}}],
        "ground_truth": [{"kilograms_to_pounds": {"kilograms": [5]}}],
    },
    {
        "task_id": "simple_8",
        "question": "Convert 2500 milliseconds to seconds.",
        "function": [{"name": "milliseconds_to_seconds",
                      "description": "Convert milliseconds to seconds.",
                      "parameters": {"type": "dict", "properties": {
                          "milliseconds": {"type": "number"}},
                          "required": ["milliseconds"]}}],
        "ground_truth": [{"milliseconds_to_seconds": {"milliseconds": [2500]}}],
    },
    {
        "task_id": "simple_9",
        "question": "Convert 1 gigabyte to megabytes.",
        "function": [{"name": "gigabytes_to_megabytes",
                      "description": "Convert gigabytes to megabytes.",
                      "parameters": {"type": "dict", "properties": {
                          "gigabytes": {"type": "number"}},
                          "required": ["gigabytes"]}}],
        "ground_truth": [{"gigabytes_to_megabytes": {"gigabytes": [1]}}],
    },
    # ── String operations ───────────────────────────────────────────────
    {
        "task_id": "simple_10",
        "question": "Count how many times the word 'the' appears in the string 'the cat sat on the mat'.",
        "function": [{"name": "count_occurrences",
                      "description": "Count occurrences of a substring.",
                      "parameters": {"type": "dict", "properties": {
                          "text":      {"type": "string"},
                          "substring": {"type": "string"}},
                          "required": ["text", "substring"]}}],
        "ground_truth": [{"count_occurrences": {
            "text": ["the cat sat on the mat"],
            "substring": ["the"]}}],
    },
    {
        "task_id": "simple_11",
        "question": "Reverse the string 'hello world'.",
        "function": [{"name": "reverse_string",
                      "description": "Reverse a string.",
                      "parameters": {"type": "dict", "properties": {
                          "text": {"type": "string"}},
                          "required": ["text"]}}],
        "ground_truth": [{"reverse_string": {"text": ["hello world"]}}],
    },
    {
        "task_id": "simple_12",
        "question": "Check if the string 'racecar' is a palindrome.",
        "function": [{"name": "is_palindrome",
                      "description": "Check if a string is a palindrome.",
                      "parameters": {"type": "dict", "properties": {
                          "text": {"type": "string"}},
                          "required": ["text"]}}],
        "ground_truth": [{"is_palindrome": {"text": ["racecar"]}}],
    },
    {
        "task_id": "simple_13",
        "question": "Truncate 'Hello, World!' to 5 characters.",
        "function": [{"name": "truncate_string",
                      "description": "Truncate a string to a maximum length.",
                      "parameters": {"type": "dict", "properties": {
                          "text":       {"type": "string"},
                          "max_length": {"type": "integer"}},
                          "required": ["text", "max_length"]}}],
        "ground_truth": [{"truncate_string": {"text": ["Hello, World!"], "max_length": [5]}}],
    },
    {
        "task_id": "simple_14",
        "question": "Pad the number 42 with leading zeros to width 6.",
        "function": [{"name": "zero_pad",
                      "description": "Pad a number with leading zeros.",
                      "parameters": {"type": "dict", "properties": {
                          "number": {"type": "integer"},
                          "width":  {"type": "integer"}},
                          "required": ["number", "width"]}}],
        "ground_truth": [{"zero_pad": {"number": [42], "width": [6]}}],
    },
    # ── Date / time ─────────────────────────────────────────────────────
    {
        "task_id": "simple_15",
        "question": "How many days are between 2024-01-01 and 2024-03-15?",
        "function": [{"name": "days_between",
                      "description": "Calculate number of days between two dates.",
                      "parameters": {"type": "dict", "properties": {
                          "start_date": {"type": "string", "description": "ISO format YYYY-MM-DD"},
                          "end_date":   {"type": "string", "description": "ISO format YYYY-MM-DD"}},
                          "required": ["start_date", "end_date"]}}],
        "ground_truth": [{"days_between": {
            "start_date": ["2024-01-01"],
            "end_date":   ["2024-03-15"]}}],
    },
    {
        "task_id": "simple_16",
        "question": "Add 30 days to the date 2024-02-01.",
        "function": [{"name": "add_days",
                      "description": "Add a number of days to a date.",
                      "parameters": {"type": "dict", "properties": {
                          "date": {"type": "string"},
                          "days": {"type": "integer"}},
                          "required": ["date", "days"]}}],
        "ground_truth": [{"add_days": {"date": ["2024-02-01"], "days": [30]}}],
    },
    {
        "task_id": "simple_17",
        "question": "What day of the week is 2024-07-04?",
        "function": [{"name": "day_of_week",
                      "description": "Get the day of week for a given date.",
                      "parameters": {"type": "dict", "properties": {
                          "date": {"type": "string", "description": "ISO format YYYY-MM-DD"}},
                          "required": ["date"]}}],
        "ground_truth": [{"day_of_week": {"date": ["2024-07-04"]}}],
    },
    # ── List / array operations ─────────────────────────────────────────
    {
        "task_id": "simple_18",
        "question": "Find the second largest number in the list [3, 1, 4, 1, 5, 9, 2, 6].",
        "function": [{"name": "nth_largest",
                      "description": "Find the nth largest value in a list.",
                      "parameters": {"type": "dict", "properties": {
                          "numbers": {"type": "string", "description": "Comma-separated numbers"},
                          "n":       {"type": "integer"}},
                          "required": ["numbers", "n"]}}],
        "ground_truth": [{"nth_largest": {"numbers": ["3,1,4,1,5,9,2,6"], "n": [2]}}],
    },
    {
        "task_id": "simple_19",
        "question": "Calculate the median of [4, 1, 7, 2, 9].",
        "function": [{"name": "median",
                      "description": "Calculate the median of a list of numbers.",
                      "parameters": {"type": "dict", "properties": {
                          "numbers": {"type": "string", "description": "Comma-separated numbers"}},
                          "required": ["numbers"]}}],
        "ground_truth": [{"median": {"numbers": ["4,1,7,2,9"]}}],
    },
    {
        "task_id": "simple_20",
        "question": "Remove duplicates from the list [1, 2, 2, 3, 3, 3, 4].",
        "function": [{"name": "remove_duplicates",
                      "description": "Remove duplicate values from a comma-separated list.",
                      "parameters": {"type": "dict", "properties": {
                          "items": {"type": "string"}},
                          "required": ["items"]}}],
        "ground_truth": [{"remove_duplicates": {"items": ["1,2,2,3,3,3,4"]}}],
    },
    # ── Lookup / search ─────────────────────────────────────────────────
    {
        "task_id": "simple_21",
        "question": "Look up the capital city of France.",
        "function": [{"name": "get_capital",
                      "description": "Get the capital city of a country.",
                      "parameters": {"type": "dict", "properties": {
                          "country": {"type": "string"}},
                          "required": ["country"]}}],
        "ground_truth": [{"get_capital": {"country": ["France"]}}],
    },
    {
        "task_id": "simple_22",
        "question": "Get the currency code for Japan.",
        "function": [{"name": "get_currency",
                      "description": "Get the ISO currency code for a country.",
                      "parameters": {"type": "dict", "properties": {
                          "country": {"type": "string"}},
                          "required": ["country"]}}],
        "ground_truth": [{"get_currency": {"country": ["Japan"]}}],
    },
    {
        "task_id": "simple_23",
        "question": "What is the atomic number of Carbon?",
        "function": [{"name": "get_atomic_number",
                      "description": "Get the atomic number of a chemical element.",
                      "parameters": {"type": "dict", "properties": {
                          "element": {"type": "string"}},
                          "required": ["element"]}}],
        "ground_truth": [{"get_atomic_number": {"element": ["Carbon", "carbon", "C"]}}],
    },
    {
        "task_id": "simple_24",
        "question": "Find the population of Brazil.",
        "function": [{"name": "get_population",
                      "description": "Get the population of a country.",
                      "parameters": {"type": "dict", "properties": {
                          "country": {"type": "string"}},
                          "required": ["country"]}}],
        "ground_truth": [{"get_population": {"country": ["Brazil"]}}],
    },
    # ── Weather / geo ───────────────────────────────────────────────────
    {
        "task_id": "simple_25",
        "question": "Get the current weather in London in Celsius.",
        "function": [{"name": "get_weather",
                      "description": "Get the current weather for a city.",
                      "parameters": {"type": "dict", "properties": {
                          "city": {"type": "string"},
                          "unit": {"type": "string",
                                   "enum": ["celsius", "fahrenheit"],
                                   "description": "Temperature unit"}},
                          "required": ["city", "unit"]}}],
        "ground_truth": [{"get_weather": {"city": ["London"], "unit": ["celsius"]}}],
    },
    {
        "task_id": "simple_26",
        "question": "Get the latitude and longitude of Tokyo.",
        "function": [{"name": "get_coordinates",
                      "description": "Get the geographic coordinates of a city.",
                      "parameters": {"type": "dict", "properties": {
                          "city": {"type": "string"}},
                          "required": ["city"]}}],
        "ground_truth": [{"get_coordinates": {"city": ["Tokyo"]}}],
    },
    {
        "task_id": "simple_27",
        "question": "Get the timezone for New York.",
        "function": [{"name": "get_timezone",
                      "description": "Get the timezone identifier for a city.",
                      "parameters": {"type": "dict", "properties": {
                          "city": {"type": "string"}},
                          "required": ["city"]}}],
        "ground_truth": [{"get_timezone": {"city": ["New York", "New York City", "NYC"]}}],
    },
    # ── Finance ─────────────────────────────────────────────────────────
    {
        "task_id": "simple_28",
        "question": "Convert 100 USD to EUR.",
        "function": [{"name": "convert_currency",
                      "description": "Convert an amount from one currency to another.",
                      "parameters": {"type": "dict", "properties": {
                          "amount":        {"type": "number"},
                          "from_currency": {"type": "string"},
                          "to_currency":   {"type": "string"}},
                          "required": ["amount", "from_currency", "to_currency"]}}],
        "ground_truth": [{"convert_currency": {
            "amount": [100],
            "from_currency": ["USD"],
            "to_currency": ["EUR"]}}],
    },
    {
        "task_id": "simple_29",
        "question": "Calculate the monthly payment for a loan of $20000 at 6% annual interest over 48 months.",
        "function": [{"name": "loan_payment",
                      "description": "Calculate monthly loan payment.",
                      "parameters": {"type": "dict", "properties": {
                          "principal":     {"type": "number"},
                          "annual_rate":   {"type": "number", "description": "Annual interest rate as decimal"},
                          "months":        {"type": "integer"}},
                          "required": ["principal", "annual_rate", "months"]}}],
        "ground_truth": [{"loan_payment": {
            "principal": [20000],
            "annual_rate": [0.06],
            "months": [48]}}],
    },
    # ── Text / NLP ──────────────────────────────────────────────────────
    {
        "task_id": "simple_30",
        "question": "Translate 'Good morning' to Spanish.",
        "function": [{"name": "translate_text",
                      "description": "Translate text to a target language.",
                      "parameters": {"type": "dict", "properties": {
                          "text":            {"type": "string"},
                          "target_language": {"type": "string"}},
                          "required": ["text", "target_language"]}}],
        "ground_truth": [{"translate_text": {
            "text": ["Good morning"],
            "target_language": ["Spanish", "spanish", "es"]}}],
    },
    {
        "task_id": "simple_31",
        "question": "Count the number of words in 'The quick brown fox jumps over the lazy dog'.",
        "function": [{"name": "count_words",
                      "description": "Count the number of words in a text.",
                      "parameters": {"type": "dict", "properties": {
                          "text": {"type": "string"}},
                          "required": ["text"]}}],
        "ground_truth": [{"count_words": {
            "text": ["The quick brown fox jumps over the lazy dog"]}}],
    },
    {
        "task_id": "simple_32",
        "question": "Extract all email addresses from the text 'Contact us at hello@example.com or support@test.org'.",
        "function": [{"name": "extract_emails",
                      "description": "Extract all email addresses from a text string.",
                      "parameters": {"type": "dict", "properties": {
                          "text": {"type": "string"}},
                          "required": ["text"]}}],
        "ground_truth": [{"extract_emails": {
            "text": ["Contact us at hello@example.com or support@test.org"]}}],
    },
    # ── Search / filter ─────────────────────────────────────────────────
    {
        "task_id": "simple_33",
        "question": "Search for Python tutorials on page 2 with 10 results per page.",
        "function": [{"name": "search",
                      "description": "Perform a search query.",
                      "parameters": {"type": "dict", "properties": {
                          "query":    {"type": "string"},
                          "page":     {"type": "integer"},
                          "per_page": {"type": "integer"}},
                          "required": ["query", "page", "per_page"]}}],
        "ground_truth": [{"search": {
            "query": ["Python tutorials", "python tutorials"],
            "page": [2],
            "per_page": [10]}}],
    },
    {
        "task_id": "simple_34",
        "question": "Filter products by category 'electronics' with a maximum price of 500.",
        "function": [{"name": "filter_products",
                      "description": "Filter products by category and price.",
                      "parameters": {"type": "dict", "properties": {
                          "category":  {"type": "string"},
                          "max_price": {"type": "number"}},
                          "required": ["category", "max_price"]}}],
        "ground_truth": [{"filter_products": {
            "category": ["electronics", "Electronics"],
            "max_price": [500]}}],
    },
    # ── System / file ───────────────────────────────────────────────────
    {
        "task_id": "simple_35",
        "question": "Read the file at path '/data/report.txt' with UTF-8 encoding.",
        "function": [{"name": "read_file",
                      "description": "Read a file from disk.",
                      "parameters": {"type": "dict", "properties": {
                          "path":     {"type": "string"},
                          "encoding": {"type": "string", "description": "Character encoding"}},
                          "required": ["path", "encoding"]}}],
        "ground_truth": [{"read_file": {
            "path": ["/data/report.txt"],
            "encoding": ["utf-8", "UTF-8"]}}],
    },
    {
        "task_id": "simple_36",
        "question": "Create a directory at path '/tmp/output' with parents allowed.",
        "function": [{"name": "create_directory",
                      "description": "Create a directory.",
                      "parameters": {"type": "dict", "properties": {
                          "path":          {"type": "string"},
                          "create_parents": {"type": "boolean"}},
                          "required": ["path", "create_parents"]}}],
        "ground_truth": [{"create_directory": {
            "path": ["/tmp/output"],
            "create_parents": [True]}}],
    },
    # ── Miscellaneous ───────────────────────────────────────────────────
    {
        "task_id": "simple_37",
        "question": "Generate a random integer between 1 and 100.",
        "function": [{"name": "random_integer",
                      "description": "Generate a random integer within a range.",
                      "parameters": {"type": "dict", "properties": {
                          "min_value": {"type": "integer"},
                          "max_value": {"type": "integer"}},
                          "required": ["min_value", "max_value"]}}],
        "ground_truth": [{"random_integer": {"min_value": [1], "max_value": [100]}}],
    },
    {
        "task_id": "simple_38",
        "question": "Sort a list of names ['Charlie', 'Alice', 'Bob'] in ascending order.",
        "function": [{"name": "sort_list",
                      "description": "Sort a comma-separated list of items.",
                      "parameters": {"type": "dict", "properties": {
                          "items":     {"type": "string"},
                          "ascending": {"type": "boolean"}},
                          "required": ["items", "ascending"]}}],
        "ground_truth": [{"sort_list": {
            "items": ["Charlie,Alice,Bob", "Alice,Bob,Charlie"],
            "ascending": [True]}}],
    },
    {
        "task_id": "simple_39",
        "question": "Hash the string 'password123' using SHA-256.",
        "function": [{"name": "hash_string",
                      "description": "Hash a string using a cryptographic algorithm.",
                      "parameters": {"type": "dict", "properties": {
                          "text":      {"type": "string"},
                          "algorithm": {"type": "string",
                                        "enum": ["md5", "sha1", "sha256", "sha512"]}},
                          "required": ["text", "algorithm"]}}],
        "ground_truth": [{"hash_string": {
            "text": ["password123"],
            "algorithm": ["sha256", "SHA-256", "sha-256"]}}],
    },
]


# ---------------------------------------------------------------------------
# multiple_function tasks — agent must select one correct function from a set
# of similar candidates. Tests disambiguation (ambiguous_selection error type).
# ---------------------------------------------------------------------------

MULTIPLE_TASKS: list[dict] = [
    # ── Weather variants ────────────────────────────────────────────────
    {
        "task_id": "multi_0",
        "question": "What is the current temperature in Paris right now?",
        "function": [
            {"name": "get_weather_current",
             "description": "Get the current real-time weather for a city.",
             "parameters": {"type": "dict", "properties": {
                 "city": {"type": "string"},
                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                 "required": ["city", "unit"]}},
            {"name": "get_weather_forecast",
             "description": "Get a multi-day weather forecast for a city.",
             "parameters": {"type": "dict", "properties": {
                 "city": {"type": "string"},
                 "days": {"type": "integer"},
                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                 "required": ["city", "days", "unit"]}},
            {"name": "get_weather_historical",
             "description": "Get historical weather data for a city on a past date.",
             "parameters": {"type": "dict", "properties": {
                 "city": {"type": "string"},
                 "date": {"type": "string"},
                 "unit": {"type": "string"}},
                 "required": ["city", "date", "unit"]}},
        ],
        "ground_truth": [{"get_weather_current": {"city": ["Paris"], "unit": ["celsius", "fahrenheit"]}}],
    },
    {
        "task_id": "multi_1",
        "question": "What will the weather be like in Tokyo over the next 5 days?",
        "function": [
            {"name": "get_weather_current",
             "description": "Get the current real-time weather for a city.",
             "parameters": {"type": "dict", "properties": {
                 "city": {"type": "string"},
                 "unit": {"type": "string"}},
                 "required": ["city", "unit"]}},
            {"name": "get_weather_forecast",
             "description": "Get a multi-day weather forecast for a city.",
             "parameters": {"type": "dict", "properties": {
                 "city": {"type": "string"},
                 "days": {"type": "integer"},
                 "unit": {"type": "string"}},
                 "required": ["city", "days", "unit"]}},
            {"name": "get_weather_historical",
             "description": "Get historical weather data for a city on a past date.",
             "parameters": {"type": "dict", "properties": {
                 "city": {"type": "string"},
                 "date": {"type": "string"},
                 "unit": {"type": "string"}},
                 "required": ["city", "date", "unit"]}},
        ],
        "ground_truth": [{"get_weather_forecast": {"city": ["Tokyo"], "days": [5], "unit": ["celsius", "fahrenheit"]}}],
    },
    # ── Search variants ─────────────────────────────────────────────────
    {
        "task_id": "multi_2",
        "question": "Search for recent news articles about climate change.",
        "function": [
            {"name": "search_news",
             "description": "Search for news articles by keyword.",
             "parameters": {"type": "dict", "properties": {
                 "query":    {"type": "string"},
                 "max_results": {"type": "integer"}},
                 "required": ["query"]}},
            {"name": "search_web",
             "description": "Perform a general web search.",
             "parameters": {"type": "dict", "properties": {
                 "query": {"type": "string"},
                 "page":  {"type": "integer"}},
                 "required": ["query"]}},
            {"name": "search_academic",
             "description": "Search academic papers and research publications.",
             "parameters": {"type": "dict", "properties": {
                 "query":     {"type": "string"},
                 "year_from": {"type": "integer"}},
                 "required": ["query"]}},
        ],
        "ground_truth": [{"search_news": {"query": ["climate change"]}}],
    },
    {
        "task_id": "multi_3",
        "question": "Find peer-reviewed papers on transformer architectures published after 2020.",
        "function": [
            {"name": "search_news",
             "description": "Search for news articles by keyword.",
             "parameters": {"type": "dict", "properties": {
                 "query": {"type": "string"}},
                 "required": ["query"]}},
            {"name": "search_web",
             "description": "Perform a general web search.",
             "parameters": {"type": "dict", "properties": {
                 "query": {"type": "string"}},
                 "required": ["query"]}},
            {"name": "search_academic",
             "description": "Search academic papers and research publications.",
             "parameters": {"type": "dict", "properties": {
                 "query":     {"type": "string"},
                 "year_from": {"type": "integer"}},
                 "required": ["query"]}},
        ],
        "ground_truth": [{"search_academic": {"query": ["transformer architectures", "transformers"], "year_from": [2020, 2021]}}],
    },
    # ── File operations ─────────────────────────────────────────────────
    {
        "task_id": "multi_4",
        "question": "Read the contents of the file at '/data/report.csv'.",
        "function": [
            {"name": "read_file",
             "description": "Read a file from disk and return its text contents.",
             "parameters": {"type": "dict", "properties": {
                 "path":     {"type": "string"},
                 "encoding": {"type": "string"}},
                 "required": ["path"]}},
            {"name": "write_file",
             "description": "Write content to a file on disk.",
             "parameters": {"type": "dict", "properties": {
                 "path":    {"type": "string"},
                 "content": {"type": "string"}},
                 "required": ["path", "content"]}},
            {"name": "delete_file",
             "description": "Delete a file from disk.",
             "parameters": {"type": "dict", "properties": {
                 "path": {"type": "string"}},
                 "required": ["path"]}},
        ],
        "ground_truth": [{"read_file": {"path": ["/data/report.csv"]}}],
    },
    {
        "task_id": "multi_5",
        "question": "Remove the file '/tmp/cache.log' from disk.",
        "function": [
            {"name": "read_file",
             "description": "Read a file from disk and return its text contents.",
             "parameters": {"type": "dict", "properties": {
                 "path": {"type": "string"}},
                 "required": ["path"]}},
            {"name": "write_file",
             "description": "Write content to a file on disk.",
             "parameters": {"type": "dict", "properties": {
                 "path":    {"type": "string"},
                 "content": {"type": "string"}},
                 "required": ["path", "content"]}},
            {"name": "delete_file",
             "description": "Delete a file from disk.",
             "parameters": {"type": "dict", "properties": {
                 "path": {"type": "string"}},
                 "required": ["path"]}},
        ],
        "ground_truth": [{"delete_file": {"path": ["/tmp/cache.log"]}}],
    },
    # ── User / account operations ────────────────────────────────────────
    {
        "task_id": "multi_6",
        "question": "Get the profile information for user with ID 42.",
        "function": [
            {"name": "get_user_profile",
             "description": "Retrieve a user's profile by user ID.",
             "parameters": {"type": "dict", "properties": {
                 "user_id": {"type": "integer"}},
                 "required": ["user_id"]}},
            {"name": "update_user_profile",
             "description": "Update fields on a user's profile.",
             "parameters": {"type": "dict", "properties": {
                 "user_id": {"type": "integer"},
                 "fields":  {"type": "string"}},
                 "required": ["user_id", "fields"]}},
            {"name": "delete_user",
             "description": "Permanently delete a user account.",
             "parameters": {"type": "dict", "properties": {
                 "user_id": {"type": "integer"}},
                 "required": ["user_id"]}},
        ],
        "ground_truth": [{"get_user_profile": {"user_id": [42]}}],
    },
    {
        "task_id": "multi_7",
        "question": "Change the email address on user 17's profile to 'new@example.com'.",
        "function": [
            {"name": "get_user_profile",
             "description": "Retrieve a user's profile by user ID.",
             "parameters": {"type": "dict", "properties": {
                 "user_id": {"type": "integer"}},
                 "required": ["user_id"]}},
            {"name": "update_user_profile",
             "description": "Update fields on a user's profile. Fields as JSON string.",
             "parameters": {"type": "dict", "properties": {
                 "user_id": {"type": "integer"},
                 "fields":  {"type": "string", "description": "JSON of fields to update"}},
                 "required": ["user_id", "fields"]}},
            {"name": "delete_user",
             "description": "Permanently delete a user account.",
             "parameters": {"type": "dict", "properties": {
                 "user_id": {"type": "integer"}},
                 "required": ["user_id"]}},
        ],
        "ground_truth": [{"update_user_profile": {
            "user_id": [17],
            "fields": ['{"email": "new@example.com"}', '{"email":"new@example.com"}']}}],
    },
    # ── Database / query variants ────────────────────────────────────────
    {
        "task_id": "multi_8",
        "question": "Count the number of records in the 'orders' table.",
        "function": [
            {"name": "db_count",
             "description": "Count rows in a database table.",
             "parameters": {"type": "dict", "properties": {
                 "table": {"type": "string"}},
                 "required": ["table"]}},
            {"name": "db_query",
             "description": "Execute a raw SQL SELECT query.",
             "parameters": {"type": "dict", "properties": {
                 "sql": {"type": "string"}},
                 "required": ["sql"]}},
            {"name": "db_insert",
             "description": "Insert a new row into a database table.",
             "parameters": {"type": "dict", "properties": {
                 "table": {"type": "string"},
                 "data":  {"type": "string"}},
                 "required": ["table", "data"]}},
        ],
        "ground_truth": [{"db_count": {"table": ["orders"]}}],
    },
    {
        "task_id": "multi_9",
        "question": "Run the SQL query 'SELECT * FROM products WHERE price < 50'.",
        "function": [
            {"name": "db_count",
             "description": "Count rows in a database table.",
             "parameters": {"type": "dict", "properties": {
                 "table": {"type": "string"}},
                 "required": ["table"]}},
            {"name": "db_query",
             "description": "Execute a raw SQL SELECT query.",
             "parameters": {"type": "dict", "properties": {
                 "sql": {"type": "string"}},
                 "required": ["sql"]}},
            {"name": "db_insert",
             "description": "Insert a new row into a database table.",
             "parameters": {"type": "dict", "properties": {
                 "table": {"type": "string"},
                 "data":  {"type": "string"}},
                 "required": ["table", "data"]}},
        ],
        "ground_truth": [{"db_query": {"sql": ["SELECT * FROM products WHERE price < 50"]}}],
    },
    # ── Send / notify variants ───────────────────────────────────────────
    {
        "task_id": "multi_10",
        "question": "Send an email to 'bob@example.com' with subject 'Meeting reminder'.",
        "function": [
            {"name": "send_email",
             "description": "Send an email message.",
             "parameters": {"type": "dict", "properties": {
                 "to":      {"type": "string"},
                 "subject": {"type": "string"},
                 "body":    {"type": "string"}},
                 "required": ["to", "subject"]}},
            {"name": "send_sms",
             "description": "Send an SMS text message.",
             "parameters": {"type": "dict", "properties": {
                 "phone_number": {"type": "string"},
                 "message":      {"type": "string"}},
                 "required": ["phone_number", "message"]}},
            {"name": "send_push_notification",
             "description": "Send a push notification to a device.",
             "parameters": {"type": "dict", "properties": {
                 "device_id": {"type": "string"},
                 "title":     {"type": "string"},
                 "body":      {"type": "string"}},
                 "required": ["device_id", "title"]}},
        ],
        "ground_truth": [{"send_email": {"to": ["bob@example.com"], "subject": ["Meeting reminder"]}}],
    },
    {
        "task_id": "multi_11",
        "question": "Text the number +1-555-1234 with the message 'Your code is 4821'.",
        "function": [
            {"name": "send_email",
             "description": "Send an email message.",
             "parameters": {"type": "dict", "properties": {
                 "to":      {"type": "string"},
                 "subject": {"type": "string"}},
                 "required": ["to", "subject"]}},
            {"name": "send_sms",
             "description": "Send an SMS text message.",
             "parameters": {"type": "dict", "properties": {
                 "phone_number": {"type": "string"},
                 "message":      {"type": "string"}},
                 "required": ["phone_number", "message"]}},
            {"name": "send_push_notification",
             "description": "Send a push notification to a device.",
             "parameters": {"type": "dict", "properties": {
                 "device_id": {"type": "string"},
                 "title":     {"type": "string"}},
                 "required": ["device_id", "title"]}},
        ],
        "ground_truth": [{"send_sms": {
            "phone_number": ["+1-555-1234", "+15551234"],
            "message": ["Your code is 4821"]}}],
    },
    # ── Calendar / scheduling ────────────────────────────────────────────
    {
        "task_id": "multi_12",
        "question": "Create a new calendar event called 'Team standup' on 2024-09-15 at 09:00.",
        "function": [
            {"name": "create_calendar_event",
             "description": "Create a new calendar event.",
             "parameters": {"type": "dict", "properties": {
                 "title": {"type": "string"},
                 "date":  {"type": "string"},
                 "time":  {"type": "string"}},
                 "required": ["title", "date"]}},
            {"name": "get_calendar_events",
             "description": "List calendar events for a given date.",
             "parameters": {"type": "dict", "properties": {
                 "date": {"type": "string"}},
                 "required": ["date"]}},
            {"name": "delete_calendar_event",
             "description": "Delete a calendar event by its ID.",
             "parameters": {"type": "dict", "properties": {
                 "event_id": {"type": "string"}},
                 "required": ["event_id"]}},
        ],
        "ground_truth": [{"create_calendar_event": {
            "title": ["Team standup"],
            "date":  ["2024-09-15"],
            "time":  ["09:00", "9:00"]}}],
    },
    {
        "task_id": "multi_13",
        "question": "What events are scheduled for 2024-09-20?",
        "function": [
            {"name": "create_calendar_event",
             "description": "Create a new calendar event.",
             "parameters": {"type": "dict", "properties": {
                 "title": {"type": "string"},
                 "date":  {"type": "string"}},
                 "required": ["title", "date"]}},
            {"name": "get_calendar_events",
             "description": "List calendar events for a given date.",
             "parameters": {"type": "dict", "properties": {
                 "date": {"type": "string"}},
                 "required": ["date"]}},
            {"name": "delete_calendar_event",
             "description": "Delete a calendar event by its ID.",
             "parameters": {"type": "dict", "properties": {
                 "event_id": {"type": "string"}},
                 "required": ["event_id"]}},
        ],
        "ground_truth": [{"get_calendar_events": {"date": ["2024-09-20"]}}],
    },
    # ── Payment / order variants ─────────────────────────────────────────
    {
        "task_id": "multi_14",
        "question": "Charge $49.99 to payment method 'pm_abc123'.",
        "function": [
            {"name": "charge_payment",
             "description": "Charge a payment method a specified amount.",
             "parameters": {"type": "dict", "properties": {
                 "payment_method_id": {"type": "string"},
                 "amount":            {"type": "number"},
                 "currency":          {"type": "string"}},
                 "required": ["payment_method_id", "amount"]}},
            {"name": "refund_payment",
             "description": "Refund a previously completed charge.",
             "parameters": {"type": "dict", "properties": {
                 "charge_id": {"type": "string"},
                 "amount":    {"type": "number"}},
                 "required": ["charge_id"]}},
            {"name": "get_payment_status",
             "description": "Get the status of a charge by charge ID.",
             "parameters": {"type": "dict", "properties": {
                 "charge_id": {"type": "string"}},
                 "required": ["charge_id"]}},
        ],
        "ground_truth": [{"charge_payment": {
            "payment_method_id": ["pm_abc123"],
            "amount": [49.99]}}],
    },
    {
        "task_id": "multi_15",
        "question": "Refund $20 from charge ID 'ch_xyz789'.",
        "function": [
            {"name": "charge_payment",
             "description": "Charge a payment method a specified amount.",
             "parameters": {"type": "dict", "properties": {
                 "payment_method_id": {"type": "string"},
                 "amount":            {"type": "number"}},
                 "required": ["payment_method_id", "amount"]}},
            {"name": "refund_payment",
             "description": "Refund a previously completed charge.",
             "parameters": {"type": "dict", "properties": {
                 "charge_id": {"type": "string"},
                 "amount":    {"type": "number"}},
                 "required": ["charge_id"]}},
            {"name": "get_payment_status",
             "description": "Get the status of a charge by charge ID.",
             "parameters": {"type": "dict", "properties": {
                 "charge_id": {"type": "string"}},
                 "required": ["charge_id"]}},
        ],
        "ground_truth": [{"refund_payment": {"charge_id": ["ch_xyz789"], "amount": [20, 20.0]}}],
    },
    # ── Translation / language ───────────────────────────────────────────
    {
        "task_id": "multi_16",
        "question": "Detect what language the text 'Bonjour le monde' is written in.",
        "function": [
            {"name": "detect_language",
             "description": "Detect the language of a given text.",
             "parameters": {"type": "dict", "properties": {
                 "text": {"type": "string"}},
                 "required": ["text"]}},
            {"name": "translate_text",
             "description": "Translate text from one language to another.",
             "parameters": {"type": "dict", "properties": {
                 "text":            {"type": "string"},
                 "target_language": {"type": "string"},
                 "source_language": {"type": "string"}},
                 "required": ["text", "target_language"]}},
            {"name": "get_language_code",
             "description": "Get the ISO 639-1 code for a language name.",
             "parameters": {"type": "dict", "properties": {
                 "language_name": {"type": "string"}},
                 "required": ["language_name"]}},
        ],
        "ground_truth": [{"detect_language": {"text": ["Bonjour le monde"]}}],
    },
    {
        "task_id": "multi_17",
        "question": "Translate 'Thank you very much' into German.",
        "function": [
            {"name": "detect_language",
             "description": "Detect the language of a given text.",
             "parameters": {"type": "dict", "properties": {
                 "text": {"type": "string"}},
                 "required": ["text"]}},
            {"name": "translate_text",
             "description": "Translate text from one language to another.",
             "parameters": {"type": "dict", "properties": {
                 "text":            {"type": "string"},
                 "target_language": {"type": "string"}},
                 "required": ["text", "target_language"]}},
            {"name": "get_language_code",
             "description": "Get the ISO 639-1 code for a language name.",
             "parameters": {"type": "dict", "properties": {
                 "language_name": {"type": "string"}},
                 "required": ["language_name"]}},
        ],
        "ground_truth": [{"translate_text": {
            "text": ["Thank you very much"],
            "target_language": ["German", "german", "de"]}}],
    },
    # ── Log / audit variants ─────────────────────────────────────────────
    {
        "task_id": "multi_18",
        "question": "Retrieve the last 100 error-level log entries from the application log.",
        "function": [
            {"name": "get_logs",
             "description": "Fetch log entries filtered by level and count.",
             "parameters": {"type": "dict", "properties": {
                 "level": {"type": "string", "enum": ["debug", "info", "warning", "error"]},
                 "count": {"type": "integer"}},
                 "required": ["level", "count"]}},
            {"name": "clear_logs",
             "description": "Delete all log entries at or below a given level.",
             "parameters": {"type": "dict", "properties": {
                 "level": {"type": "string"}},
                 "required": ["level"]}},
            {"name": "export_logs",
             "description": "Export log entries to a file.",
             "parameters": {"type": "dict", "properties": {
                 "output_path": {"type": "string"},
                 "level":       {"type": "string"}},
                 "required": ["output_path"]}},
        ],
        "ground_truth": [{"get_logs": {"level": ["error"], "count": [100]}}],
    },
    {
        "task_id": "multi_19",
        "question": "Export all warning logs to the file '/var/log/warnings_export.txt'.",
        "function": [
            {"name": "get_logs",
             "description": "Fetch log entries filtered by level and count.",
             "parameters": {"type": "dict", "properties": {
                 "level": {"type": "string"},
                 "count": {"type": "integer"}},
                 "required": ["level", "count"]}},
            {"name": "clear_logs",
             "description": "Delete all log entries at or below a given level.",
             "parameters": {"type": "dict", "properties": {
                 "level": {"type": "string"}},
                 "required": ["level"]}},
            {"name": "export_logs",
             "description": "Export log entries to a file.",
             "parameters": {"type": "dict", "properties": {
                 "output_path": {"type": "string"},
                 "level":       {"type": "string"}},
                 "required": ["output_path"]}},
        ],
        "ground_truth": [{"export_logs": {
            "output_path": ["/var/log/warnings_export.txt"],
            "level":       ["warning", "warnings"]}}],
    },
]
