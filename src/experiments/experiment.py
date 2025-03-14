import re


def regex_match(strings: list[str], pattern: str) -> list[str]:
    # Compile the regex pattern once outside the loop for better performance
    compiled_pattern = re.compile(pattern)

    matched = [s for s in strings if compiled_pattern.match(s)]
    return matched
