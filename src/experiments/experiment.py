def is_palindrome(text: str) -> bool:
    cleaned_text = "".join(filter(str.isalnum, text)).lower()
    return cleaned_text == cleaned_text[::-1]
