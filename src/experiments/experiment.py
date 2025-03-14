from collections import Counter


def word_frequency(text: str) -> dict[str, int]:
    words = text.lower().split()
    frequency = Counter(words)
    return dict(frequency)
