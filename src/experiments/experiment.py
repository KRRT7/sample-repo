def levenshtein_distance(s1: str, s2: str) -> int:
    # Ensure that len(s1) >= len(s2) for space optimization
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    # Initialize the previous row
    previous_row = list(range(len(s2) + 1))

    # Iterate over each character in s1
    for i in range(1, len(s1) + 1):
        previous_char = s1[i - 1]
        current_row = [i]

        # Iterate over each character in s2
        for j in range(1, len(s2) + 1):
            if previous_char == s2[j - 1]:
                current_row.append(previous_row[j - 1])
            else:
                current_row.append(
                    min(
                        previous_row[j] + 1,
                        current_row[j - 1] + 1,
                        previous_row[j - 1] + 1,
                    )
                )

        # Update the previous row to the current one
        previous_row = current_row

    return previous_row[-1]
