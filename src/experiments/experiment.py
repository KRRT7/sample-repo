def string_concat(n):
    # Use list comprehension and str.join for better performance
    return "".join([str(i) for i in range(n)])
