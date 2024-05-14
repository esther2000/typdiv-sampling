import math


def entropy(string: str) -> float:
    """Calculates the Shannon entropy of a string
    from: https://stackoverflow.com/questions/67059620/calculate-entropy-from-binary-bit-string
    """
    # get probability of chars in string
    prob = [string.count(c) / len(string) for c in set(list(string))]
    entropy = -sum(p * math.log(p) / math.log(2.0) for p in prob)

    return entropy


def fvi(string: str) -> float:
    """
    Return the number of unique feature values included for a feature string.
    """
    string = "".join([x for x in string if x == '0' or x == '1'])
    return len(set(string)) / 2


def mpd(pairs: list, distances: dict) -> float:
    """
    Calculate mean pairwise distance for each string
    """
    mpds = []
    for pair in pairs:
        mpds.append(distances[pair[0]][pair[1]])

    return sum(mpds) / len(mpds)
