import random

PLAUSIBLE = ["dogs", "cats", "birds", "students", "teachers", "cars", "flowers"]
IMPLAUSIBLE = ["dragons", "ghosts", "planets", "galaxies", "robots"]

def valid_syllogism(a,b,c):
    return f"All {a} are {b}. All {b} are {c}. Therefore, all {a} are {c}."

def invalid_syllogism(a,b,c):
    return f"All {a} are {b}. All {c} are {b}. Therefore, all {a} are {c}."

def generate_dataset(n=1000):
    data = []

    for _ in range(n):
        P = random.sample(PLAUSIBLE, 3)
        I = random.sample(IMPLAUSIBLE, 3)

        combos = [
            (valid_syllogism(*P), "valid"),
            (valid_syllogism(*I), "valid"),
            (invalid_syllogism(*P), "invalid"),
            (invalid_syllogism(*I), "invalid"),
        ]

        for text, label in combos:
            data.append({"text": text, "label": label})

    return data
