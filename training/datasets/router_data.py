"""Synthetic intent datasets for the learned router. Written independently of the evaluation sentences in
scripts/run_phase5.py / run_phase7.py (a test asserts no overlap)."""
from __future__ import annotations
import random

VERBS = ["", "", "please ", "can you ", "i want to ", "help me ", "tell me ", "i need to ", "quickly ", "just "]
PHRASES = {
    "compare_numbers": ["compare two numbers", "check which number is greater", "see if one number is bigger than another", "work out how two numbers relate",
        "decide whether the first value is less than equal to or greater than the second", "determine the ordering of a pair of values", "find the relation between two numbers",
        "say which is larger", "tell which of the two numbers is smaller", "judge if two numbers are equal", "rank a pair of numbers", "figure out which value is higher"],
    "point_region": ["check whether a point is inside the circle", "see if the coordinates fall within the round region", "decide whether this location lies inside the disc",
        "test if the point is in the circular area", "find out if the x y position is inside the shape", "classify a point as inside or outside the circle",
        "is the point within the circular zone", "tell me if these coordinates are in the round region", "determine whether a position is inside the ring"],
    "argmax_position": ["get the position of the greatest number", "tell which entry is the biggest", "get the index of the maximum element", "locate where the highest number sits",
        "work out which slot holds the top value", "find the location of the maximum among the values", "say which of the four values is largest", "pick the index of the greatest entry"],
    "majority_vote": ["decide whether most of the values exceed one half", "check if the majority of the values are above a half", "see whether more than half of the five values are high",
        "count if most values pass the threshold", "tell me if the vote of the values is a majority", "determine whether the majority of entries are over one half"],
}
OOS = ["translate this text into french", "what is the weather like today", "send an email to my boss", "summarise this long document", "write a short poem about autumn",
       "convert miles to kilometres", "book a flight to london", "play some relaxing music", "set an alarm for seven", "define the word serendipity", "plan a trip to italy",
       "calculate the sales tax on this bill", "sort this list alphabetically", "who won the football game", "reverse the string hello", "add these numbers together",
       "work out the average of these values", "what is the time in tokyo", "turn off the lights", "order a pizza", "find a recipe for pancakes", "check my bank balance",
       "recommend a good movie", "how tall is mount everest", "read my calendar for tomorrow", "call a taxi", "tell me the news", "spell necessary", "describe this picture", "explain photosynthesis"]
HARD_NEG = ["compare prices of different products online", "compare two phone plans", "find the largest file on the disk", "check the point of sale system", "region settings of my computer",
            "inside the building there is a fire alarm", "position of the company in the market", "maximum speed on this road", "which is the biggest country", "locate the nearest circle k store",
            "number of pages in this book", "is the shop open at this hour", "the value of this house", "half of the pizza is gone", "vote for your favourite singer"]
UNKNOWN = "UNKNOWN"


def make_dataset(capabilities: list[str], seed=0, n_per_class=70, hard_negatives=True):
    rng = random.Random(seed); ex = []
    for c in capabilities:
        for _ in range(n_per_class):
            ex.append((rng.choice(VERBS) + rng.choice(PHRASES[c]), c))
    pool = OOS + (HARD_NEG if hard_negatives else [])
    for _ in range(n_per_class * 2):
        ex.append((rng.choice(VERBS) + rng.choice(pool), UNKNOWN))
    return ex


def overlaps(intents, capabilities):
    """Evaluation intents that also occur (exactly, or after a leading verb phrase) in any training set we generate."""
    train = {t.strip() for caps in (capabilities,) for t, _ in make_dataset(caps, hard_negatives=True)}
    train |= {t.strip() for t, _ in make_dataset(capabilities, seed=77, n_per_class=20, hard_negatives=True)}
    train |= {t.strip() for t, _ in make_dataset(capabilities, hard_negatives=False)}
    return sorted({i for i in intents if any(t == i or t.endswith(" " + i) for t in train)})
