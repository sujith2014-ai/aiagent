"""Ten ordinary missing actions. For each: the request (visible examples only), an independent reference implementation used to build HIDDEN spec cases, a correct generated source and a plausibly buggy first attempt.
All generated sources here are SCRIPTED (no LLM was called): what is measured is the gates, not a model's coding ability."""
from __future__ import annotations
import datetime, json, re
from dataclasses import dataclass
from integrations.toolgrowth.package import Candidate


@dataclass
class Task:
    task_id: str
    intent: str                 # what the user asked for in words
    description: str
    keywords: list
    visible: list               # examples shown to the generator
    ref: callable               # independent reference
    inputs: list                # hidden spec inputs (expected outputs come from ref)
    good: str
    buggy: str

    def spec_cases(self) -> list: return [{"input": i, "expected": self.ref(i)} for i in self.inputs]
    def visible_cases(self) -> list: return [{"input": i, "expected": self.ref(i)} for i in self.visible]


def _slug(i):
    s = re.sub(r"[^a-z0-9]+", "-", i["text"].lower()).strip("-"); return {"result": s}
def _csvsum(i):
    import csv
    rows = list(csv.reader(i["csv"].splitlines()))
    if not rows or i["column"] not in rows[0]: return {"error": "no such column"}
    c = rows[0].index(i["column"])
    try: return {"result": float(sum(float(r[c]) for r in rows[1:] if len(r) > c and r[c].strip() != ""))}
    except ValueError: return {"error": "non-numeric value"}
def _emails(i):
    seen, out = set(), []
    for m in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", i["text"]):
        m = m.lower()
        if m not in seen: seen.add(m); out.append(m)
    return {"result": out}
def _roman(i):
    n = i["n"]
    if not isinstance(n, int) or not 1 <= n <= 3999: return {"error": "out of range"}
    out = ""
    for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v: out += s; n -= v
    return {"result": out}
def _median(i):
    v = sorted(i["values"])
    if not v: return {"error": "empty"}
    m = len(v) // 2; return {"result": float(v[m]) if len(v) % 2 else (v[m - 1] + v[m]) / 2}
def _isoweek(i):
    try: y, w, _ = datetime.date.fromisoformat(i["date"]).isocalendar(); return {"result": {"year": y, "week": w}}
    except ValueError: return {"error": "invalid date"}
def _wordfreq(i):
    c = {}
    for w in re.findall(r"[a-z0-9']+", i["text"].lower()): c[w] = c.get(w, 0) + 1
    return {"result": [[w, n] for w, n in sorted(c.items(), key=lambda t: (-t[1], t[0]))[: i["k"]]]}
def _duration(i):
    m = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", i["text"].strip())
    if not m or not any(m.groups()): return {"error": "invalid duration"}
    d, h, mi, s = (int(x or 0) for x in m.groups()); return {"result": d * 86400 + h * 3600 + mi * 60 + s}
def _luhn(i):
    digits = i["number"].replace(" ", "").replace("-", "")
    if not digits.isdigit() or len(digits) < 2: return {"error": "not a number"}
    total = 0
    for k, ch in enumerate(reversed(digits)):
        d = int(ch)
        if k % 2: d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return {"result": total % 10 == 0}
def _dedupe(i):
    seen, out = set(), []
    for x in i["items"]:
        k = (type(x).__name__, x)
        if k not in seen: seen.add(k); out.append(x)
    return {"result": out}


TASKS = [
    Task("slugify", "turn this title into a url slug", "Convert text to a lowercase hyphenated URL slug", ["slug", "slugify", "url", "title", "hyphenated"],
         [{"text": "Hello, World!"}, {"text": "  Many   spaces "}], _slug,
         [{"text": t} for t in ["Hello, World!", "  Many   spaces ", "---x---", "Café au lait", "A_B C", "", "!!!", "2024 Report (final).pdf", "UPPER lower", "tabs\tand\nnewlines", "a--b", "x"]],
         "import re\ndef run(inp):\n    s = re.sub(r'[^a-z0-9]+', '-', inp['text'].lower())\n    return {'result': s.strip('-')}\n",
         "import re\ndef run(inp):\n    s = re.sub(r'[^a-z0-9]+', '-', inp['text'].lower())\n    return {'result': s}\n"),
    Task("csv_column_sum", "sum a column of this csv", "Sum the numeric values of one named csv column", ["csv", "column", "sum", "total", "numeric"],
         [{"csv": "a,b\n1,2\n3,4", "column": "b"}], _csvsum,
         [{"csv": c, "column": k} for c, k in [("a,b\n1,2\n3,4", "b"), ("a,b\n1,2\n3,4", "a"), ("x\n1.5\n2.5\n", "x"), ("a,b\n1,2", "z"), ("a,b\n1,x", "b"), ("a,b\n1,\n3,4", "b"), ("a\n", "a"), ("p,q\n-1,5\n-2,5", "p")]],
         "import csv\ndef run(inp):\n    rows = list(csv.reader(inp['csv'].splitlines()))\n    if not rows or inp['column'] not in rows[0]:\n        return {'error': 'no such column'}\n    c = rows[0].index(inp['column'])\n    try:\n        return {'result': float(sum(float(r[c]) for r in rows[1:] if len(r) > c and r[c].strip() != ''))}\n    except ValueError:\n        return {'error': 'non-numeric value'}\n",
         "import csv\ndef run(inp):\n    rows = list(csv.reader(inp['csv'].splitlines()))\n    c = rows[0].index(inp['column'])\n    return {'result': float(sum(float(r[c]) for r in rows[1:]))}\n"),
    Task("extract_emails", "pull the email addresses out of this text", "Extract unique email addresses from text, lowercased, in order of first appearance", ["email", "emails", "extract", "addresses", "text"],
         [{"text": "mail bob@x.org now"}], _emails,
         [{"text": t} for t in ["mail bob@x.org now", "A@B.com a@b.com", "none here", "x@y.z", "first.last+tag@sub.example.co.uk, other@example.com;", "bad@@x.com ok@x.io", "", "UPPER@CASE.ORG upper@case.org"]],
         "import re\ndef run(inp):\n    seen, out = set(), []\n    for m in re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', inp['text']):\n        m = m.lower()\n        if m not in seen:\n            seen.add(m); out.append(m)\n    return {'result': out}\n",
         "import re\ndef run(inp):\n    return {'result': [m.lower() for m in re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', inp['text'])]}\n"),
    Task("roman_numerals", "write this number as a roman numeral", "Convert an integer from 1 to 3999 to a roman numeral", ["roman", "numeral", "numerals", "integer", "convert"],
         [{"n": 14}, {"n": 2024}], _roman, [{"n": n} for n in [1, 4, 9, 14, 40, 90, 400, 444, 1994, 2024, 3999, 0, 4000, -5]],
         "def run(inp):\n    n = inp['n']\n    if not isinstance(n, int) or not 1 <= n <= 3999:\n        return {'error': 'out of range'}\n    out = ''\n    for v, s in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:\n        while n >= v:\n            out += s; n -= v\n    return {'result': out}\n",
         "def run(inp):\n    n = inp['n']\n    if not isinstance(n, int) or not 1 <= n <= 3999:\n        return {'error': 'out of range'}\n    out = ''\n    for v, s in [(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(5,'V'),(1,'I')]:\n        while n >= v:\n            out += s; n -= v\n    return {'result': out}\n"),
    Task("median_of_numbers", "what is the median of these numbers", "Median of a list of numbers", ["median", "numbers", "middle", "list", "statistics"],
         [{"values": [3, 1, 2]}], _median, [{"values": v} for v in [[3, 1, 2], [4, 1, 3, 2], [5], [], [2.5, 0.5], [-1, -3, -2], [10, 10, 10, 10], [1, 100, 3, 7, 5, 9]]],
         "def run(inp):\n    v = sorted(inp['values'])\n    if not v:\n        return {'error': 'empty'}\n    m = len(v) // 2\n    return {'result': float(v[m]) if len(v) % 2 else (v[m-1] + v[m]) / 2}\n",
         "def run(inp):\n    v = inp['values']\n    if not v:\n        return {'error': 'empty'}\n    m = len(v) // 2\n    return {'result': float(v[m]) if len(v) % 2 else (v[m-1] + v[m]) / 2}\n"),
    Task("iso_week_number", "which iso week is this date in", "ISO year and week number of a date (YYYY-MM-DD)", ["iso", "week", "number", "date", "calendar"],
         [{"date": "2024-12-30"}], _isoweek, [{"date": d} for d in ["2024-12-30", "2021-01-03", "2020-12-31", "2024-02-29", "2023-01-01", "2026-06-15", "2024-13-01", "not a date", "2019-12-29", "2015-01-01"]],
         "import datetime\ndef run(inp):\n    try:\n        y, w, _ = datetime.date.fromisoformat(inp['date']).isocalendar()\n        return {'result': {'year': y, 'week': w}}\n    except ValueError:\n        return {'error': 'invalid date'}\n",
         "import datetime\ndef run(inp):\n    try:\n        d = datetime.date.fromisoformat(inp['date'])\n        return {'result': {'year': d.year, 'week': int(d.strftime('%W'))}}\n    except ValueError:\n        return {'error': 'invalid date'}\n"),
    Task("word_frequency", "count the most common words in this text", "Top-k most frequent words (lowercase, ties broken alphabetically)", ["word", "words", "frequency", "common", "count"],
         [{"text": "a b a", "k": 1}], _wordfreq, [{"text": t, "k": k} for t, k in [("a b a", 1), ("the cat the dog the end", 2), ("b a c", 3), ("", 2), ("It's it's IT'S", 1), ("x y z x y z", 2), ("one", 5), ("a a b b c c", 3)]],
         "import re\ndef run(inp):\n    c = {}\n    for w in re.findall(r\"[a-z0-9']+\", inp['text'].lower()):\n        c[w] = c.get(w, 0) + 1\n    top = sorted(c.items(), key=lambda t: (-t[1], t[0]))[:inp['k']]\n    return {'result': [[w, n] for w, n in top]}\n",
         "import re\ndef run(inp):\n    c = {}\n    for w in re.findall(r\"[a-z0-9']+\", inp['text'].lower()):\n        c[w] = c.get(w, 0) + 1\n    top = sorted(c.items(), key=lambda t: -t[1])[:inp['k']]\n    return {'result': [[w, n] for w, n in top]}\n"),
    Task("parse_duration", "convert this duration to seconds", "Convert a duration like 1d2h3m4s to seconds", ["duration", "seconds", "convert", "parse", "time"],
         [{"text": "1h30m"}, {"text": "45s"}], _duration, [{"text": t} for t in ["1h30m", "45s", "2d", "1d2h3m4s", "10m", "", "abc", "5x", "1h 30m", "0s", "3h3s"]],
         "import re\ndef run(inp):\n    m = re.fullmatch(r'(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', inp['text'].strip())\n    if not m or not any(m.groups()):\n        return {'error': 'invalid duration'}\n    d, h, mi, s = (int(x or 0) for x in m.groups())\n    return {'result': d*86400 + h*3600 + mi*60 + s}\n",
         "import re\ndef run(inp):\n    m = re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', inp['text'].strip())\n    if not m or not any(m.groups()):\n        return {'error': 'invalid duration'}\n    h, mi, s = (int(x or 0) for x in m.groups())\n    return {'result': h*3600 + mi*60 + s}\n"),
    Task("luhn_check", "is this card number valid", "Luhn checksum validity of a digit string (spaces and hyphens ignored)", ["luhn", "card", "number", "valid", "checksum"],
         [{"number": "4539 1488 0343 6467"}], _luhn, [{"number": n} for n in ["4539 1488 0343 6467", "4539 1488 0343 6468", "79927398713", "79927398710", "0000", "1", "12a4", "", "5500-0000-0000-0004", "378282246310005"]],
         "def run(inp):\n    digits = inp['number'].replace(' ', '').replace('-', '')\n    if not digits.isdigit() or len(digits) < 2:\n        return {'error': 'not a number'}\n    total = 0\n    for k, ch in enumerate(reversed(digits)):\n        d = int(ch)\n        if k % 2:\n            d = d*2 - 9 if d > 4 else d*2\n        total += d\n    return {'result': total % 10 == 0}\n",
         "def run(inp):\n    digits = inp['number'].replace(' ', '').replace('-', '')\n    if not digits.isdigit() or len(digits) < 2:\n        return {'error': 'not a number'}\n    total = 0\n    for k, ch in enumerate(digits):\n        d = int(ch)\n        if k % 2:\n            d = d*2 - 9 if d > 4 else d*2\n        total += d\n    return {'result': total % 10 == 0}\n"),
    Task("dedupe_items", "remove duplicates from this list but keep the order", "Remove duplicate items from a list of numbers and strings, keeping first occurrences (1 and '1' differ)", ["duplicates", "dedupe", "list", "order", "unique"],
         [{"items": [1, 2, 1]}], _dedupe, [{"items": v} for v in [[1, 2, 1], [], ["a", "b", "a"], [1, "1", 1], [3, 3, 3], [2.0, 2, 3], ["x"], [0, False, 0], ["b", "a", "b", "c", "a"]]],
         "def run(inp):\n    seen, out = set(), []\n    for x in inp['items']:\n        k = (type(x).__name__, x)\n        if k not in seen:\n            seen.add(k); out.append(x)\n    return {'result': out}\n",
         "def run(inp):\n    seen, out = [], []\n    for x in inp['items']:\n        if x not in seen:\n            seen.append(x); out.append(x)\n    return {'result': out}\n"),
]
BY_ID = {t.task_id: t for t in TASKS}


def candidate(task: Task, version="0.1.0", buggy=False, generator="scripted") -> Candidate:
    src = task.buggy if buggy else task.good
    # the generator writes its own tests from the visible examples (it never sees the hidden spec)
    return Candidate(task.task_id, version, task.description, src, tests=task.visible_cases(), keywords=task.keywords, generator=generator)
