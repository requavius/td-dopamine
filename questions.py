# Question types, and how hard each one is for one particular person.

import math
import random
from dataclasses import dataclass, field


def _add1():
    a, b = random.randint(2, 9), random.randint(2, 9)
    return f"{a} + {b}", a + b


def _add2():
    a, b = random.randint(11, 99), random.randint(11, 99)
    return f"{a} + {b}", a + b


def _sub2():
    a = random.randint(30, 99)
    b = random.randint(11, a - 1)
    return f"{a} - {b}", a - b


def _times():
    a, b = random.randint(3, 12), random.randint(3, 12)
    return f"{a} × {b}", a * b


def _mul21():
    a, b = random.randint(12, 99), random.randint(3, 9)
    return f"{a} × {b}", a * b


def _sub3():
    a = random.randint(300, 999)
    b = random.randint(101, a - 1)
    return f"{a} - {b}", a - b


def _div31():
    b = random.randint(3, 9)
    q = random.randint(-(-100 // b), 999 // b)  # so that b * q has three digits
    return f"{b * q} ÷ {b}", q


def _mul22():
    a, b = random.randint(12, 99), random.randint(12, 99)
    return f"{a} × {b}", a * b


def _mul32():
    a, b = random.randint(101, 999), random.randint(12, 99)
    return f"{a} × {b}", a * b


@dataclass(frozen=True)
class QuestionType:
    name: str
    prior: float  # starting guess at difficulty, in logits; re-rated from answers
    make: object  # () -> (question text, integer answer)


TYPES = (
    QuestionType("add, 1-digit", -2.0, _add1),
    QuestionType("add, 2-digit", -0.5, _add2),
    QuestionType("subtract, 2-digit", 0.0, _sub2),
    QuestionType("times tables", 0.0, _times),
    QuestionType("multiply, 2-digit × 1-digit", 1.0, _mul21),
    QuestionType("subtract, 3-digit", 1.0, _sub3),
    QuestionType("divide, 3-digit ÷ 1-digit", 2.0, _div31),
    QuestionType("multiply, 2-digit × 2-digit", 2.5, _mul22),
    QuestionType("multiply, 3-digit × 2-digit", 3.5, _mul32),
)
N_TYPES = len(TYPES)

# Elo step sizes, shrinking as answers accumulate.
K_ABILITY = 0.6
SETTLE_ABILITY = 30  # answers overall at which the ability step has halved
K_TYPE = 0.6
SETTLE_TYPE = 10     # answers of one type at which that type's step has halved

# Types within this of the best match are all eligible, so near-equivalent types each
# get shown, and so re-rated.
TARGET_BAND = 0.1


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


# A type whose chance of being answered right is nearest `target`.
def choose_type(probs, target, rng=random):
    gaps = [abs(p - target) for p in probs]
    best = min(gaps)
    return rng.choice([t for t, g in enumerate(gaps) if g <= best + TARGET_BAND])


# What is known about one person: their ability and each type's difficulty for them.
@dataclass
class Ratings:
    ability: float = 0.0
    difficulty: list = field(default_factory=lambda: [t.prior for t in TYPES])
    answers: list = field(default_factory=lambda: [0] * N_TYPES)
    total: int = 0

    def p(self, t):
        return sigmoid(self.ability - self.difficulty[t]) # same logit scale as the simulator

    def probs(self):
        return [self.p(t) for t in range(N_TYPES)]

    # Rate one answer. Returns the chance they had of getting it right.
    def update(self, t, correct):
        p = self.p(t)
        err = float(correct) - p # surprise: what the rating failed to predict
        self.ability += K_ABILITY / (1 + self.total / SETTLE_ABILITY) * err
        self.difficulty[t] -= K_TYPE / (1 + self.answers[t] / SETTLE_TYPE) * err
        self.answers[t] += 1
        self.total += 1
        return p

    def to_dict(self):
        return {"ability": self.ability, "difficulty": list(self.difficulty),
                "answers": list(self.answers), "total": self.total}

    @classmethod
    def from_dict(cls, d):
        r = cls(ability=float(d["ability"]), total=int(d["total"]))
        # Keep the ratings that still line up; types added since start from their priors.
        for i, value in enumerate(d["difficulty"][:N_TYPES]):
            r.difficulty[i] = float(value)
        for i, value in enumerate(d["answers"][:N_TYPES]):
            r.answers[i] = int(value)
        return r


# A simulated person's true difficulty for each type: the priors, shifted per person.
def synthetic_type_diffs(spread=1.0, rng=random):
    return [t.prior + rng.gauss(0.0, spread) for t in TYPES]
