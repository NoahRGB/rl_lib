from dataclasses import dataclass

@dataclass
class Discrete:
    n: int
    shape: tuple

@dataclass
class Continuous:
    shape: tuple
    mins: tuple
    maxs: tuple