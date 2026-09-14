from dataclasses import dataclass

@dataclass
class Discrete:
    n: int

@dataclass
class Continuous:
    shape: tuple
    mins: tuple
    maxs: tuple