from math import sqrt

DOSE_TOLERANCE = 0.02
DTA = 2


class Point:
    def __init__(self, position: float, dose: float):
        self.position = position
        self.dose = dose


def gamma_punctual(reference_point: Point, evaluation_point: Point) -> float:
    return sqrt(((reference_point.dose - evaluation_point.dose) / DOSE_TOLERANCE) ** 2 +
                ((reference_point.position - evaluation_point.position) / DTA) ** 2)


def gamma_1d(reference_points: list[Point], evaluation_points: list[Point]) -> list[float]:
    gammas: list[float] = []
    for ref in reference_points:
        min_gamma: float = float('inf')
        for evaluation in evaluation_points:
            punctual_gamma: float = gamma_punctual(ref, evaluation)
            if punctual_gamma < min_gamma:
                min_gamma = punctual_gamma
        gammas.append(min_gamma)
    return gammas

