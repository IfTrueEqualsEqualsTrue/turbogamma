import pytest

from turbogamma.gamma import Point, gamma_punctual, gamma_1d, DTA, DOSE_TOLERANCE

ATOL = 1e-4
ARRAY_SIZE = 16
NULL_POINT: Point = Point(0.0, 0.0)
TEST_CONSTANT_DOSE = 5.0


def ramp(i: int) -> float:
    return TEST_CONSTANT_DOSE * i


@pytest.fixture
def zeros_points() -> list[Point]:
    return [NULL_POINT] * ARRAY_SIZE


@pytest.fixture
def constant_dose_offset_points() -> list[Point]:
    return [Point(0.0, TEST_CONSTANT_DOSE)] * ARRAY_SIZE


@pytest.fixture
def dta_ramp_points():
    def _make(shift: int) -> list[Point]:
        return [Point(DTA * (i + shift), ramp(i)) for i in range(0, ARRAY_SIZE)]

    return _make


class TestGammaPunctual:
    point_1 = Point(1.0, 2.4)
    point_2 = Point(-3.2, 9.3)

    @pytest.mark.parametrize("point", [NULL_POINT, point_1])
    def test_same_points_is_zero(self, point):
        assert gamma_punctual(point, point) == 0

    def test_symmetric(self):
        assert gamma_punctual(self.point_1, self.point_2) == pytest.approx(
            gamma_punctual(self.point_2, self.point_1),
            abs=ATOL,
        )

    def test_known_value(self):
        KNOWN_RESULT = 230.00426083009856
        assert gamma_punctual(self.point_2, self.point_1) == pytest.approx(KNOWN_RESULT, abs=ATOL)


class TestGamma1d:

    def test_zeros(self, zeros_points):
        assert gamma_1d(zeros_points, zeros_points) == [0.0] * ARRAY_SIZE

    def test_constant_dose_offset(self, zeros_points, constant_dose_offset_points):
        assert gamma_1d(zeros_points, constant_dose_offset_points) == [
            pytest.approx(TEST_CONSTANT_DOSE / DOSE_TOLERANCE, abs=ATOL)] * ARRAY_SIZE

    @pytest.mark.parametrize("shift", [1, 3])
    def test_dta_ramp(self, dta_ramp_points, shift: int):
        ref = dta_ramp_points(0)
        evaluation = dta_ramp_points(shift)
        assert gamma_1d(ref, evaluation) == [pytest.approx(float(shift), abs=ATOL)] * ARRAY_SIZE

    def test_tradeoff(self):
        ref = Point(0.0, 0.0)
        close_pos_far_dose = Point(DTA, DOSE_TOLERANCE * 5)  # gamma ~5
        far_pos_close_dose = Point(DTA * 5, DOSE_TOLERANCE)  # gamma ~5
        close_pos_close_dose = Point(DTA, DOSE_TOLERANCE)  # Gamma ~1.4
        best_gamma: float = 1.4
        assert gamma_1d([ref], [close_pos_far_dose, far_pos_close_dose, close_pos_close_dose]) == [pytest.approx(
            best_gamma, abs=1e-1)]
