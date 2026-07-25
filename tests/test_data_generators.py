"""
data/ 생성기 회귀 테스트.

각 테스트는 코드 리뷰에서 발견된 결함에 대응한다. 수정 전에는 실패하고
수정 후에 통과해야 한다.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

import air_gate as ag          # noqa: E402
import health_synth as hs      # noqa: E402


class AmbientBiasTests(unittest.TestCase):
    """
    적외선 체온계의 심부온 환산식은 ir_calibration_ambient(22℃) 기준으로
    고정되어 있다. 주변온도가 그와 다르면 환산 자체가 틀어져야 한다.
    이 편향이 빠지면 주변온도 보정 실험의 전제가 무너진다.
    """

    def setUp(self):
        self.people = hs.generate(300, seed=11)
        self.masked = np.zeros(len(self.people), dtype=bool)
        self.walk = np.full(len(self.people), 1.1)
        self.cfg = ag.GateSensorConfig()

    def _measure_at(self, ambient: float) -> float:
        """같은 생리값·같은 난수로 주변온도만 바꿔 측정한다."""
        env = {"temperature": ambient, "humidity": 50.0, "pm25": 20.0}
        out = ag.measure(self.people, env, self.masked, 0.0, self.walk,
                         self.cfg, np.random.default_rng(3))
        return float(np.nanmean(out["body_temperature"].to_numpy(dtype=float)))

    def test_measured_temperature_shifts_with_ambient(self):
        """생리값이 동일해도 주변온도가 다르면 판독값이 달라져야 한다."""
        cold, hot = self._measure_at(10.0), self._measure_at(34.0)
        self.assertGreater(hot, cold,
                           "주변온도가 24℃ 높은데 판독체온이 오르지 않았다")

    def test_bias_magnitude_matches_declared_sensitivity(self):
        """편향 크기가 설정된 민감도와 일치해야 한다 (선언과 구현의 일치)."""
        delta_ambient = 34.0 - 10.0
        observed = self._measure_at(34.0) - self._measure_at(10.0)
        expected = self.cfg.ir_skin_ambient_sensitivity * delta_ambient
        self.assertAlmostEqual(observed, expected, delta=0.05)

    def test_calibration_point_has_no_bias(self):
        """보정 기준 온도에서는 편향이 0이어야 한다."""
        env = {"temperature": self.cfg.ir_calibration_ambient,
               "humidity": 50.0, "pm25": 20.0}
        out = ag.measure(self.people, env, self.masked, 0.0, self.walk,
                         self.cfg, np.random.default_rng(3))
        recorded = out["_quality"].iloc[0]["ir_ambient_bias_c"]
        self.assertAlmostEqual(recorded, 0.0, places=6)

    def test_recorded_bias_matches_applied_bias(self):
        """품질 메타데이터의 ir_ambient_bias_c 가 실제 적용값과 같아야 한다."""
        ambient = 34.0
        env = {"temperature": ambient, "humidity": 50.0, "pm25": 20.0}
        out = ag.measure(self.people, env, self.masked, 0.0, self.walk,
                         self.cfg, np.random.default_rng(3))
        recorded = out["_quality"].iloc[0]["ir_ambient_bias_c"]
        expected = self.cfg.ir_skin_ambient_sensitivity * (
            ambient - self.cfg.ir_calibration_ambient)
        self.assertAlmostEqual(recorded, round(expected, 2), places=6)


class TimeseriesStateDistributionTests(unittest.TestCase):
    """
    사용자가 지정한 상태분포는 시계열 끝까지 유지되어야 한다.
    전이행렬이 DEFAULT_STATE_PROBS 로 고정되면 첫 시점 이후 기본값으로
    수렴해, 유행 시나리오가 조용히 평시로 되돌아간다.
    """

    def _febrile_rate(self, df, step):
        g = df[df["t"] == step] if "t" in df else df
        return float((g["latent_state"] == hs.State.FEBRILE_INFECTION.value).mean())

    def test_custom_state_probs_persist_to_last_step(self):
        cfg = hs.PopulationConfig(state_probs={
            hs.State.NORMAL: 0.10,
            hs.State.FEBRILE_INFECTION: 0.90,
        })
        df = hs.generate_timeseries(n_subjects=250, n_steps=96, cfg=cfg, seed=5)
        step_col = "t" if "t" in df.columns else df.columns[df.columns.str.contains("step|time", case=False)][0]
        first = float((df[df[step_col] == df[step_col].min()]["latent_state"]
                       == hs.State.FEBRILE_INFECTION.value).mean())
        last = float((df[df[step_col] == df[step_col].max()]["latent_state"]
                      == hs.State.FEBRILE_INFECTION.value).mean())
        self.assertGreater(first, 0.80, "첫 시점부터 지정 분포가 반영되지 않았다")
        self.assertGreater(last, 0.80,
                           f"마지막 시점 발열 비율이 {last:.1%} 로 떨어졌다 "
                           "— 전이행렬이 기본 분포로 수렴하고 있다")

    def test_transition_matrix_stationary_equals_given_probs(self):
        """T = (1-p)·1·baseᵀ + p·I 의 정상분포는 정확히 base 여야 한다."""
        labels = [hs.State.NORMAL, hs.State.FEBRILE_INFECTION]
        probs = {hs.State.NORMAL: 0.3, hs.State.FEBRILE_INFECTION: 0.7}
        T = hs._transition_matrix(labels, state_probs=probs)
        power = np.linalg.matrix_power(T, 4000)
        np.testing.assert_allclose(power[0], [0.3, 0.7], atol=1e-6)

    def test_default_is_unchanged_when_probs_omitted(self):
        """인자를 생략하면 기존 동작(기본 분포)을 유지해야 한다."""
        labels = list(hs.DEFAULT_STATE_PROBS.keys())
        expected = np.array([hs.DEFAULT_STATE_PROBS[s] for s in labels])
        expected = expected / expected.sum()
        power = np.linalg.matrix_power(hs._transition_matrix(labels), 4000)
        np.testing.assert_allclose(power[0], expected, atol=1e-6)


class FlatCsvExportTests(unittest.TestCase):
    """README 가 산출물로 설명하는 CSV 가 실제로 만들어져야 한다."""

    def test_measurement_impact_matches_documented_schema(self):
        recs = ag.generate_gate_records(facility_type="school", n_records=25, seed=2)
        flat = ag.measurement_impact(recs)
        documented = ["timestamp", "ambient_temp", "pm25", "crowding", "masked",
                      "ir_bias", "n_valid", "meas_temp", "true_temp", "meas_hr",
                      "true_hr", "meas_spo2", "true_spo2", "latent_state",
                      "is_abnormal", "on_beta_blocker", "is_elderly"]
        self.assertEqual(list(flat.columns), documented)
        self.assertGreater(len(flat), 0)


if __name__ == "__main__":
    unittest.main()
