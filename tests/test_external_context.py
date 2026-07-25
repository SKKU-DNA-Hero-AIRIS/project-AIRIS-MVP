import unittest

from external_context import (
    air_quality_level,
    build_facility_context,
    infer_facility_type,
)


class ExternalContextTests(unittest.TestCase):
    def test_air_quality_boundaries(self):
        self.assertEqual(air_quality_level(15), "GOOD")
        self.assertEqual(air_quality_level(35), "MODERATE")
        self.assertEqual(air_quality_level(35.1), "BAD")

    def test_hospital_is_high_risk(self):
        context = build_facility_context("병원", True)
        self.assertEqual(context["risk_level"], "HIGH")
        self.assertTrue(context["vulnerable_population"])

    def test_facility_is_inferred_from_name(self):
        self.assertEqual(infer_facility_type("성균관대학교"), "학교")
        self.assertEqual(infer_facility_type("서울대학교병원"), "병원")
        self.assertEqual(infer_facility_type("행복어린이집"), "어린이집")


if __name__ == "__main__":
    unittest.main()
