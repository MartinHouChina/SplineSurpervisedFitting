from __future__ import annotations

# ruff: noqa: E402

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_sparse_knot_paper import build_cases, run_reproduction


class SparseKnotPaperReproductionTests(unittest.TestCase):
    def test_full_chebyshev_case_uses_paper_experiment_settings(self) -> None:
        chebyshev = build_cases(quick=False)[0]

        self.assertEqual(chebyshev.name, "chebyshev_t10")
        self.assertEqual(chebyshev.parameters.numel(), 401)
        self.assertEqual(chebyshev.initial_internal_knots, 25)
        self.assertEqual(chebyshev.epsilon, 0.003)
        self.assertEqual(chebyshev.paper_reference_final_knot_count, 14)
        self.assertEqual(chebyshev.paper_reference_mse, 3.4745e-5)

    def test_quick_run_writes_two_json_and_png_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            reports = run_reproduction(output_dir, quick=True, dpi=72)

            self.assertEqual(len(reports), 2)
            for name in ("chebyshev_t10", "known_cubic_bspline"):
                json_path = output_dir / f"{name}.json"
                png_path = output_dir / f"{name}.png"
                self.assertTrue(json_path.is_file())
                self.assertGreater(png_path.stat().st_size, 1_000)
                report = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(report["case"], name)
                self.assertEqual(report["dimension"], 1)
                self.assertIn("not the paper's CVX solve", report["reproduction_scope"])
                self.assertIn("mse", report["sparse_stage"])
                self.assertIn("internal_knots", report["final_standard_bspline_refit"])


if __name__ == "__main__":
    unittest.main()
