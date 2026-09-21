"""Fail-closed provenance for source-paper-native baseline comparisons.

Disabling a repository repair is not evidence of a faithful reproduction.
This registry records the audit independently of numerical fit quality; neither
a low residual nor a method name can promote an adaptation to a native method.
No verified source-paper implementation is currently registered.
"""

from __future__ import annotations

from dataclasses import dataclass


BASELINE_PROTOCOLS = ("adaptation", "native")
NATIVE_AUDIT_VERSION = "source_paper_audit_20260921_v1"


@dataclass(frozen=True)
class _Provenance:
    doi: str | None
    source_url: str | None
    source_access: str
    departures: tuple[str, ...]
    author_code_url: str | None = None
    author_code_sha256: str | None = None
    reproduction_status: str = "reproduction_unverified"


_PROVENANCE = {
    "park_dominant_point_2007_adaptation": _Provenance(
        "10.1016/j.cad.2006.12.006",
        "https://www.sciencedirect.com/science/article/pii/S0010448507000024",
        "publisher abstract/introduction; full implementation not verified",
        (
            "Common-MSE stopping replaces the source error-bound protocol; no iterative orthogonal-distance parameter update.",
            "Degree-compatible shape-seed completion, endpoint knot clamps, and endpoint-constrained refits are repository choices, not independently source-verified.",
            "Discrete Menger curvature and incremental LCM-seed consumption have not been checked against an author-code reference.",
        ),
    ),
    "liang_feature_iki_2017_adaptation": _Provenance(
        "10.1088/1361-6501/aa6a05",
        "https://doi.org/10.1088/1361-6501/aa6a05",
        "abstract/preview only; exact feature equation and IKI details unavailable",
        (
            "The normalized arc/unsigned-turning-angle feature blend (default 0.5) is a repository construction, not a verified original equation.",
            "Sampled feature-CDF inversion and worst-residual span bisection replace unverified analytic-feature/IKI details.",
            "Common-MSE stopping, dense/initial knot counts, and endpoint-constrained fits are comparison settings.",
        ),
    ),
    "dung_direct_knot_2017_adaptation": _Provenance(
        "10.1371/journal.pone.0173857",
        "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173857",
        "full primary article and official S1 MATLAB source inspected; not executed",
        (
            "Only simple knots are optimized; the author's multiplicity 1..degree+1 and joining-angle/continuity classifier are absent.",
            "Only serial splitting is implemented; parallel split/join/shift is absent (serial is itself a published variant).",
            "Capacity overflow uniformly discards coarse breaks; this is not an author-code operation.",
            "The final endpoint-constrained refit changes the author's unconstrained least-squares spline.",
            "sqrt(mse_tolerance) is not an equivalent native maximum-error constraint; finite-difference solver/search and short-tail guards are repository choices.",
        ),
        author_code_url="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0173857.s006&type=supplementary",
        author_code_sha256="2fc1830966f71ce77f7c29e33ecf321b5fddca5f5bfca75458d667a1d1ce0399",
    ),
    "kang_sparse_2015_adaptation": _Provenance(
        "10.1016/j.cad.2014.08.022",
        "https://www.sciencedirect.com/science/article/pii/S0010448514001912",
        "full primary article inspected in existing tmp/Kang2015.pdf",
        (
            "Scalar constrained l1 optimization is changed to vector group-l1 with approximate ADMM/penalty bisection instead of the source CVX solve.",
            "Algorithms 1/3 (repeated convex solves for general-data relocation) are absent; long active runs are returned without that stage.",
            "The degree+1 cluster-size gate, 1.25 spacing factor, singleton interval search, and absolute/relative jump cutoffs are repository choices.",
            "The comparison wrapper replaces the final unconstrained spline by an endpoint-constrained refit.",
        ),
    ),
    "luo_linf_de_2022_adaptation": _Provenance(
        "10.4208/jcm.2012-m2020-0203",
        "https://www.global-sci.com/JCM/article/view/12502",
        "publisher abstract and preview; full source/author-code equivalence not verified",
        (
            "Regularization is selected by a common-MSE search instead of a verified source-paper parameter protocol.",
            "DE fitness and the reported spline use endpoint-constrained least squares; a source-native solver/output is not preserved.",
            "Fallback to the largest jump when no peak survives, minimum-gap projections, random population initialization, and finite DE budgets are unverified implementation choices.",
            "The ADMM realization, local-maxima edge cases, and DE variant have no author-code/paper-example equivalence validation.",
        ),
    ),
    "yeh_feature_cdf_2020": _Provenance(
        "10.1016/j.cad.2020.102905",
        "https://www.cs.purdue.edu/homes/xmt/papers/Knot-Placement_CAD_2020.pdf",
        "supplementary method outside the current five-paper audit",
        (
            "Ascending cardinality scan replaces source target-error regression; reported fit is endpoint-constrained.",
            "No source-native implementation is registered for this method.",
        ),
    ),
    "uniform_gradient_pruning": _Provenance(
        None,
        None,
        "repository numerical control, not a cited original algorithm",
        ("This is a repository control, not a source-paper-native baseline.",),
        reproduction_status="repository_control",
    ),
}


def baseline_provenance(method: str) -> dict[str, object]:
    """Return a fresh JSON-safe audit record; unknown names fail closed.

    ``native_comparison_available`` means that the source-native solver and
    output can actually be run here, not merely that a paper or code exists.
    The returned lists are copies so callers cannot mutate the audit registry.
    """

    if method not in _PROVENANCE:
        raise ValueError(f"unknown baseline method: {method!r}")
    record = _PROVENANCE[method]
    return {
        "audit_version": NATIVE_AUDIT_VERSION,
        "method": method,
        "reproduction_status": record.reproduction_status,
        "native_available": False,
        "native_comparison_available": False,
        "implementation_origin": "repository_implementation",
        "author_code_executed": False,
        "reference_doi": record.doi,
        "source_url": record.source_url,
        "source_access": record.source_access,
        "author_code_url": record.author_code_url,
        "author_code_sha256": record.author_code_sha256,
        "known_departures": list(record.departures),
        "audit_document": "docs/baseline_native_audit.md",
    }


class NativeBaselineUnavailableError(ValueError):
    """A requested source-paper-native method has no verified runnable adapter."""

    def __init__(self, method: str, provenance: dict[str, object]) -> None:
        self.method = method
        self.provenance = provenance
        super().__init__(
            f"Native comparison unavailable for {method!r}: "
            f"{provenance['reproduction_status']}. Disabling repairs does not "
            "restore the source algorithm. See docs/baseline_native_audit.md."
        )


def validate_baseline_protocol(method: str, protocol: str) -> dict[str, object]:
    """Validate before parameterization, searches, refits, or timing warmups.

    Native requests never fall back to an adaptation. A caller may preserve an
    unavailable row with null fit/time metrics and this exception's provenance;
    it must not attribute an implementation gap to the paper's performance.
    """

    if protocol not in BASELINE_PROTOCOLS:
        raise ValueError(f"baseline_protocol must be one of {BASELINE_PROTOCOLS}")
    provenance = baseline_provenance(method)
    if protocol == "native" and not provenance["native_comparison_available"]:
        raise NativeBaselineUnavailableError(method, provenance)
    return provenance


__all__ = [
    "BASELINE_PROTOCOLS",
    "NATIVE_AUDIT_VERSION",
    "NativeBaselineUnavailableError",
    "baseline_provenance",
    "validate_baseline_protocol",
]
