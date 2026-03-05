"""
Generates structured reports from one or many evaluation results.

Think of this like a dashboard analyst who takes raw scores
from multiple test runs and synthesizes them into actionable
reports — per-run summaries, suite-level trends, backend
comparisons, and prioritized improvement recommendations.

The reporter never runs tests or evaluates conversations.
It only reads completed EvaluationResults and produces reports.
This separation means reports can be regenerated any time
from stored results without re-running tests.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from src.lambdas.result_processor.evaluator import EvaluationResult


@dataclass
class RunReport:
    """
    Report for a single test run.
    Direct representation of one EvaluationResult as a formatted report.
    """
    test_id:        str
    persona:        str
    scenario:       str
    backend:        str
    grade:          str
    overall_score:  float
    summary:        str
    strengths:      list[str]
    weaknesses:     list[str]
    recommendations: list[str]
    fallout_points: list[str]
    scores:         dict        # dimension scores keyed by name
    generated_at:   str         = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SuiteReport:
    """
    Aggregate report across a full test suite run.
    Synthesizes patterns across multiple personas and scenarios.
    """
    suite_id:           str
    flow_type:          str
    total_runs:         int
    passed_runs:        int         # grade A or B
    failed_runs:        int         # grade D or F
    average_score:      float
    score_by_dimension: dict        # average score per dimension
    score_by_backend:   dict        # average score per backend
    common_weaknesses:  list[str]   # issues appearing in 3+ runs
    common_strengths:   list[str]   # strengths appearing in 3+ runs
    top_recommendations: list[str]  # highest priority improvements
    run_reports:        list[RunReport]
    generated_at:       str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BackendComparisonReport:
    """
    Side-by-side comparison of Anthropic vs Bedrock backend performance.
    Same scenarios, different backends — which produces better caller simulation?
    """
    flow_type:              str
    anthropic_avg_score:    float
    bedrock_avg_score:      float
    anthropic_avg_turns:    float   # lower is better — more efficient caller
    bedrock_avg_turns:      float
    dimension_comparison:   dict    # per-dimension score delta (bedrock - anthropic)
    recommendation:         str     # which backend performed better and why
    generated_at:           str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Single run report ──────────────────────────────────────────────────────

def generate_run_report(evaluation: EvaluationResult) -> RunReport:
    """
    Generate a formatted report for a single test run.
    Direct transformation of EvaluationResult into RunReport.
    """
    scores = {
        "goal_completion":      evaluation.goal_completion,
        "turn_efficiency":      evaluation.turn_efficiency,
        "intent_accuracy":      evaluation.intent_accuracy,
        "sentiment_trajectory": evaluation.sentiment_trajectory,
        "fallout_risk":         evaluation.fallout_risk
    }

    print(f"[Reporter] Run report generated — {evaluation.test_id} — grade: {evaluation.grade}")

    return RunReport(
        test_id         = evaluation.test_id,
        persona         = evaluation.persona,
        scenario        = evaluation.scenario,
        backend         = evaluation.backend,
        grade           = evaluation.grade,
        overall_score   = evaluation.overall_score,
        summary         = evaluation.summary,
        strengths       = evaluation.strengths,
        weaknesses      = evaluation.weaknesses,
        recommendations = evaluation.recommendations,
        fallout_points  = evaluation.fallout_points,
        scores          = scores
    )


# ── Suite report ───────────────────────────────────────────────────────────

def generate_suite_report(
    suite_id:    str,
    flow_type:   str,
    evaluations: list[EvaluationResult]
) -> SuiteReport:
    """
    Generate an aggregate report across a full test suite.

    Identifies patterns across all runs — which weaknesses appear
    repeatedly, which strengths are consistent, what the highest
    priority improvements are across the full suite.
    """
    if not evaluations:
        return _empty_suite_report(suite_id, flow_type)

    run_reports = [generate_run_report(e) for e in evaluations]

    # ── Score aggregation ──────────────────────────────────────
    total_runs    = len(evaluations)
    passed_runs   = sum(1 for e in evaluations if e.grade in ("A", "B"))
    failed_runs   = sum(1 for e in evaluations if e.grade in ("D", "F"))
    average_score = sum(e.overall_score for e in evaluations) / total_runs

    # Average per dimension across all runs
    score_by_dimension = {
        "goal_completion":      _avg(evaluations, "goal_completion"),
        "turn_efficiency":      _avg(evaluations, "turn_efficiency"),
        "intent_accuracy":      _avg(evaluations, "intent_accuracy"),
        "sentiment_trajectory": _avg(evaluations, "sentiment_trajectory"),
        "fallout_risk":         _avg(evaluations, "fallout_risk")
    }

    # Average per backend
    score_by_backend = _scores_by_backend(evaluations)

    # ── Pattern detection ──────────────────────────────────────
    # Find weaknesses and strengths appearing in 3+ runs
    common_weaknesses = _find_common_items(
        [e.weaknesses for e in evaluations],
        min_occurrences = max(2, total_runs // 3)
    )

    common_strengths = _find_common_items(
        [e.strengths for e in evaluations],
        min_occurrences = max(2, total_runs // 3)
    )

    # Collect all recommendations, deduplicate, prioritize by frequency
    top_recommendations = _prioritize_recommendations(
        [rec for e in evaluations for rec in e.recommendations]
    )

    print(f"[Reporter] Suite report generated — {suite_id} — {total_runs} runs, avg score: {average_score:.1f}")

    return SuiteReport(
        suite_id            = suite_id,
        flow_type           = flow_type,
        total_runs          = total_runs,
        passed_runs         = passed_runs,
        failed_runs         = failed_runs,
        average_score       = round(average_score, 2),
        score_by_dimension  = score_by_dimension,
        score_by_backend    = score_by_backend,
        common_weaknesses   = common_weaknesses,
        common_strengths    = common_strengths,
        top_recommendations = top_recommendations,
        run_reports         = run_reports
    )


# ── Backend comparison report ──────────────────────────────────────────────

def generate_backend_comparison(
    flow_type:   str,
    evaluations: list[EvaluationResult]
) -> BackendComparisonReport:
    """
    Compare Anthropic vs Bedrock backend performance across the same scenarios.

    Splits evaluations by backend, computes per-dimension deltas,
    and produces a recommendation on which backend performed better.
    """
    anthropic_evals = [e for e in evaluations if e.backend == "anthropic"]
    bedrock_evals   = [e for e in evaluations if e.backend == "bedrock"]

    if not anthropic_evals or not bedrock_evals:
        print(f"[Reporter] Cannot compare — need both backends. "
              f"Anthropic: {len(anthropic_evals)}, Bedrock: {len(bedrock_evals)}")
        return None

    anthropic_avg = sum(e.overall_score for e in anthropic_evals) / len(anthropic_evals)
    bedrock_avg   = sum(e.overall_score for e in bedrock_evals)   / len(bedrock_evals)

    # Per-dimension comparison — positive delta means Bedrock scored higher
    dimensions = [
        "goal_completion", "turn_efficiency",
        "intent_accuracy", "sentiment_trajectory", "fallout_risk"
    ]
    dimension_comparison = {
        dim: round(
            _avg(bedrock_evals, dim) - _avg(anthropic_evals, dim), 2
        )
        for dim in dimensions
    }

    # Which backend won overall?
    delta = bedrock_avg - anthropic_avg
    if abs(delta) < 0.5:
        recommendation = (
            f"Backends performed comparably (delta: {delta:+.2f}). "
            f"Prefer Bedrock for AWS-native deployment and IAM authentication."
        )
    elif delta > 0:
        recommendation = (
            f"Bedrock outperformed Anthropic by {delta:.2f} points. "
            f"Recommend Bedrock as primary backend."
        )
    else:
        recommendation = (
            f"Anthropic outperformed Bedrock by {abs(delta):.2f} points. "
            f"Evaluate whether Bedrock model selection can close the gap."
        )

    print(f"[Reporter] Backend comparison — Anthropic: {anthropic_avg:.1f}, Bedrock: {bedrock_avg:.1f}")

    return BackendComparisonReport(
        flow_type            = flow_type,
        anthropic_avg_score  = round(anthropic_avg, 2),
        bedrock_avg_score    = round(bedrock_avg, 2),
        anthropic_avg_turns  = _avg_turns(anthropic_evals),
        bedrock_avg_turns    = _avg_turns(bedrock_evals),
        dimension_comparison = dimension_comparison,
        recommendation       = recommendation
    )


# ── Serialization ──────────────────────────────────────────────────────────

def report_to_dict(report) -> dict:
    """
    Convert any report dataclass to a JSON-serializable dict.
    Used before storing to S3 or returning from Lambda.
    """
    import dataclasses
    return dataclasses.asdict(report)


def save_report_to_s3(report, bucket: str, key: str) -> bool:
    """Save a report to S3 as JSON."""
    import boto3
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket      = bucket,
            Key         = key,
            Body        = json.dumps(report_to_dict(report), indent=2),
            ContentType = "application/json"
        )
        print(f"[Reporter] Report saved — s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"[Reporter] S3 save failed: {e}")
        return False


# ── Helpers ────────────────────────────────────────────────────────────────

def _avg(evaluations: list[EvaluationResult], field_name: str) -> float:
    """Average a numeric field across a list of evaluations."""
    values = [getattr(e, field_name) for e in evaluations]
    return round(sum(values) / len(values), 2) if values else 0.0


def _avg_turns(evaluations: list[EvaluationResult]) -> float:
    """Average turns taken — pulled from step_results."""
    totals = []
    for e in evaluations:
        if e.step_results:
            totals.append(sum(s.turns_taken for s in e.step_results))
    return round(sum(totals) / len(totals), 1) if totals else 0.0


def _scores_by_backend(evaluations: list[EvaluationResult]) -> dict:
    """Group and average scores by backend name."""
    by_backend = {}
    for e in evaluations:
        if e.backend not in by_backend:
            by_backend[e.backend] = []
        by_backend[e.backend].append(e.overall_score)

    return {
        backend: round(sum(scores) / len(scores), 2)
        for backend, scores in by_backend.items()
    }


def _find_common_items(
    item_lists:      list[list[str]],
    min_occurrences: int = 2
) -> list[str]:
    """
    Find items that appear across multiple runs.
    Used to surface systemic issues vs one-off observations.

    Uses keyword matching rather than exact string match —
    different runs may phrase the same issue differently.
    """
    from collections import Counter

    # Flatten all items and count keyword occurrences
    all_items  = [item for sublist in item_lists for item in sublist]
    word_counts = Counter()

    for item in all_items:
        # Extract significant keywords (longer than 4 chars)
        words = [w.lower() for w in item.split() if len(w) > 4]
        word_counts.update(words)

    # Find items whose keywords appear frequently
    common = []
    seen_keywords = set()

    for item in all_items:
        words    = [w.lower() for w in item.split() if len(w) > 4]
        key_word = max(words, key=lambda w: word_counts[w]) if words else ""

        if key_word and word_counts[key_word] >= min_occurrences and key_word not in seen_keywords:
            common.append(item)
            seen_keywords.add(key_word)

    return common[:10]   # cap at 10 to keep reports readable


def _prioritize_recommendations(recommendations: list[str]) -> list[str]:
    """
    Deduplicate and prioritize recommendations by frequency.
    Recommendations appearing across multiple runs are ranked higher.
    """
    from collections import Counter

    if not recommendations:
        return []

    # Score each recommendation by keyword frequency
    word_counts = Counter()
    for rec in recommendations:
        words = [w.lower() for w in rec.split() if len(w) > 4]
        word_counts.update(words)

    def score_rec(rec: str) -> float:
        words = [w.lower() for w in rec.split() if len(w) > 4]
        return sum(word_counts[w] for w in words) / max(len(words), 1)

    # Deduplicate by keyword similarity, rank by score
    seen      = set()
    scored    = []

    for rec in recommendations:
        key   = " ".join(sorted(set(w.lower() for w in rec.split() if len(w) > 4)))
        if key not in seen:
            seen.add(key)
            scored.append((score_rec(rec), rec))

    scored.sort(reverse=True)
    return [rec for _, rec in scored[:5]]


def _empty_suite_report(suite_id: str, flow_type: str) -> SuiteReport:
    """Return an empty suite report when no evaluations provided."""
    return SuiteReport(
        suite_id            = suite_id,
        flow_type           = flow_type,
        total_runs          = 0,
        passed_runs         = 0,
        failed_runs         = 0,
        average_score       = 0.0,
        score_by_dimension  = {},
        score_by_backend    = {},
        common_weaknesses   = [],
        common_strengths    = [],
        top_recommendations = [],
        run_reports         = []
    )
