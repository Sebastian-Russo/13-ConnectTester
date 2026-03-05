"""
The conductor of the full test suite.

Think of this like a test lab manager who receives a test plan,
assigns each test to a worker, runs them all simultaneously,
collects results as they come in, and hands everything to the
reporter for final synthesis.

The orchestrator knows nothing about phone calls, AI backends,
or scoring logic. It only knows about coordinating work across
multiple parallel test runs and collecting results.

This is the main entry point for everything — Lambda handlers,
the local Flask server, and future API integrations all call
this file directly.
"""

import time
import uuid
import concurrent.futures
from dataclasses import dataclass, field
from src.backends.anthropic_backend  import AnthropicBackend
from src.backends.bedrock_backend    import BedrockBackend
from src.backends.base_backend       import BaseBackend
from src.personas.persona_generator  import Persona, generate_persona_batch
from src.personas.scenario_generator import Scenario, generate_scenario_batch
from src.lambdas.test_runner.caller_agent  import get_opening_utterance, respond_to_connect
from src.lambdas.test_runner.voice_caller  import TwilioVoiceCaller
from src.lambdas.test_runner.audio_bridge  import AudioBridge
from src.lambdas.result_processor.evaluator import evaluate_test_run, EvaluationResult
from src.lambdas.report_generator.reporter  import (
    generate_suite_report, generate_backend_comparison,
    save_report_to_s3, report_to_dict
)
from src.infrastructure.config import (
    MAX_PARALLEL_TESTS,
    RESULTS_BUCKET,
    RESULTS_TABLE
)


@dataclass
class TestSuiteConfig:
    """
    Full configuration for a test suite run.
    Everything the orchestrator needs to generate personas,
    scenarios, run tests, and produce reports.
    """
    flow_type:      str                    # e.g. "hotel_reservation"
    flow_context:   str                    # what the flow can handle
    stream_url:     str                    # WebSocket URL for Twilio media stream
    persona_count:  int         = 5        # how many personas to generate
    backends:       list[str]   = field(default_factory=lambda: ["anthropic", "bedrock"])
    trait_matrix:   list[list[str]] = field(default_factory=list)   # persona trait distribution
    tag_matrix:     list[list[str]] = field(default_factory=list)   # scenario tag distribution
    max_parallel:   int         = MAX_PARALLEL_TESTS


@dataclass
class SuiteRun:
    """
    A complete test suite run — personas, scenarios, results, reports.
    """
    suite_id:    str
    config:      TestSuiteConfig
    personas:    list[Persona]    = field(default_factory=list)
    scenarios:   list[Scenario]   = field(default_factory=list)
    evaluations: list[EvaluationResult] = field(default_factory=list)
    status:      str = "pending"    # pending | running | completed | failed
    started_at:  float = field(default_factory=time.time)
    duration_seconds: float = 0.0


class Orchestrator:
    """
    Coordinates the full test suite lifecycle.

    One Orchestrator instance per suite run. Manages persona
    generation, scenario generation, parallel test execution,
    evaluation, and report generation.
    """

    def __init__(self, config: TestSuiteConfig):
        self.config   = config
        self.suite_id = f"suite-{config.flow_type}-{uuid.uuid4().hex[:8]}"

        # Initialize both backends — suite runs tests on all configured backends
        self.backends: dict[str, BaseBackend] = {}
        if "anthropic" in config.backends:
            self.backends["anthropic"] = AnthropicBackend()
        if "bedrock" in config.backends:
            self.backends["bedrock"] = BedrockBackend()

        # Use the first available backend for persona/scenario generation
        # These don't need to run on all backends — just need one good backend
        self.generation_backend = list(self.backends.values())[0]

        self.run = SuiteRun(suite_id=self.suite_id, config=config)

    def execute(self) -> dict:
        """
        Run the full test suite end to end.

        Steps:
        1. Generate personas
        2. Generate scenarios
        3. Run all tests in parallel across all backends
        4. Evaluate all results
        5. Generate suite report and backend comparison
        6. Store everything to S3

        Returns the complete suite report as a dict.
        """
        print(f"\n[Orchestrator] Starting suite — ID: {self.suite_id}")
        self.run.status = "running"

        try:
            # ── Step 1: Generate personas ──────────────────────
            print(f"[Orchestrator] Generating {self.config.persona_count} personas...")
            self.run.personas = generate_persona_batch(
                backend       = self.generation_backend,
                scenario_type = self.config.flow_type,
                count         = self.config.persona_count,
                trait_matrix  = self.config.trait_matrix or self._default_trait_matrix()
            )

            # ── Step 2: Generate scenarios ─────────────────────
            print(f"[Orchestrator] Generating scenarios...")
            self.run.scenarios = generate_scenario_batch(
                backend      = self.generation_backend,
                personas     = self.run.personas,
                flow_type    = self.config.flow_type,
                flow_context = self.config.flow_context,
                tag_matrix   = self.config.tag_matrix or self._default_tag_matrix()
            )

            # ── Step 3: Run tests in parallel ──────────────────
            # Each persona+scenario runs on every configured backend
            # Total test runs = personas × backends
            test_configs = [
                (persona, scenario, backend_name, backend)
                for (persona, scenario), (backend_name, backend)
                in [
                    ((p, s), (bn, b))
                    for p, s in zip(self.run.personas, self.run.scenarios)
                    for bn, b in self.backends.items()
                ]
            ]

            print(f"[Orchestrator] Running {len(test_configs)} tests "
                  f"({self.config.persona_count} personas × {len(self.backends)} backends) "
                  f"with max {self.config.max_parallel} parallel...")

            test_results = self._run_parallel(test_configs)

            # ── Step 4: Evaluate all results ───────────────────
            print(f"[Orchestrator] Evaluating {len(test_results)} results...")
            self.run.evaluations = self._evaluate_all(test_results)

            # ── Step 5: Generate reports ───────────────────────
            suite_report = generate_suite_report(
                suite_id    = self.suite_id,
                flow_type   = self.config.flow_type,
                evaluations = self.run.evaluations
            )

            comparison_report = None
            if len(self.backends) > 1:
                comparison_report = generate_backend_comparison(
                    flow_type   = self.config.flow_type,
                    evaluations = self.run.evaluations
                )

            # ── Step 6: Store reports ──────────────────────────
            self._store_reports(suite_report, comparison_report)

            self.run.status           = "completed"
            self.run.duration_seconds = round(time.time() - self.run.started_at, 1)

            print(f"\n[Orchestrator] Suite complete — "
                  f"{self.run.duration_seconds}s — "
                  f"avg score: {suite_report.average_score:.1f}")

            # Return full output
            output = {
                "suite_id":     self.suite_id,
                "suite_report": report_to_dict(suite_report),
                "duration":     self.run.duration_seconds,
                "total_runs":   suite_report.total_runs,
                "avg_score":    suite_report.average_score
            }

            if comparison_report:
                output["backend_comparison"] = report_to_dict(comparison_report)

            return output

        except Exception as e:
            self.run.status = "failed"
            print(f"[Orchestrator] Suite failed: {e}")
            raise

    def _run_parallel(self, test_configs: list) -> list[dict]:
        """
        Run all test configurations in parallel using ThreadPoolExecutor.

        Each test is one complete voice call — persona dials Connect,
        has the conversation, hangs up. All run simultaneously up to
        max_parallel limit.

        One test failing doesn't stop the others.
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_parallel
        ) as executor:

            future_to_config = {
                executor.submit(
                    self._run_single_test,
                    persona,
                    scenario,
                    backend_name,
                    backend
                ): (persona, scenario, backend_name)
                for persona, scenario, backend_name, backend in test_configs
            }

            for future in concurrent.futures.as_completed(future_to_config):
                persona, scenario, backend_name = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[Orchestrator] ✓ {persona.name} / {backend_name} complete")
                except Exception as e:
                    print(f"[Orchestrator] ✗ {persona.name} / {backend_name} failed: {e}")
                    results.append(self._failed_result(persona, scenario, backend_name, str(e)))

        return results

    def _run_single_test(
        self,
        persona:      Persona,
        scenario:     Scenario,
        backend_name: str,
        backend:      BaseBackend
    ) -> dict:
        """
        Run a single voice test — one persona, one scenario, one backend.
        Imports handler logic directly to avoid Lambda invocation overhead
        when running locally.
        """
        from src.lambdas.test_runner.handler import _run_voice_test

        audio_bridge = AudioBridge(emotional_state=persona.emotional_state)
        voice_caller = TwilioVoiceCaller(stream_url=self.config.stream_url)

        return _run_voice_test(
            persona      = persona,
            scenario     = scenario,
            backend      = backend,
            audio_bridge = audio_bridge,
            voice_caller = voice_caller
        )

    def _evaluate_all(self, test_results: list[dict]) -> list[EvaluationResult]:
        """
        Evaluate all test results using the evaluation backend.
        Runs sequentially — evaluations are cheaper than full test runs.
        """
        evaluations = []

        for result in test_results:
            # Find the matching scenario for context
            scenario = next(
                (s for s in self.run.scenarios if s.name == result.get("scenario")),
                self.run.scenarios[0] if self.run.scenarios else None
            )

            if not scenario:
                print(f"[Orchestrator] No matching scenario for {result.get('test_id')} — skipping")
                continue

            evaluation = evaluate_test_run(
                test_result = result,
                scenario    = scenario,
                backend     = self.generation_backend
            )
            evaluations.append(evaluation)
            print(f"[Orchestrator] Evaluated {evaluation.test_id} — score: {evaluation.overall_score}")

        return evaluations

    def _store_reports(self, suite_report, comparison_report) -> None:
        """Store suite report and comparison report to S3."""
        if not RESULTS_BUCKET:
            print(f"[Orchestrator] No RESULTS_BUCKET configured — skipping S3 storage")
            return

        save_report_to_s3(
            report = suite_report,
            bucket = RESULTS_BUCKET,
            key    = f"reports/{self.suite_id}/suite_report.json"
        )

        if comparison_report:
            save_report_to_s3(
                report = comparison_report,
                bucket = RESULTS_BUCKET,
                key    = f"reports/{self.suite_id}/backend_comparison.json"
            )

    def _default_trait_matrix(self) -> list[list[str]]:
        """
        Default trait distribution when none specified.
        Ensures persona variety across common customer types.
        """
        return [
            ["calm", "patient"],
            ["frustrated", "impatient"],
            ["confused", "elderly"],
            ["urgent", "tech-savvy"],
            ["casual", "first-time-caller"]
        ][:self.config.persona_count]

    def _default_tag_matrix(self) -> list[list[str]]:
        """
        Default scenario tag distribution when none specified.
        Ensures coverage of happy paths and edge cases.
        """
        return [
            ["happy_path"],
            ["edge_case", "angry_caller"],
            ["off_topic", "confused"],
            ["repeat_caller", "specific_request"],
            ["happy_path", "upsell_opportunity"]
        ][:self.config.persona_count]

    def _failed_result(
        self,
        persona:      Persona,
        scenario:     Scenario,
        backend_name: str,
        reason:       str
    ) -> dict:
        """Build a failed test result when a test run throws an exception."""
        return {
            "test_id":         f"{persona.name}-{scenario.name}-{backend_name}-failed",
            "persona":         persona.name,
            "scenario":        scenario.name,
            "backend":         backend_name,
            "transcript":      [],
            "steps_completed": 0,
            "total_steps":     len(scenario.steps),
            "total_turns":     0,
            "end_reason":      reason,
            "call_duration":   0,
            "status":          "failed"
        }
