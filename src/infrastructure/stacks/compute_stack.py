"""
Compute infrastructure — Lambda functions and API Gateway.

Three Lambda functions:
- test_runner:       executes voice tests, manages Twilio WebSocket stream
- result_processor:  evaluates completed test transcripts
- report_generator:  generates suite and comparison reports on demand

One API Gateway with both REST and WebSocket APIs:
- REST API:       triggers test runs, generates personas, serves reports
- WebSocket API:  handles Twilio media stream for live voice calls

All Lambdas share the same code bundle (src/ + lambdas/) but
have different handlers, memory, and timeout configurations
based on their workload.
"""

from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_apigatewayv2 as apigwv2,
    aws_iam as iam,
    aws_logs as logs,
    CfnOutput
)
from constructs import Construct
from src.infrastructure.stacks.storage_stack import StorageStack


class ComputeStack(Stack):

    def __init__(
        self,
        scope:         Construct,
        construct_id:  str,
        storage_stack: StorageStack,
        **kwargs
    ):
        super().__init__(scope, construct_id, **kwargs)

        # ── Shared Lambda layer ────────────────────────────────
        # Python dependencies shared across all Lambda functions
        # Packaged separately from code to speed up deployments
        # Code changes don't require re-uploading dependencies
        self.deps_layer = lambda_.LayerVersion(
            self, "DependenciesLayer",
            code                = lambda_.Code.from_asset("lambda_layers/dependencies"),
            compatible_runtimes = [lambda_.Runtime.PYTHON_3_12],
            description         = "Shared Python dependencies: anthropic, boto3, twilio, pyyaml"
        )

        # ── Shared IAM role ────────────────────────────────────
        self.lambda_role = self._create_lambda_role(storage_stack)

        # ── Lambda functions ───────────────────────────────────
        self.test_runner      = self._create_test_runner(storage_stack)
        self.result_processor = self._create_result_processor(storage_stack)
        self.report_generator = self._create_report_generator(storage_stack)

        # ── API Gateway REST ───────────────────────────────────
        self.rest_api = self._create_rest_api()

        # ── API Gateway WebSocket ──────────────────────────────
        self.ws_api = self._create_websocket_api()

        # ── Outputs ────────────────────────────────────────────
        CfnOutput(
            self, "RestApiUrl",
            value       = self.rest_api.url,
            export_name = "ConnectTesterRestApi",
            description = "REST API endpoint for triggering tests"
        )

        CfnOutput(
            self, "WebSocketApiUrl",
            value       = self.ws_api.attr_api_endpoint,
            export_name = "ConnectTesterWebSocketApi",
            description = "WebSocket endpoint for Twilio media stream"
        )

    # ── IAM ────────────────────────────────────────────────────────────────

    def _create_lambda_role(self, storage_stack: StorageStack) -> iam.Role:
        """
        Shared IAM role for all Lambda functions.
        Grants least-privilege access to every service the Lambdas need.
        """
        role = iam.Role(
            self, "LambdaRole",
            assumed_by  = iam.ServicePrincipal("lambda.amazonaws.com"),
            description = "Shared role for Connect Tester Lambda functions"
        )

        # CloudWatch Logs — basic Lambda execution
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )

        # S3 — read/write results and scenarios
        role.add_to_policy(iam.PolicyStatement(
            actions   = [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            resources = [
                storage_stack.results_bucket.bucket_arn,
                f"{storage_stack.results_bucket.bucket_arn}/*",
                storage_stack.scenarios_bucket.bucket_arn,
                f"{storage_stack.scenarios_bucket.bucket_arn}/*"
            ]
        ))

        # DynamoDB — read/write results table and GSIs
        role.add_to_policy(iam.PolicyStatement(
            actions   = [
                "dynamodb:PutItem",
                "dynamodb:GetItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            resources = [
                storage_stack.results_table.table_arn,
                f"{storage_stack.results_table.table_arn}/index/*"
            ]
        ))

        # Bedrock — invoke Claude models
        role.add_to_policy(iam.PolicyStatement(
            actions   = ["bedrock:InvokeModel"],
            resources = ["arn:aws:bedrock:*::foundation-model/*"]
        ))

        # Polly — synthesize speech for caller audio
        role.add_to_policy(iam.PolicyStatement(
            actions   = ["polly:SynthesizeSpeech"],
            resources = ["*"]
        ))

        # Transcribe — transcribe Connect audio to text
        role.add_to_policy(iam.PolicyStatement(
            actions   = [
                "transcribe:StartTranscriptionJob",
                "transcribe:GetTranscriptionJob"
            ],
            resources = ["*"]
        ))

        # Secrets Manager — Twilio credentials, Anthropic API key
        # Secrets stored at connect-tester/* prefix
        role.add_to_policy(iam.PolicyStatement(
            actions   = ["secretsmanager:GetSecretValue"],
            resources = ["arn:aws:secretsmanager:*:*:secret:connect-tester/*"]
        ))

        return role

    # ── Lambda functions ───────────────────────────────────────────────────

    def _create_test_runner(self, storage_stack: StorageStack) -> lambda_.Function:
        """
        Test runner Lambda — executes voice tests and manages WebSocket stream.

        1GB memory: audio processing and parallel AI calls are memory-intensive
        10 min timeout: voice calls can run up to MAX_CALL_DURATION + buffer
        """
        return lambda_.Function(
            self, "TestRunner",
            function_name = "connect-tester-test-runner",
            runtime       = lambda_.Runtime.PYTHON_3_12,
            handler       = "src.lambdas.test_runner.handler.lambda_handler",
            code          = lambda_.Code.from_asset("."),
            layers        = [self.deps_layer],
            role          = self.lambda_role,
            memory_size   = 1024,
            timeout       = Duration.minutes(10),
            log_retention = logs.RetentionDays.ONE_MONTH,
            environment   = {
                "RESULTS_BUCKET":   storage_stack.results_bucket.bucket_name,
                "SCENARIOS_BUCKET": storage_stack.scenarios_bucket.bucket_name,
                "RESULTS_TABLE":    storage_stack.results_table.table_name,
                "ENVIRONMENT":      "production"
            }
        )

    def _create_result_processor(self, storage_stack: StorageStack) -> lambda_.Function:
        """
        Result processor Lambda — evaluates completed test transcripts.

        512MB memory: LLM calls only, no audio processing
        5 min timeout: evaluation takes 30-60 seconds per run
        """
        return lambda_.Function(
            self, "ResultProcessor",
            function_name = "connect-tester-result-processor",
            runtime       = lambda_.Runtime.PYTHON_3_12,
            handler       = "src.lambdas.result_processor.handler.lambda_handler",
            code          = lambda_.Code.from_asset("."),
            layers        = [self.deps_layer],
            role          = self.lambda_role,
            memory_size   = 512,
            timeout       = Duration.minutes(5),
            log_retention = logs.RetentionDays.ONE_MONTH,
            environment   = {
                "RESULTS_BUCKET": storage_stack.results_bucket.bucket_name,
                "RESULTS_TABLE":  storage_stack.results_table.table_name,
                "ENVIRONMENT":    "production"
            }
        )

    def _create_report_generator(self, storage_stack: StorageStack) -> lambda_.Function:
        """
        Report generator Lambda — builds suite and comparison reports on demand.

        256MB memory: data aggregation only, no heavy compute
        2 min timeout: reports are fast — data is already computed and stored
        """
        return lambda_.Function(
            self, "ReportGenerator",
            function_name = "connect-tester-report-generator",
            runtime       = lambda_.Runtime.PYTHON_3_12,
            handler       = "src.lambdas.report_generator.handler.lambda_handler",
            code          = lambda_.Code.from_asset("."),
            layers        = [self.deps_layer],
            role          = self.lambda_role,
            memory_size   = 256,
            timeout       = Duration.minutes(2),
            log_retention = logs.RetentionDays.ONE_MONTH,
            environment   = {
                "RESULTS_BUCKET": storage_stack.results_bucket.bucket_name,
                "RESULTS_TABLE":  storage_stack.results_table.table_name,
                "ENVIRONMENT":    "production"
            }
        )

    # ── API Gateway REST ───────────────────────────────────────────────────

    def _create_rest_api(self) -> apigw.RestApi:
        """
        REST API for test management and report retrieval.

        Routes:
          POST /suite/run          → test_runner
          POST /test/run           → test_runner
          POST /personas/generate  → test_runner
          POST /scenarios/generate → test_runner
          GET  /reports/{suite_id} → report_generator
          GET  /health             → test_runner
        """
        api = apigw.RestApi(
            self, "RestApi",
            rest_api_name = "connect-tester-api",
            description   = "Connect Tester REST API",
            default_cors_preflight_options = apigw.CorsOptions(
                allow_origins = apigw.Cors.ALL_ORIGINS,
                allow_methods = apigw.Cors.ALL_METHODS
            )
        )

        runner_integration  = apigw.LambdaIntegration(self.test_runner)
        reporter_integration = apigw.LambdaIntegration(self.report_generator)

        # /suite/run
        suite = api.root.add_resource("suite")
        suite.add_resource("run").add_method("POST", runner_integration)

        # /test/run
        test = api.root.add_resource("test")
        test.add_resource("run").add_method("POST", runner_integration)

        # /personas/generate
        personas = api.root.add_resource("personas")
        personas.add_resource("generate").add_method("POST", runner_integration)

        # /scenarios/generate
        scenarios = api.root.add_resource("scenarios")
        scenarios.add_resource("generate").add_method("POST", runner_integration)

        # /reports/{suite_id}
        reports = api.root.add_resource("reports")
        reports.add_resource("{suite_id}").add_method("GET", reporter_integration)

        # /health
        api.root.add_resource("health").add_method("GET", runner_integration)

        return api

    # ── API Gateway WebSocket ──────────────────────────────────────────────

    def _create_websocket_api(self) -> apigwv2.CfnApi:
        """
        WebSocket API for Twilio media stream.

        Three routes match Twilio's WebSocket lifecycle:
          $connect    → called when Twilio opens the stream
          message     → called for every audio packet
          $disconnect → called when Twilio closes the stream

        All three route to the same test_runner Lambda —
        the handler routes internally based on routeKey.
        """
        ws_api = apigwv2.CfnApi(
            self, "WebSocketApi",
            name                       = "connect-tester-ws",
            protocol_type              = "WEBSOCKET",
            route_selection_expression = "$request.body.event"
        )

        ws_integration = apigwv2.CfnIntegration(
            self, "WsIntegration",
            api_id           = ws_api.ref,
            integration_type = "AWS_PROXY",
            integration_uri  = (
                f"arn:aws:apigateway:{self.region}:lambda:path"
                f"/2015-03-31/functions/{self.test_runner.function_arn}/invocations"
            )
        )

        # Grant API Gateway permission to invoke the Lambda
        self.test_runner.add_permission(
            "WebSocketInvoke",
            principal  = iam.ServicePrincipal("apigateway.amazonaws.com"),
            source_arn = f"arn:aws:execute-api:{self.region}:{self.account}:{ws_api.ref}/*"
        )

        # $connect route
        apigwv2.CfnRoute(
            self, "ConnectRoute",
            api_id    = ws_api.ref,
            route_key = "$connect",
            target    = f"integrations/{ws_integration.ref}"
        )

        # message route — Twilio audio packets
        apigwv2.CfnRoute(
            self, "MessageRoute",
            api_id    = ws_api.ref,
            route_key = "message",
            target    = f"integrations/{ws_integration.ref}"
        )

        # $disconnect route
        apigwv2.CfnRoute(
            self, "DisconnectRoute",
            api_id    = ws_api.ref,
            route_key = "$disconnect",
            target    = f"integrations/{ws_integration.ref}"
        )

        # Deploy the WebSocket API
        apigwv2.CfnDeployment(
            self, "WsDeployment",
            api_id = ws_api.ref
        )

        return ws_api
