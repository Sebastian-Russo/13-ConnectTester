"""
Storage infrastructure — S3 buckets and DynamoDB table.

Everything else in the stack depends on these resources,
so this stack is deployed first and exports its resource
names for other stacks to import.

Two S3 buckets:
- Results bucket: stores full test transcripts and reports as JSON
- Scenarios bucket: stores YAML scenario definitions

One DynamoDB table:
- Results table: fast lookup of test run metadata and scores
  Full transcripts stay in S3 — DynamoDB holds the queryable summary
"""

from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    CfnOutput,
    Duration
)
from constructs import Construct


class StorageStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ── Results bucket ─────────────────────────────────────
        # Stores full test transcripts, suite reports, backend
        # comparison reports — anything too large for DynamoDB
        self.results_bucket = s3.Bucket(
            self, "ResultsBucket",
            bucket_name        = f"connect-tester-results-{self.account}",
            removal_policy     = RemovalPolicy.RETAIN,   # never auto-delete test data
            versioned          = True,                   # protect against accidental overwrites
            lifecycle_rules    = [
                s3.LifecycleRule(
                    id             = "archive-old-results",
                    enabled        = True,
                    transitions    = [
                        s3.Transition(
                            storage_class   = s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after = Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class   = s3.StorageClass.GLACIER,
                            transition_after = Duration.days(90)
                        )
                    ]
                )
            ],
            cors               = [
                s3.CorsRule(
                    allowed_methods = [s3.HttpMethods.GET],
                    allowed_origins = ["*"],
                    allowed_headers = ["*"]
                )
            ]
        )

        # ── Scenarios bucket ───────────────────────────────────
        # Stores YAML scenario definition files
        # Smaller, simpler bucket — no lifecycle rules needed
        self.scenarios_bucket = s3.Bucket(
            self, "ScenariosBucket",
            bucket_name    = f"connect-tester-scenarios-{self.account}",
            removal_policy = RemovalPolicy.RETAIN,
            versioned      = True
        )

        # ── Results DynamoDB table ─────────────────────────────
        # Fast lookup of test run metadata by test_id
        # GSIs enable querying by persona, scenario, backend, and date
        #
        # Access patterns this table supports:
        #   - Get single test run by test_id (primary key)
        #   - List all runs for a persona (GSI: persona-index)
        #   - List all runs for a scenario (GSI: scenario-index)
        #   - List all runs by backend (GSI: backend-index)
        #   - List recent runs sorted by date (GSI: date-index)
        self.results_table = dynamodb.Table(
            self, "ResultsTable",
            table_name     = "connect-tester-results",
            partition_key  = dynamodb.Attribute(
                name = "test_id",
                type = dynamodb.AttributeType.STRING
            ),
            billing_mode   = dynamodb.BillingMode.PAY_PER_REQUEST,  # no capacity planning
            removal_policy = RemovalPolicy.RETAIN,
            point_in_time_recovery = True                            # protect against accidental deletes
        )

        # GSI: query by persona name
        self.results_table.add_global_secondary_index(
            index_name     = "persona-index",
            partition_key  = dynamodb.Attribute(
                name = "persona",
                type = dynamodb.AttributeType.STRING
            ),
            sort_key       = dynamodb.Attribute(
                name = "overall_score",
                type = dynamodb.AttributeType.NUMBER
            )
        )

        # GSI: query by scenario name
        self.results_table.add_global_secondary_index(
            index_name     = "scenario-index",
            partition_key  = dynamodb.Attribute(
                name = "scenario",
                type = dynamodb.AttributeType.STRING
            ),
            sort_key       = dynamodb.Attribute(
                name = "overall_score",
                type = dynamodb.AttributeType.NUMBER
            )
        )

        # GSI: query by backend — enables backend comparison queries
        self.results_table.add_global_secondary_index(
            index_name     = "backend-index",
            partition_key  = dynamodb.Attribute(
                name = "backend",
                type = dynamodb.AttributeType.STRING
            ),
            sort_key       = dynamodb.Attribute(
                name = "overall_score",
                type = dynamodb.AttributeType.NUMBER
            )
        )

        # ── Stack outputs ──────────────────────────────────────
        # CDK exports these values so other stacks can import them
        # Also visible in CloudFormation console after deployment
        CfnOutput(
            self, "ResultsBucketName",
            value       = self.results_bucket.bucket_name,
            export_name = "ConnectTesterResultsBucket",
            description = "S3 bucket for test results and reports"
        )

        CfnOutput(
            self, "ScenariosBucketName",
            value       = self.scenarios_bucket.bucket_name,
            export_name = "ConnectTesterScenariosBucket",
            description = "S3 bucket for scenario YAML files"
        )

        CfnOutput(
            self, "ResultsTableName",
            value       = self.results_table.table_name,
            export_name = "ConnectTesterResultsTable",
            description = "DynamoDB table for test results"
        )
