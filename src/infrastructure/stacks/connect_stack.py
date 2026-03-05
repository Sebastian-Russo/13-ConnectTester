"""
Amazon Connect infrastructure — integrates the test suite
with your Connect instance.

This stack doesn't create a Connect instance — it assumes
you already have one (from the workshop or your professional work).
Instead it creates the resources that connect the test suite
TO your existing Connect instance:

- Lambda permissions for Connect to invoke test runner
- Connect contact flow (the flow under test — hotel reservation)
- IAM role for Connect to invoke Lambdas
- CloudWatch log group for Connect flow logs

Note: Some Connect resources (phone numbers, queues, routing profiles)
are managed through the Connect console rather than CDK because
the CDK Connect L2 constructs are still limited. We use CfnResource
for the lower-level constructs where needed.
"""

from aws_cdk import (
    Stack,
    aws_connect as connect,
    aws_iam as iam,
    aws_logs as logs,
    aws_lambda as lambda_,
    CfnOutput,
    Fn
)
from constructs import Construct
from src.infrastructure.stacks.compute_stack import ComputeStack
from src.infrastructure.config import (
    CONNECT_INSTANCE_ID,
    CONNECT_INSTANCE_ARN
)


class ConnectStack(Stack):

    def __init__(
        self,
        scope:         Construct,
        construct_id:  str,
        compute_stack: ComputeStack,
        **kwargs
    ):
        super().__init__(scope, construct_id, **kwargs)

        if not CONNECT_INSTANCE_ID or not CONNECT_INSTANCE_ARN:
            raise ValueError(
                "CONNECT_INSTANCE_ID and CONNECT_INSTANCE_ARN must be set in .env "
                "before deploying the Connect stack. "
                "Find these in the Connect console under Instance settings."
            )

        # ── Connect IAM role ───────────────────────────────────
        # Connect needs permission to invoke your Lambda functions
        # This role is assumed by the Connect service when a flow
        # invokes a Lambda block
        self.connect_role = self._create_connect_role(compute_stack)

        # ── Lambda permissions for Connect ─────────────────────
        # Connect must be explicitly granted permission to invoke
        # each Lambda function it calls from a flow block
        self._grant_connect_lambda_permissions(compute_stack)

        # ── CloudWatch log group ───────────────────────────────
        # Connect flow logs — essential for debugging flow failures
        # Shows exactly which block succeeded/failed and why
        self.flow_log_group = logs.LogGroup(
            self, "ConnectFlowLogs",
            log_group_name  = "/aws/connect/connect-tester",
            retention       = logs.RetentionDays.ONE_MONTH
        )

        # ── Contact flow ───────────────────────────────────────
        # The hotel reservation flow from the workshop —
        # this is the flow the test suite runs tests against
        self.hotel_flow = self._create_hotel_reservation_flow()

        # ── Outputs ────────────────────────────────────────────
        CfnOutput(
            self, "ConnectInstanceId",
            value       = CONNECT_INSTANCE_ID,
            description = "Amazon Connect instance ID"
        )

        CfnOutput(
            self, "HotelFlowId",
            value       = self.hotel_flow.attr_contact_flow_arn,
            export_name = "ConnectTesterHotelFlow",
            description = "Hotel reservation contact flow ARN"
        )

        CfnOutput(
            self, "FlowLogGroup",
            value       = self.flow_log_group.log_group_name,
            description = "CloudWatch log group for Connect flow logs"
        )

    # ── IAM ────────────────────────────────────────────────────────────────

    def _create_connect_role(self, compute_stack: ComputeStack) -> iam.Role:
        """
        IAM role assumed by Connect when invoking Lambda functions.

        Connect needs this role to:
        - Invoke Lambda functions from flow blocks
        - Write logs to CloudWatch
        """
        role = iam.Role(
            self, "ConnectRole",
            assumed_by  = iam.ServicePrincipal("connect.amazonaws.com"),
            description = "Role for Connect to invoke Lambda functions"
        )

        # Invoke Lambda functions from flow blocks
        role.add_to_policy(iam.PolicyStatement(
            actions   = ["lambda:InvokeFunction"],
            resources = [
                compute_stack.test_runner.function_arn,
                compute_stack.result_processor.function_arn,
                compute_stack.report_generator.function_arn
            ]
        ))

        # Write flow logs to CloudWatch
        role.add_to_policy(iam.PolicyStatement(
            actions   = [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            resources = ["*"]
        ))

        return role

    def _grant_connect_lambda_permissions(self, compute_stack: ComputeStack) -> None:
        """
        Grant Connect service principal permission to invoke each Lambda.

        Both the IAM role AND resource-based Lambda permissions are needed —
        Connect checks both. Missing either one causes silent invocation failures
        in the flow that are frustrating to debug.
        """
        for fn_name, fn in [
            ("TestRunner",      compute_stack.test_runner),
            ("ResultProcessor", compute_stack.result_processor),
            ("ReportGenerator", compute_stack.report_generator)
        ]:
            fn.add_permission(
                f"ConnectInvoke{fn_name}",
                principal  = iam.ServicePrincipal("connect.amazonaws.com"),
                source_arn = CONNECT_INSTANCE_ARN
            )

    # ── Contact flow ───────────────────────────────────────────────────────

    def _create_hotel_reservation_flow(self) -> connect.CfnContactFlow:
        """
        Hotel reservation contact flow — the system under test.

        This recreates the flow from the Bedrock AgentCore workshop:
        - Greets the caller
        - Invokes the Bedrock agent (hotel reservation AI)
        - Handles agent responses
        - Transfers to queue if agent can't help

        The flow content is defined as Connect flow JSON —
        the same format you see when you export a flow from
        the Connect console. This makes it version-controllable
        and deployable via CDK rather than manual console clicks.

        Note: The Bedrock agent ARN and knowledge base are referenced
        by environment-specific values from config.py — swap these
        for your actual ARNs after deploying AgentCore.
        """
        flow_content = {
            "Version": "2019-10-30",
            "StartAction": "greeting",
            "Actions": [
                {
                    "Identifier": "greeting",
                    "Type": "MessageParticipant",
                    "Parameters": {
                        "Text": (
                            "Thank you for calling. "
                            "I'm your AI hotel concierge. "
                            "How can I help you today?"
                        )
                    },
                    "Transitions": {
                        "NextAction": "invoke-agent",
                        "Errors":     [{"NextAction": "error-handler", "ErrorType": "Any"}]
                    }
                },
                {
                    "Identifier": "invoke-agent",
                    "Type":       "InvokeAgentBlock",
                    "Parameters": {
                        # Replace with your actual Bedrock AgentCore ARN
                        # after deploying the AgentCore stack
                        "AgentArn":  "{{BEDROCK_AGENT_ARN}}",
                        "AliasArn":  "{{BEDROCK_AGENT_ALIAS_ARN}}"
                    },
                    "Transitions": {
                        "NextAction": "agent-response",
                        "Errors":     [{"NextAction": "error-handler", "ErrorType": "Any"}]
                    }
                },
                {
                    "Identifier": "agent-response",
                    "Type":       "MessageParticipant",
                    "Parameters": {
                        "Text": "$.AgentResponse"
                    },
                    "Transitions": {
                        "NextAction": "check-complete",
                        "Errors":     [{"NextAction": "error-handler", "ErrorType": "Any"}]
                    }
                },
                {
                    "Identifier": "check-complete",
                    "Type":       "CheckAttribute",
                    "Parameters": {
                        "Attribute": "$.AgentSessionState",
                        "Value":     "ENDED"
                    },
                    "Transitions": {
                        "NextAction": "goodbye",
                        "Conditions": [
                            {
                                "NextAction": "invoke-agent",
                                "Condition": {
                                    "Operator": "Equals",
                                    "Operands": ["false"]
                                }
                            }
                        ],
                        "Errors": [{"NextAction": "error-handler", "ErrorType": "Any"}]
                    }
                },
                {
                    "Identifier": "goodbye",
                    "Type":       "MessageParticipant",
                    "Parameters": {
                        "Text": "Thank you for calling. Have a great day!"
                    },
                    "Transitions": {
                        "NextAction": "end-flow",
                        "Errors":     [{"NextAction": "error-handler", "ErrorType": "Any"}]
                    }
                },
                {
                    "Identifier": "error-handler",
                    "Type":       "MessageParticipant",
                    "Parameters": {
                        "Text": (
                            "I'm sorry, I'm having trouble processing your request. "
                            "Please try again or hold for an agent."
                        )
                    },
                    "Transitions": {
                        "NextAction": "end-flow",
                        "Errors":     [{"NextAction": "end-flow", "ErrorType": "Any"}]
                    }
                },
                {
                    "Identifier": "end-flow",
                    "Type":       "DisconnectParticipant",
                    "Parameters": {}
                }
            ]
        }

        import json
        return connect.CfnContactFlow(
            self, "HotelReservationFlow",
            instance_arn  = CONNECT_INSTANCE_ARN,
            name          = "Hotel Reservation - AI Agent",
            type          = "CONTACT_FLOW",
            description   = "Hotel reservation flow powered by Bedrock AgentCore",
            content       = json.dumps(flow_content)
        )
