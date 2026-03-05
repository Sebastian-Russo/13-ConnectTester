#!/usr/bin/env python3
"""
CDK app entry point.

This is what CDK runs when you execute cdk deploy, cdk synth,
or cdk diff. It instantiates all three stacks in dependency order
and wires them together.

Stack deployment order:
1. StorageStack  — S3 buckets and DynamoDB table (no dependencies)
2. ComputeStack  — Lambda functions and API Gateway (depends on storage)
3. ConnectStack  — Connect integration (depends on compute)

Run from src/infrastructure/ with the CDK venv active:
  cdk synth    — synthesize CloudFormation templates, no deployment
  cdk diff     — show what will change vs deployed stack
  cdk deploy   — deploy all stacks
  cdk deploy ConnectTesterStorage  — deploy one stack only
  cdk destroy  — tear down all stacks (careful — buckets are RETAIN)
"""

import aws_cdk as cdk
from stacks.storage_stack import StorageStack
from stacks.compute_stack import ComputeStack
from stacks.connect_stack import ConnectStack
from config import AWS_ACCOUNT, AWS_REGION

app = cdk.App()

# ── Environment ────────────────────────────────────────────────────────────
# Explicitly set account and region so CDK can resolve environment-specific
# values like availability zones and service endpoints.
# Without this CDK runs in "environment-agnostic" mode and some constructs
# (like Lambda layers and VPCs) won't synthesize correctly.
env = cdk.Environment(
    account = AWS_ACCOUNT,
    region  = AWS_REGION
)

# ── Stack 1: Storage ───────────────────────────────────────────────────────
storage_stack = StorageStack(
    app, "ConnectTesterStorage",
    env         = env,
    description = "Connect Tester — S3 buckets and DynamoDB table"
)

# ── Stack 2: Compute ───────────────────────────────────────────────────────
# Receives storage_stack so it can reference bucket names and table ARN
# in Lambda environment variables and IAM policies
compute_stack = ComputeStack(
    app, "ConnectTesterCompute",
    env           = env,
    storage_stack = storage_stack,
    description   = "Connect Tester — Lambda functions and API Gateway"
)

# CDK tracks this dependency automatically because ComputeStack references
# storage_stack resources — but making it explicit is good practice
compute_stack.add_dependency(storage_stack)

# ── Stack 3: Connect ───────────────────────────────────────────────────────
# Receives compute_stack so it can grant Connect permission to invoke Lambdas
connect_stack = ConnectStack(
    app, "ConnectTesterConnect",
    env           = env,
    compute_stack = compute_stack,
    description   = "Connect Tester — Amazon Connect integration"
)

connect_stack.add_dependency(compute_stack)

# ── Tags ───────────────────────────────────────────────────────────────────
# Tags applied to every resource in every stack.
# Makes cost allocation and resource filtering easy in the AWS console.
for stack in [storage_stack, compute_stack, connect_stack]:
    cdk.Tags.of(stack).add("Project",     "ConnectTester")
    cdk.Tags.of(stack).add("ManagedBy",   "CDK")
    cdk.Tags.of(stack).add("Environment", "dev")

app.synth()

