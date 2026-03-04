import os
from dotenv import load_dotenv

load_dotenv()

# ── AWS ───────────────────────────────────────────────────────
AWS_ACCOUNT    = os.getenv("AWS_ACCOUNT_ID")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")

# ── Amazon Connect ────────────────────────────────────────────
CONNECT_INSTANCE_ID  = os.getenv("CONNECT_INSTANCE_ID")
CONNECT_INSTANCE_ARN = os.getenv("CONNECT_INSTANCE_ARN")

# ── AI Backends ───────────────────────────────────────────────
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")
BEDROCK_MODEL_ID     = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241001-v1:0")
BEDROCK_REGION       = os.getenv("BEDROCK_REGION", "us-east-1")

# ── Twilio (voice calls) ──────────────────────────────────────
TWILIO_ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER")   # your Twilio number
CONNECT_PHONE_NUMBER = os.getenv("CONNECT_PHONE_NUMBER")  # the Connect number to call

# ── Storage ───────────────────────────────────────────────────
# These get set by CDK after deployment — populated via stack outputs
RESULTS_BUCKET     = os.getenv("RESULTS_BUCKET")          # S3 bucket for test results
RESULTS_TABLE      = os.getenv("RESULTS_TABLE", "connect-tester-results")  # DynamoDB table
SCENARIOS_BUCKET   = os.getenv("SCENARIOS_BUCKET")        # S3 bucket for scenario YAML files

# ── Test Run Settings ─────────────────────────────────────────
MAX_PARALLEL_TESTS = int(os.getenv("MAX_PARALLEL_TESTS", "5"))
MAX_CALL_DURATION  = int(os.getenv("MAX_CALL_DURATION", "300"))   # seconds, 5 min max per call
MAX_TURNS          = int(os.getenv("MAX_TURNS", "20"))            # max conversation turns per test

# ── Environment ───────────────────────────────────────────────
ENVIRONMENT        = os.getenv("ENVIRONMENT", "dev")   # dev | staging | prod
