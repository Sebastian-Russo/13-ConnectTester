# Backends

Practically speaking you only need one. In production you'd pick Bedrock and be done with it — it's AWS-native, IAM auth, no external API keys, integrates cleanly with Lambda. That's the right answer for a real Connect testing tool.

But as a learning project having both gives you something valuable — a direct comparison. Same prompt, same persona, same scenario, two backends.

## You can measure:

- Response quality — does Bedrock Claude answer differently than Anthropic Claude?
- Latency — which is faster for caller agent turns?
- Cost — Bedrock pricing vs Anthropic API pricing for the same workload
- Integration friction — how much harder is Bedrock to set up vs Anthropic?
