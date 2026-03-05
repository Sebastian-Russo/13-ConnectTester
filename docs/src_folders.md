## src/personas/

Two files. **persona_generator.py** uses an LLM to create realistic caller personas — name, age, emotional state, what they want, how patient they are, whether they have an accent or speech pattern worth simulating. **scenario_generator.py** takes a persona and generates the ordered sequence of intents they'll try to accomplish during the call. "Maria wants to book 2 nights, will ask about pet policy, then ask about check-in time, then get frustrated if the agent repeats itself."

## src/caller/

Three files. **caller_agent.py** is the AI playing the customer role — it receives what the contact flow just said and decides what to say next, staying in character as the persona. **voice_caller.py** manages the actual phone call lifecycle via Twilio — dialing out, maintaining the connection, feeding audio in and out. **audio_bridge.py** handles the translation layer — Polly converts the caller agent's text responses to audio to send into the call, Transcribe converts what Connect says back to text so the caller agent can read and respond.

## src/backends/

Three files. **base_backend.py** defines the interface both AI backends must implement — one generate() method with consistent inputs and outputs. **anthropic_backend.py** implements that interface using the Anthropic SDK. **bedrock_backend.py** implements the same interface using boto3. The rest of the codebase never imports directly from either — it only talks to base_backend.py. Swapping backends requires zero changes anywhere else.

## src/evaluator/

One file. **evaluator.py** reads the completed conversation transcript and scores it across multiple dimensions — goal completion rate, how many turns it took, where the caller had to repeat themselves, sentiment trajectory across the call, intent recognition accuracy, fallout points where a real customer might have hung up.

## src/reporter/

One file. **reporter.py** takes evaluator scores from one or many test runs and produces structured output — per-run reports, aggregate reports across a full test suite, comparison reports between Anthropic and Bedrock backends, trend reports across multiple suite runs over time.

## src/orchestrator.py

The conductor. **orchestrator.py** takes a test suite definition (list of personas + scenarios + config), spins up caller agents for each test, runs them in parallel, collects results as they complete, hands everything to the reporter. The Lambda handlers in lambdas/ call this directly — it's the main entry point for everything.
