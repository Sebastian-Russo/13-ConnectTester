# What the Caller Agent Actually Does

Every previous agent in the curriculum was trying to help the user accomplish something. This agent is different — it's pretending to be a human customer calling into a contact flow.
It has to stay in character the entire call. It knows the persona (Maria, 34, frustrated, wants to book 2 nights with a cat). It knows the scenario (ordered steps to accomplish). But it doesn't follow a script — it listens to what Connect says and decides how to respond naturally as Maria would, including getting frustrated if the flow repeats itself, going slightly off topic, using Maria's speech patterns.

## The loop looks like this:

Connect speaks
      ↓
audio_bridge transcribes to text
      ↓
caller_agent receives text + full conversation history + persona + current step
      ↓
caller_agent decides:
  - what to say as this persona
  - whether the current step was completed
  - whether to move to the next step
  - whether the call should end
      ↓
returns text response + state update
      ↓
audio_bridge converts text to speech
      ↓
voice_caller sends audio back into the call

The agent returns structured output every turn — not just what to say, but also whether the current scenario step was completed, what the sentiment is, and whether to continue or hang up.


# What the Audio Bridge Does

The caller agent produces text. The contact flow expects audio. The contact flow produces audio. The caller agent expects text. The bridge translates in both directions every turn.

Caller agent text → Polly → MP3 audio → Twilio → Connect hears it
Connect speaks   → Twilio → audio stream → Transcribe → caller agent reads it

## Two AWS services doing the heavy lifting:

**Amazon Polly** — text to speech. You send it a string, it returns an audio stream. Multiple voices, multiple languages, supports SSML for controlling pace, emphasis, and pauses. Important for realism — a frustrated caller speaks faster, a confused caller speaks slower. You can encode that in SSML.

**Amazon Transcribe** — speech to text. Two modes: batch (send a file, get a transcript back) and streaming (real-time transcription of a live audio stream). For a live phone call you need streaming mode — you can't wait for the call to end to transcribe it.
