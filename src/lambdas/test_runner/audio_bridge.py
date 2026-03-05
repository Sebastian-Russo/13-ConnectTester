"""
The translation layer between the AI caller agent and the real phone call.

Think of this like a UN interpreter sitting between two people
who speak different languages. The caller agent speaks text,
the phone call speaks audio. This bridge translates continuously
in both directions so neither side knows about the other's format.

Polly:      text → audio  (caller agent → phone call)
Transcribe: audio → text  (phone call → caller agent)

Chime SDK placeholder: when we swap Twilio for Chime SDK,
only this file changes. The caller agent and voice caller
never know the difference.
"""

import io
import boto3
import time
from dataclasses import dataclass
from src.infrastructure.config import AWS_REGION


# ── Polly voice configuration ──────────────────────────────────────────────
# Neural voices sound significantly more human than standard voices
# Different emotional states should use different speaking styles
VOICE_MAP = {
    "calm":       {"voice_id": "Joanna", "engine": "neural"},
    "frustrated": {"voice_id": "Joanna", "engine": "neural"},
    "confused":   {"voice_id": "Joanna", "engine": "neural"},
    "urgent":     {"voice_id": "Joanna", "engine": "neural"},
}

# SSML templates for different emotional states
# SSML lets us control rate, pitch, and emphasis
# Real customers don't all speak at the same pace
SSML_TEMPLATES = {
    "calm":       '<speak><prosody rate="medium" pitch="medium">{text}</prosody></speak>',
    "frustrated": '<speak><prosody rate="fast" pitch="high">{text}</prosody></speak>',
    "confused":   '<speak><prosody rate="slow" pitch="low">{text}</prosody></speak>',
    "urgent":     '<speak><prosody rate="fast" pitch="high"><emphasis level="strong">{text}</emphasis></prosody></speak>',
}


@dataclass
class AudioSegment:
    """
    A single audio segment produced by Polly.
    Contains the raw audio bytes and metadata.
    """
    audio_bytes:  bytes
    sample_rate:  int = 8000       # 8kHz — standard telephony sample rate
    format:       str = "pcm"      # raw PCM for Twilio compatibility
    duration_ms:  int = 0          # estimated duration


@dataclass
class TranscriptionResult:
    """
    Result from Transcribe for a single audio segment.
    """
    text:          str
    confidence:    float           # 0.0 - 1.0
    is_final:      bool            # streaming transcribe sends partial results first
    duration_ms:   int = 0


class AudioBridge:
    """
    Manages text-to-speech and speech-to-text for the test call.

    Initialized once per test run. Maintains Polly and Transcribe
    clients for the duration of the call.

    # ── Chime SDK placeholder ──────────────────────────────────
    # When swapping Twilio for Chime SDK, the public interface
    # of this class stays identical. Only the internal implementation
    # of _send_audio_to_call() and _receive_audio_from_call() changes.
    # The caller agent and voice caller never need to know.
    # ──────────────────────────────────────────────────────────
    """

    def __init__(self, emotional_state: str = "calm"):
        self.polly       = boto3.client("polly",      region_name=AWS_REGION)
        self.transcribe  = boto3.client("transcribe", region_name=AWS_REGION)
        self.emotional_state = emotional_state

    # ── Text to Speech ─────────────────────────────────────────────────────

    def synthesize_speech(self, text: str, emotional_state: str = None) -> AudioSegment:
        """
        Convert caller agent text to audio using Amazon Polly.

        Uses SSML to apply emotional prosody — frustrated callers
        speak faster, confused callers speak slower.
        This makes the voice testing more realistic than a flat monotone.
        """
        state = emotional_state or self.emotional_state
        voice = VOICE_MAP.get(state, VOICE_MAP["calm"])
        ssml  = SSML_TEMPLATES.get(state, SSML_TEMPLATES["calm"])

        # Escape any XML special characters in the text before inserting into SSML
        safe_text = (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        ssml_text = ssml.format(text=safe_text)

        response = self.polly.synthesize_speech(
            Text         = ssml_text,
            TextType     = "ssml",
            VoiceId      = voice["voice_id"],
            Engine       = voice["engine"],
            OutputFormat = "pcm",              # raw PCM, 8kHz for telephony
            SampleRate   = "8000"
        )

        audio_bytes = response["AudioStream"].read()

        # Estimate duration: PCM at 8kHz mono = 8000 bytes per second
        duration_ms = int((len(audio_bytes) / 8000) * 1000)

        return AudioSegment(
            audio_bytes = audio_bytes,
            sample_rate = 8000,
            format      = "pcm",
            duration_ms = duration_ms
        )

    # ── Speech to Text ─────────────────────────────────────────────────────

    def transcribe_audio_file(
        self,
        s3_bucket:   str,
        s3_key:      str,
        job_name:    str = None
    ) -> TranscriptionResult:
        """
        Transcribe a recorded audio file stored in S3.
        Used for post-call analysis and when streaming isn't available.

        Batch transcription — sends job to Transcribe, polls until complete.
        Suitable for processing recorded call segments.
        """
        if not job_name:
            job_name = f"connect-tester-{int(time.time())}"

        # Start transcription job
        self.transcribe.start_transcription_job(
            TranscriptionJobName = job_name,
            MediaFormat          = "wav",
            Media                = {"MediaFileUri": f"s3://{s3_bucket}/{s3_key}"},
            LanguageCode         = "en-US",
            Settings             = {
                "ShowSpeakerLabels": False,
                "ChannelIdentification": False
            }
        )

        # Poll until complete — Transcribe is async
        while True:
            result = self.transcribe.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = result["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                transcript_uri = result["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                return self._fetch_transcript(transcript_uri)

            elif status == "FAILED":
                reason = result["TranscriptionJob"].get("FailureReason", "Unknown")
                print(f"[AudioBridge] Transcription failed: {reason}")
                return TranscriptionResult(text="", confidence=0.0, is_final=True)

            time.sleep(2)

    def transcribe_utterance(self, audio_bytes: bytes, temp_bucket: str) -> TranscriptionResult:
        """
        Transcribe a single audio utterance from the call.
        Saves to S3 temporarily, transcribes, returns text.

        This is the per-turn transcription used during a live call —
        each time Connect finishes speaking we transcribe that segment
        and feed the text to the caller agent.
        """
        import uuid

        # Save audio bytes to S3 temporarily
        s3 = boto3.client("s3", region_name=AWS_REGION)
        key = f"temp-audio/{uuid.uuid4()}.wav"

        s3.put_object(
            Bucket      = temp_bucket,
            Key         = key,
            Body        = audio_bytes,
            ContentType = "audio/wav"
        )

        result = self.transcribe_audio_file(
            s3_bucket = temp_bucket,
            s3_key    = key
        )

        # Clean up temp file
        s3.delete_object(Bucket=temp_bucket, Key=key)

        return result

    def _fetch_transcript(self, transcript_uri: str) -> TranscriptionResult:
        """
        Fetch and parse the completed transcript JSON from the URI
        that Transcribe provides after a job completes.
        """
        import urllib.request
        import json

        with urllib.request.urlopen(transcript_uri) as f:
            data = json.loads(f.read())

        transcript = data["results"]["transcripts"][0]["transcript"]
        items      = data["results"].get("items", [])

        # Average confidence across all words
        confidences = [
            float(item["alternatives"][0]["confidence"])
            for item in items
            if item["type"] == "pronunciation"
            and item["alternatives"][0].get("confidence")
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return TranscriptionResult(
            text       = transcript,
            confidence = avg_confidence,
            is_final   = True
        )

"""
The Chime SDK placeholder comment in the class docstring is intentional

- When you come back to swap Twilio for Chime SDK, _send_audio_to_call()
  and _receive_audio_from_call() are the two methods that change.
- The public interface — synthesize_speech() and transcribe_utterance() —
  stays identical so nothing else in the project needs to change.
"""
