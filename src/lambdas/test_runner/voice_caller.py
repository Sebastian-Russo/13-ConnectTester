"""
Manages the actual phone call lifecycle via Twilio.

Think of this like a phone operator — it dials the number,
keeps the line open, routes audio in and out, and hangs up
when the test is done.

The voice caller knows nothing about AI or personas.
It only knows about phone calls — connecting, streaming
audio, and disconnecting cleanly.

# ── Chime SDK placeholder ──────────────────────────────────
# To swap Twilio for Chime SDK PSTN:
# 1. Replace TwilioVoiceCaller with ChimeVoiceCaller
# 2. Implement the same connect/stream/disconnect interface
# 3. Update voice_caller imports in handler.py only
# Everything else in the project stays identical.
# ──────────────────────────────────────────────────────────
"""

import time
from dataclasses import dataclass, field
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Stream, Start
from src.infrastructure.config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    CONNECT_PHONE_NUMBER,
    MAX_CALL_DURATION
)


@dataclass
class CallSession:
    """
    Represents an active test call.
    Tracks the Twilio call SID and connection state
    for the duration of the test.
    """
    call_sid:        str
    from_number:     str
    to_number:       str
    start_time:      float = field(default_factory=time.time)
    status:          str   = "initiated"    # initiated | in-progress | completed | failed
    duration_seconds: int  = 0
    recording_url:   str   = ""             # Twilio recording URL if enabled


@dataclass
class AudioChunk:
    """
    A chunk of audio received from the call.
    Twilio streams audio in small chunks — we collect
    them and reassemble into utterances.
    """
    payload:     bytes
    timestamp:   float
    track:       str      # "inbound" (Connect speaking) or "outbound" (caller speaking)
    sequence:    int      = 0


class TwilioVoiceCaller:
    """
    Manages test calls via Twilio Programmable Voice.

    Flow:
    1. Dial Connect's phone number from a Twilio number
    2. Twilio connects the call and opens a media stream
    3. We stream Polly-generated audio into the call (caller speaks)
    4. We receive audio from Connect and send to Transcribe
    5. Repeat until test scenario completes or call ends
    6. Hang up and collect call metadata

    Twilio media streams use WebSocket — audio flows bidirectionally
    in real time. This is what makes live voice testing possible.
    """

    def __init__(self, stream_url: str):
        """
        stream_url: your WebSocket endpoint URL where Twilio will
                    send the audio stream e.g. wss://your-ngrok-url/stream
                    During development: use ngrok to expose localhost
                    In production: your Lambda URL or API Gateway WebSocket
        """
        self.client     = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.stream_url = stream_url
        self.session    = None

    def initiate_call(self) -> CallSession:
        """
        Dial Connect's phone number and open a media stream.

        Twilio makes the call and immediately starts streaming
        audio to/from our WebSocket endpoint. The call is live
        as soon as Connect picks up.
        """
        # Build TwiML instruction — tells Twilio what to do with the call
        # Start a bidirectional media stream to our WebSocket URL
        response = VoiceResponse()
        start    = Start()
        start.stream(url=self.stream_url, track="both_tracks")
        response.append(start)

        # Keep call alive while we handle the stream
        # Pause for MAX_CALL_DURATION seconds max
        response.pause(length=MAX_CALL_DURATION)

        call = self.client.calls.create(
            from_ = TWILIO_PHONE_NUMBER,
            to    = CONNECT_PHONE_NUMBER,
            twiml = str(response)
        )

        self.session = CallSession(
            call_sid    = call.sid,
            from_number = TWILIO_PHONE_NUMBER,
            to_number   = CONNECT_PHONE_NUMBER,
            status      = "initiated"
        )

        print(f"[VoiceCaller] Call initiated — SID: {call.sid}")
        return self.session

    def wait_for_answer(self, timeout: int = 30) -> bool:
        """
        Poll Twilio until Connect answers or timeout is reached.
        Returns True if answered, False if timeout or failed.
        """
        start = time.time()

        while time.time() - start < timeout:
            call = self.client.calls(self.session.call_sid).fetch()

            if call.status == "in-progress":
                self.session.status = "in-progress"
                print(f"[VoiceCaller] Call answered by Connect")
                return True

            elif call.status in ("busy", "failed", "no-answer", "canceled"):
                self.session.status = "failed"
                print(f"[VoiceCaller] Call failed — status: {call.status}")
                return False

            time.sleep(1)

        print(f"[VoiceCaller] Answer timeout after {timeout}s")
        return False

    def send_audio(self, audio_bytes: bytes) -> bool:
        """
        Send synthesized audio into the call via the media stream.

        In production this writes PCM audio bytes to the WebSocket
        connection that Twilio opened. The WebSocket handler
        (in handler.py) owns the actual connection — this method
        signals it to send audio.

        Returns True if audio was sent successfully.
        """
        # WebSocket send is handled by the stream handler in handler.py
        # This method is a coordination point — sets audio to send
        # on the next available stream write cycle
        if not self.session or self.session.status != "in-progress":
            print(f"[VoiceCaller] Cannot send audio — call not in progress")
            return False

        # Signal audio bridge to push bytes to WebSocket stream
        # Actual WebSocket write happens in the stream handler
        self._pending_audio = audio_bytes
        return True

    def receive_audio(self, chunk: AudioChunk) -> None:
        """
        Receive an audio chunk from the call stream.
        Called by the WebSocket handler each time Twilio sends audio.
        Accumulates chunks until silence detected (end of utterance).
        """
        if not hasattr(self, "_inbound_buffer"):
            self._inbound_buffer  = []
            self._last_chunk_time = time.time()

        if chunk.track == "inbound":   # inbound = Connect speaking
            self._inbound_buffer.append(chunk.payload)
            self._last_chunk_time = time.time()

    def get_utterance_audio(self, silence_threshold_ms: int = 800) -> bytes | None:
        """
        Return accumulated inbound audio if silence detected.

        Silence detection: if no new audio chunk received for
        silence_threshold_ms, assume Connect finished speaking
        and return the buffered audio for transcription.

        Returns None if Connect is still speaking.
        """
        if not hasattr(self, "_inbound_buffer") or not self._inbound_buffer:
            return None

        silence_ms = (time.time() - self._last_chunk_time) * 1000

        if silence_ms >= silence_threshold_ms:
            audio = b"".join(self._inbound_buffer)
            self._inbound_buffer  = []
            self._last_chunk_time = time.time()
            return audio

        return None

    def end_call(self) -> CallSession:
        """
        Hang up the call and collect final metadata.
        Called when the scenario completes or caller agent decides to hang up.
        """
        if not self.session:
            return None

        try:
            call = self.client.calls(self.session.call_sid).update(status="completed")
            self.session.status           = "completed"
            self.session.duration_seconds = int(time.time() - self.session.start_time)
            print(f"[VoiceCaller] Call ended — duration: {self.session.duration_seconds}s")

        except Exception as e:
            print(f"[VoiceCaller] Error ending call: {e}")
            self.session.status = "failed"

        return self.session

    def get_call_status(self) -> str:
        """Fetch current call status from Twilio."""
        if not self.session:
            return "no-session"

        call = self.client.calls(self.session.call_sid).fetch()
        return call.status


"""
Two things worth noting.
First, the silence_threshold_ms detection in get_utterance_audio()
is how we know Connect finished speaking. We can't send the audio to Transcribe mid-sentence,
so we watch for 800ms of silence and treat that as the end of an utterance.
That number is tunable — too low and you cut off slow speakers, too high and you add latency to every turn.
Second, the WebSocket note in send_audio() — Twilio media streams are WebSocket-based, not REST.
The actual WebSocket connection lives in handler.py because Lambda owns the connection lifecycle.
This file just signals what audio to send on the next write cycle.
"""