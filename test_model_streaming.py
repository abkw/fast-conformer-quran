"""
quran_streaming_asr.py
──────────────────────
Live streaming inference for the fine-tuned Quran FastConformer model.

How it works:
  Microphone → 80ms audio frames → cache-aware FastConformer encoder
  → RNNT decoder emits tokens frame-by-frame → word callback → your UI

Three usage modes:
  1. StreamingASR class       — integrate into your own app
  2. live_demo()              — terminal demo with live word display
  3. FastAPI server           — REST + WebSocket endpoint for a mobile app

Requirements:
  pip install nemo_toolkit[asr] pyaudio numpy fastapi uvicorn websockets

Model loading order (critical):
  1. restore_from() / from_pretrained()  — load weights as saved (bilateral)
  2. _fix_conv_padding()                 — ensure symmetric padding before
                                           any attention mode switch
  3. _apply_streaming_attention()        — switch to causal local attention
                                           and set conv to causal padding
  The .nemo file was saved with bilateral attention and symmetric conv
  padding. Step 3 switches both. If you skip step 2 you may end up with
  mismatched padding between conv and attention.
"""

import numpy as np
import queue
import threading
import time
import json
import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamingConfig:
    # Path to your .nemo checkpoint (local path or HuggingFace repo ID)
    model_path: str = "quran_asr/checkpoints/phase1_top3/phase1_top3_wer0.0038.nemo"

    # Audio
    sample_rate: int = 16000
    frame_ms: int = 80        # one encoder step ≈ 80ms
    chunk_ms: int = 1600      # process in 1.6s chunks (20 frames at a time)

    # Streaming attention — must match what you used in the streaming eval
    att_context_left: int = 128   # frames (~10.24s lookback at 80ms/frame)
    att_context_right: int = 0    # 0 = fully causal

    # RNNT decoding
    max_symbols_per_step: int = 10

    # Silence detection
    silence_threshold_db: float = -40.0
    silence_frames_to_stop: int = 20   # ~1.6s of silence triggers ayah flush

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordResult:
    text: str
    start_ms: float
    end_ms: float
    is_partial: bool = False

    def __str__(self):
        return f"{'~' if self.is_partial else '✓'} {self.text}"


@dataclass
class AyahResult:
    words: List[WordResult]
    full_text: str
    duration_ms: float
    rtf: float

    def __str__(self):
        return self.full_text


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-10)
    return 20 * np.log10(rms / 32768.0)


def _fix_conv_padding_symmetric(model):
    """
    Ensure all conformer conv layers have symmetric padding.

    Call this BEFORE _apply_streaming_attention(). The .nemo file was saved
    with bilateral attention and symmetric conv padding. This call is a
    safety check — it's a no-op if padding is already correct, but guards
    against any state left over from a previous streaming session on the
    same model instance.
    """
    fixed = 0
    for layer in model.encoder.layers:
        if hasattr(layer, "conv") and hasattr(layer.conv, "conv"):
            conv = layer.conv.conv
            if hasattr(conv, "padding"):
                ks = (
                    conv.kernel_size[0]
                    if isinstance(conv.kernel_size, tuple)
                    else conv.kernel_size
                )
                symmetric = ((ks - 1) // 2, (ks - 1) // 2)
                if conv.padding != symmetric:
                    conv.padding = symmetric
                    fixed += 1
    return fixed


def _apply_streaming_attention(model, left_context: int, right_context: int):
    """
    Switch model to streaming (causal local) attention AND causal conv padding.

    After this call:
      - encoder.self_attention_model == "rel_pos_local_attn"
      - encoder.att_context_size == [left_context, right_context]
      - all conformer conv layers have causal (asymmetric) padding
    """
    from omegaconf import open_dict

    model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[left_context, right_context],
    )
    with open_dict(model.cfg):
        model.cfg.encoder.conv_context_size = "causal"

    # Set causal (asymmetric) conv padding to match causal attention
    for layer in model.encoder.layers:
        if hasattr(layer, "conv") and hasattr(layer.conv, "conv"):
            conv = layer.conv.conv
            if hasattr(conv, "padding"):
                ks = (
                    conv.kernel_size[0]
                    if isinstance(conv.kernel_size, tuple)
                    else conv.kernel_size
                )
                conv.padding = (ks - 1, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Core streaming ASR
# ─────────────────────────────────────────────────────────────────────────────

class StreamingASR:
    """
    Cache-aware streaming ASR for Quran recitation.

    Usage:
        asr = StreamingASR(
            StreamingConfig(model_path="./checkpoint.nemo"),
            on_word=lambda w: print(w.text, end=" ", flush=True),
            on_ayah=lambda a: print(f"\\n>>> {a.full_text}"),
        ).load()

        # Feed 80ms PCM int16 frames
        for frame in your_audio_source:
            asr.process_frame(frame)
    """

    def __init__(
        self,
        config: StreamingConfig,
        on_word: Optional[Callable[[WordResult], None]] = None,
        on_ayah: Optional[Callable[[AyahResult], None]] = None,
    ):
        self.config = config
        self.on_word = on_word
        self.on_ayah = on_ayah

        self._model = None
        # Two separate caches required by cache_aware_stream_step:
        #   _cache_last_channel: attention/conv channel cache
        #   _cache_last_time:    time-dimension cache
        self._cache_last_channel = None
        self._cache_last_time = None

        self._audio_buffer = np.array([], dtype=np.float32)
        self._word_buffer: List[WordResult] = []
        # _total_frames counts across the entire session (never resets);
        # used to compute absolute timestamps.
        self._total_frames = 0
        # _ayah_frames counts within the current ayah (resets on flush).
        self._ayah_frames = 0
        self._silence_counter = 0
        self._stream_start_time = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def load(self) -> "StreamingASR":
        """
        Load the model and configure it for streaming inference.

        Loading sequence (order matters):
          1. restore_from() / from_pretrained()
          2. _fix_conv_padding_symmetric()   — safety reset to bilateral state
          3. _apply_streaming_attention()    — switch to causal streaming mode
          4. set greedy decoder via OmegaConf (not a plain dict)
        """
        import torch
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf

        print(f"Loading model from: {self.config.model_path}")

        # Step 1 — load weights
        try:
            self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                self.config.model_path,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
            print("  ✓ Loaded from local .nemo file")
        except Exception:
            self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                self.config.model_path
            )
            print("  ✓ Loaded from HuggingFace")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        # Step 2 — safety reset of conv padding to symmetric
        n_fixed = _fix_conv_padding_symmetric(self._model)
        print(f"  ✓ Conv padding verified symmetric ({n_fixed} layers adjusted)")

        # Step 3 — switch to causal streaming (attention + conv)
        _apply_streaming_attention(
            self._model,
            self.config.att_context_left,
            self.config.att_context_right,
        )
        ctx = self._model.encoder.att_context_size
        mode = self._model.encoder.self_attention_model
        print(f"  ✓ Streaming attention: {mode}, context={ctx}")

        # Step 4 — greedy decoder via OmegaConf (plain dict is rejected by NeMo)
        decoding_cfg = OmegaConf.structured(
            self._model.cfg.decoding
        )
        OmegaConf.set_struct(decoding_cfg, False)
        decoding_cfg.strategy = "greedy"
        decoding_cfg.greedy.max_symbols = self.config.max_symbols_per_step
        self._model.change_decoding_strategy(decoding_cfg)
        print(f"  ✓ Greedy decoder (max_symbols={self.config.max_symbols_per_step})")

        import torch as _torch
        _torch.set_float32_matmul_precision('high')  # use tensor cores, big RTF improvement
        self._model.eval()
        self._reset_state()

        # Warm up: one silent chunk so the first real chunk isn't slow
        self._warmup()

        print("Model ready for streaming.")
        return self

    def reset(self):
        """Reset encoder cache and buffers. Call between ayahs if needed."""
        self._reset_state()

    def process_frame(self, audio_frame: np.ndarray) -> List[WordResult]:
        """
        Process one frame of PCM int16 audio (80ms = 1280 samples at 16kHz).
        Returns any new WordResult objects emitted this frame.

        Args:
            audio_frame: np.ndarray of int16 PCM, shape (frame_samples,)
        """
        if self._model is None:
            raise RuntimeError("Call load() before process_frame()")

        # Absolute timestamp for this frame (survives ayah resets)
        frame_start_ms = self._total_frames * self.config.frame_ms
        self._total_frames += 1

        # Silence detection
        if _rms_db(audio_frame) < self.config.silence_threshold_db:
            self._silence_counter += 1
            if self._silence_counter >= self.config.silence_frames_to_stop:
                self._flush_ayah()
            return []
        else:
            self._silence_counter = 0

        # Convert int16 → float32 [-1, 1]
        audio_f32 = audio_frame.astype(np.float32) / 32768.0
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_f32])
        self._ayah_frames += 1

        # Process once we have a full chunk
        if len(self._audio_buffer) < self.config.chunk_samples:
            return []

        chunk = self._audio_buffer[: self.config.chunk_samples]
        self._audio_buffer = self._audio_buffer[self.config.chunk_samples:]

        return self._forward_chunk(chunk, frame_start_ms)

    # ── private ───────────────────────────────────────────────────────────────

    def _warmup(self):
        """
        Push one silent chunk through the model to initialise CUDA kernels
        and allocate the encoder cache. The first real chunk is then fast
        and the cache is in a known-good state.
        """
        import torch
        silence = np.zeros(self.config.chunk_samples, dtype=np.float32)
        device = next(self._model.parameters()).device
        audio_tensor = torch.tensor(silence, dtype=torch.float32, device=device).unsqueeze(0)
        audio_len = torch.tensor([len(silence)], device=device)
        with torch.no_grad():
            processed, processed_len = self._model.preprocessor(
                input_signal=audio_tensor, length=audio_len
            )
            _, _, self._cache_last_channel, self._cache_last_time, _ = (
                self._model.encoder.cache_aware_stream_step(
                    processed_signal=processed,
                    processed_signal_length=processed_len,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    keep_all_outputs=False,
                )
            )
        # Reset after warmup — we don't want silent-chunk artifacts in the cache
        self._cache_last_channel = None
        self._cache_last_time = None
        print("  ✓ Warm-up complete")

    def _forward_chunk(self, audio_chunk: np.ndarray, start_ms: float) -> List[WordResult]:
        """
        Run one cache-aware encoder + RNNT decoder step.

        cache_aware_stream_step() processes only the new audio frames while
        reading past context from _cache_last_channel / _cache_last_time,
        then updates both caches. This makes each step O(chunk) rather than
        O(full_audio).
        """
        import torch

        device = next(self._model.parameters()).device
        audio_tensor = torch.tensor(
            audio_chunk, dtype=torch.float32, device=device
        ).unsqueeze(0)
        audio_len = torch.tensor([len(audio_chunk)], device=device)

        with torch.no_grad():
            # Mel spectrogram
            processed, processed_len = self._model.preprocessor(
                input_signal=audio_tensor,
                length=audio_len,
            )

            # Cache-aware encoder step
            # Returns: (encoded, encoded_len, new_channel_cache, new_time_cache, cache_last_channel_len)
            # The 5th value (cache_last_channel_len) is a length tensor for the channel cache;
            # it is not needed for decoding so we discard it.
            encoded, encoded_len, self._cache_last_channel, self._cache_last_time, _ = (
                self._model.encoder.cache_aware_stream_step(
                    processed_signal=processed,
                    processed_signal_length=processed_len,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    keep_all_outputs=False,
                )
            )

            if encoded is None or encoded.shape[-1] == 0:
                return []

            # RNNT greedy decode.
            # NeMo versions differ: some return a bare list of Hypothesis objects,
            # others return a (list, secondary_list) tuple. Unpack safely.
            _raw = self._model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
            )
            # If it's a tuple/list-of-lists, the first element is the hypotheses list
            if isinstance(_raw, tuple):
                hypotheses = _raw[0]
            elif isinstance(_raw, list) and len(_raw) > 0 and isinstance(_raw[0], list):
                hypotheses = _raw[0]
            else:
                hypotheses = _raw

        new_words = []
        if hypotheses and len(hypotheses) > 0:
            hyp = hypotheses[0]
            # .text is populated on Hypothesis when return_hypotheses=True
            text = hyp.text.strip() if hasattr(hyp, "text") else ""
            if text:
                words = text.split()
                ms_per_word = self.config.chunk_ms / max(len(words), 1)
                for i, word in enumerate(words):
                    wr = WordResult(
                        text=word,
                        start_ms=start_ms + i * ms_per_word,
                        end_ms=start_ms + (i + 1) * ms_per_word,
                        is_partial=False,
                    )
                    self._word_buffer.append(wr)
                    new_words.append(wr)
                    if self.on_word:
                        self.on_word(wr)

        return new_words

    def _flush_ayah(self):
        """Called on silence — emit completed ayah and reset per-ayah state."""
        if not self._word_buffer:
            self._reset_state()
            return

        full_text = " ".join(w.text for w in self._word_buffer)
        duration_ms = self._ayah_frames * self.config.frame_ms
        elapsed_s = time.time() - self._stream_start_time
        rtf = elapsed_s / max(duration_ms / 1000.0, 0.001)

        result = AyahResult(
            words=list(self._word_buffer),
            full_text=full_text,
            duration_ms=duration_ms,
            rtf=rtf,
        )

        if self.on_ayah:
            self.on_ayah(result)

        self._reset_state()

    def _reset_state(self):
        """Reset per-ayah state. _total_frames is intentionally NOT reset."""
        self._cache_last_channel = None
        self._cache_last_time = None
        self._audio_buffer = np.array([], dtype=np.float32)
        self._word_buffer = []
        self._ayah_frames = 0
        self._silence_counter = 0
        self._stream_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Microphone input thread
# ─────────────────────────────────────────────────────────────────────────────

class MicrophoneStream:
    """
    Reads from the default microphone and pushes frames to StreamingASR.

    Usage:
        mic = MicrophoneStream(asr).start()
        time.sleep(30)   # record for 30s
        mic.stop()
    """

    def __init__(self, asr: StreamingASR):
        self.asr = asr
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._q: queue.Queue = queue.Queue()

    def start(self) -> "MicrophoneStream":
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.asr.config.sample_rate,
            input=True,
            frames_per_buffer=self.asr.config.frame_samples,
            stream_callback=self._callback,
        )
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self._stream.start_stream()
        return self

    def stop(self):
        self._stop_event.set()
        if hasattr(self, "_stream"):
            self._stream.stop_stream()
            self._stream.close()
        if hasattr(self, "_pa"):
            self._pa.terminate()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio
        self._q.put(np.frombuffer(in_data, dtype=np.int16).copy())
        return (None, pyaudio.paContinue)

    def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self._q.get(timeout=0.1)
                self.asr.process_frame(frame)
            except queue.Empty:
                continue


# ─────────────────────────────────────────────────────────────────────────────
# Terminal demo
# ─────────────────────────────────────────────────────────────────────────────

def live_demo(model_path: str):
    """
    Live terminal demo — speak and see words appear in real time.
    Press Ctrl+C to stop.
    """
    print("\n" + "═" * 60)
    print("  Quran FastConformer — Live Streaming Demo")
    print("═" * 60)

    def on_word(w: WordResult):
        print(w.text, end=" ", flush=True)

    def on_ayah(a: AyahResult):
        print(f"\n\n✓ [{a.duration_ms:.0f}ms | RTF={a.rtf:.3f}] {a.full_text}\n")
        print("─" * 60)

    config = StreamingConfig(model_path=model_path)
    asr = StreamingASR(config, on_word=on_word, on_ayah=on_ayah).load()

    print("\nSpeak now (Ctrl+C to stop)...\n" + "─" * 60 + "\n")

    mic = MicrophoneStream(asr).start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        mic.stop()


# ─────────────────────────────────────────────────────────────────────────────
# File transcription (offline test — uses full bilateral context)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_file(model_path: str, audio_path: str):
    """
    Transcribe a .wav file using OFFLINE (full bilateral) mode.
    Use this to verify checkpoint accuracy before going live.

    The model is loaded fresh here so it does not affect any StreamingASR
    instance you may have running.
    """
    import torch
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    print(f"Loading model for offline transcription: {model_path}")

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        model_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Ensure symmetric padding before using bilateral attention
    n_fixed = _fix_conv_padding_symmetric(model)
    print(f"  ✓ Conv padding verified symmetric ({n_fixed} layers adjusted)")

    # Full bilateral context for best accuracy
    model.change_attention_model(
        self_attention_model="rel_pos",
        att_context_size=[-1, -1],
    )
    with open_dict(model.cfg):
        model.cfg.encoder.conv_context_size = None

    model.eval()

    print(f"Transcribing: {audio_path}")
    result = model.transcribe([audio_path])
    text = result[0].text if hasattr(result[0], "text") else str(result[0])
    print(f"\nTranscription: {text}\n")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI WebSocket server
# ─────────────────────────────────────────────────────────────────────────────

def run_server(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """
    FastAPI server with:
      POST /transcribe      — upload a .wav file, get transcription back
      WS   /stream          — send raw PCM int16 frames, receive words as JSON

    WebSocket protocol:
      Client sends: raw bytes (int16 PCM, 1280 bytes = 80ms at 16kHz)
      Server sends: JSON {"type": "word",  "text": "...", "start_ms": 0}
                 or JSON {"type": "ayah",  "text": "...", "duration_ms": 0, "rtf": 0}

    Each WebSocket connection gets its OWN StreamingASR instance so that
    multiple clients don't share (and corrupt) the same encoder cache.
    The underlying model weights are shared read-only.
    """
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
        from fastapi.responses import JSONResponse
        import uvicorn
        import tempfile
        import os
    except ImportError:
        print("Install FastAPI: pip install fastapi uvicorn python-multipart")
        return

    # Shared model weights (read-only after startup)
    _shared_model = {"instance": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load model once at startup; clean up at shutdown."""
        import torch
        import nemo.collections.asr as nemo_asr

        print(f"Loading model: {model_path}")
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            model_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
        _fix_conv_padding_symmetric(model)
        model.eval()
        _shared_model["instance"] = model
        print("Model ready.")
        yield
        # Shutdown
        _shared_model["instance"] = None

    app = FastAPI(title="Quran Streaming ASR", lifespan=lifespan)

    @app.post("/transcribe")
    async def transcribe_endpoint(file: UploadFile = File(...)):
        """Upload a WAV file, returns {"text": "..."}."""
        model = _shared_model["instance"]
        if model is None:
            return JSONResponse({"error": "Model not loaded"}, status_code=503)

        from omegaconf import open_dict

        # Restore bilateral context for highest accuracy
        _fix_conv_padding_symmetric(model)
        model.change_attention_model("rel_pos", [-1, -1])
        with open_dict(model.cfg):
            model.cfg.encoder.conv_context_size = None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            result = model.transcribe([tmp_path])
            text = result[0].text if hasattr(result[0], "text") else str(result[0])
        finally:
            os.unlink(tmp_path)

        return JSONResponse({"text": text})

    @app.websocket("/stream")
    async def stream_endpoint(ws: WebSocket):
        """
        Each connection gets its own StreamingASR instance (own cache).
        The underlying model weights are shared via _shared_model["instance"].
        """
        await ws.accept()
        print(f"WebSocket client connected: {ws.client}")

        pending_events: queue.Queue = queue.Queue()

        def on_word(w: WordResult):
            pending_events.put({
                "type": "word",
                "text": w.text,
                "start_ms": w.start_ms,
            })

        def on_ayah(a: AyahResult):
            pending_events.put({
                "type": "ayah",
                "text": a.full_text,
                "duration_ms": a.duration_ms,
                "rtf": a.rtf,
            })

        # Each connection owns its StreamingASR with its own cache state
        config = StreamingConfig(model_path=model_path)
        asr = StreamingASR(config, on_word=on_word, on_ayah=on_ayah)

        # Inject shared weights — do NOT call load(), that would reload from disk
        asr._model = _shared_model["instance"]
        # Apply streaming attention to the shared model for this connection.
        # NOTE: if you need simultaneous offline + streaming, run separate servers.
        _apply_streaming_attention(
            asr._model,
            config.att_context_left,
            config.att_context_right,
        )
        asr._reset_state()

        try:
            while True:
                raw = await ws.receive_bytes()
                frame = np.frombuffer(raw, dtype=np.int16)
                asr.process_frame(frame)

                # Drain and forward any pending events
                while not pending_events.empty():
                    event = pending_events.get_nowait()
                    await ws.send_text(json.dumps(event, ensure_ascii=False))

        except WebSocketDisconnect:
            print(f"WebSocket client disconnected: {ws.client}")

    uvicorn.run(app, host=host, port=port)


# ─────────────────────────────────────────────────────────────────────────────
# Ayah alignment helper (for recitation correction apps)
# ─────────────────────────────────────────────────────────────────────────────

def align_to_ayah(hypothesis: str, reference: str) -> dict:
    """
    Compare a transcribed ayah to the expected reference text.
    Returns word-level status for UI highlighting.

    Returns:
        {
            "wer": float,
            "words": [
                {"word": "...", "status": "correct" | "substitution" | "insertion" | "deletion"},
                ...
            ]
        }
    """
    hyp_words = hypothesis.strip().split()
    ref_words = reference.strip().split()
    n, m = len(ref_words), len(hyp_words)

    # Standard edit-distance DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution preferred over pure del/ins for equal cost
                sub  = dp[i - 1][j - 1] + 1
                del_ = dp[i - 1][j]     + 1
                ins  = dp[i][j - 1]     + 1
                dp[i][j] = min(sub, del_, ins)

    # Traceback
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            aligned.append({"word": ref_words[i - 1], "status": "correct"})
            i -= 1; j -= 1
        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            # Substitution
            aligned.append({"word": ref_words[i - 1], "status": "substitution"})
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion (ref word missing from hyp)
            aligned.append({"word": ref_words[i - 1], "status": "deletion"})
            i -= 1
        else:
            # Insertion (hyp word not in ref)
            aligned.append({"word": hyp_words[j - 1], "status": "insertion"})
            j -= 1

    aligned.reverse()
    wer = dp[n][m] / max(n, 1)
    return {"wer": wer, "words": aligned}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quran FastConformer Streaming ASR")
    parser.add_argument(
        "--model",
        default="quran_asr/checkpoints/phase1_top3/phase1_top3_wer0.0038.nemo",
        help="Path to .nemo checkpoint or HuggingFace repo ID",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "server", "transcribe"],
        default="demo",
        help="demo: live mic | server: FastAPI | transcribe: single file",
    )
    parser.add_argument("--file", default=None, help="Audio file for --mode transcribe")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "demo":
        live_demo(args.model)
    elif args.mode == "transcribe":
        if not args.file:
            print("Provide --file path.wav for transcribe mode")
        else:
            transcribe_file(args.model, args.file)
    elif args.mode == "server":
        print(f"Starting server on {args.host}:{args.port}")
        run_server(args.model, host=args.host, port=args.port)