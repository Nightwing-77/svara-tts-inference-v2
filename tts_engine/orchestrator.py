
from __future__ import annotations
import os
from typing import Iterator, AsyncIterator, List, Optional, Literal, Union
import concurrent.futures
import asyncio
import logging
import torch
from .transports import VLLMEmbeddedTransport
from .mapper import NeuCodecMapper, extract_codebook_token_numbers
from .codec import NeuCodecWrapper, get_or_load_tokenizer
from .encoder import simple_text_to_tokens
from .utils import create_speaker_id
from .buffers import AudioBuffer, SyncFuture, crossfade_pcm
from .utils import chunk_text

logger = logging.getLogger(__name__)


def _detect_max_workers(device: Optional[str] = None) -> int:
    """
    Detect recommended max_workers for NeuCodec decoding.

    When NeuCodec runs on CPU, scale with available CPU cores.
    When NeuCodec runs on GPU, limit workers to avoid GPU contention.
    """
    import os

    # If NeuCodec is on CPU, use half the CPU cores (minimum 2)
    device = device or os.getenv("DEVICE", "cpu")
    if device == "cpu":
        cpu_count = os.cpu_count() or 4
        workers = max(2, cpu_count // 2)
        logger.info(f"NeuCodec on CPU: {cpu_count} cores available, using {workers} workers")
        return workers

    # NeuCodec on GPU — keep workers limited to avoid contention
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            logger.info(f"GPU detected: {props.name}, {vram_gb:.1f}GB VRAM, compute {props.major}.{props.minor}")
            if vram_gb >= 16 and props.major >= 8:
                return 4
        except Exception:
            pass
    return 2


class NeuCodecTTSOrchestrator:
    """
    Sync/Async TTS orchestrator for NeuCodec-based models:
    transport -> mapper -> decoder -> PCM int16 chunks.

    Args:
        transport: The VLLMEmbeddedTransport instance.
        model: The model name (for tokenizer lookup).
        prebuffer_seconds: The number of seconds to prebuffer before yielding audio.
        concurrent_decode: If True, decode concurrently.
        max_workers: The number of workers to use for decoding (None = auto-detect).
        buffer_size: Number of tokens to buffer before decoding (default 100).
        device: Device for NeuCodec decoder (cuda, mps, cpu, or None for auto).
    """
    def __init__(self,
                 transport: VLLMEmbeddedTransport,
                 model: str = "kenpath/qwen3.5-0.8b-stage5",
                 prebuffer_seconds: float = 0.5,
                 concurrent_decode: bool = True,
                 max_workers: Optional[int] = None,
                 buffer_size: int = 100,
                 device: Optional[str] = None):

        self.model_name = model
        self.tokenizer_model = os.getenv("TOKENIZER_MODEL", os.getenv("VLLM_MODEL", "kenpath/qwen3.5-0.8b-stage5"))
        self.tokenizer      = get_or_load_tokenizer(self.tokenizer_model)

        self.transport      = transport
        self.codec      = NeuCodecWrapper(device)
        self.prebuffer_samples = int(self.codec.sample_rate * prebuffer_seconds)
        self.concurrent_decode = concurrent_decode

        # Auto-detect optimal workers based on device
        self.max_workers = max_workers if max_workers is not None else _detect_max_workers(device)

        # Token buffer size
        self.buffer_size = buffer_size

        # Long-text chunking config
        self.max_chunk_chars = 200
        self.crossfade_ms = 50         # Crossfade overlap between chunks

        logger.info(f"Orchestrator: max_workers={self.max_workers}, "
                     f"buffer_size={self.buffer_size}, "
                     f"prebuffer={prebuffer_seconds}s")

    def warmup(self):
        """Run a dummy NeuCodec decode to warm caches."""
        logger.info("Warming up NeuCodec decoder...")
        self.codec.decode_tokens([1] * self.buffer_size)
        logger.info("NeuCodec warmup complete")

    # ------------ SYNC path ------------
    def stream(self,
               text: str,
               audio_reference: Optional[List[int]] = None,
               reference_text: Optional[str] = None,
               speaker_id: Optional[str] = None,
               chunk_size: Optional[int] = None,
               buffer_ms: Optional[int] = None,
               **gen_kwargs) -> Iterator[bytes]:
        """Stream the TTS output, automatically chunking long texts.

        For texts longer than chunk_size, splits at sentence boundaries
        and crossfades between chunks for smooth audio stitching.
        Streams audio progressively within each chunk — only holds back
        the last overlap_ms for crossfading with the next chunk.
        """
        max_chars = chunk_size or self.max_chunk_chars
        prebuf = int(self.codec.sample_rate * buffer_ms / 1000) if buffer_ms is not None else None
        chunks = chunk_text(text, max_len=max_chars)

        if len(chunks) <= 1:
            yield from self._stream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, prebuffer_samples=prebuf, **gen_kwargs)
            return

        logger.info(f"Long text ({len(text)} chars) split into {len(chunks)} chunks")
        overlap_bytes = int(self.codec.sample_rate * self.crossfade_ms / 1000) * 2  # 2 bytes per sample
        prev_tail: Optional[bytes] = None

        for chunk_text_str in chunks:
            is_last_chunk = (chunk_text_str is chunks[-1])
            trailing = bytearray()

            for b in self._stream_one(chunk_text_str, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, prebuffer_samples=prebuf, **gen_kwargs):
                trailing.extend(b)

                if prev_tail is not None:
                    head = bytes(trailing[:overlap_bytes]) if len(trailing) >= overlap_bytes else bytes(trailing)
                    if len(head) >= overlap_bytes:
                        blended = crossfade_pcm(prev_tail, head,
                                                overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                        yield blended
                        trailing = bytearray(trailing[overlap_bytes:])
                        prev_tail = None
                    continue

                if len(trailing) > overlap_bytes:
                    to_yield = bytes(trailing[:-overlap_bytes])
                    trailing = bytearray(trailing[-overlap_bytes:])
                    yield to_yield

            if prev_tail is not None:
                if trailing:
                    blended = crossfade_pcm(prev_tail, bytes(trailing),
                                            overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                    yield blended
                else:
                    yield prev_tail
                prev_tail = None
                trailing = bytearray()

            if not is_last_chunk and len(trailing) > overlap_bytes:
                yield bytes(trailing[:-overlap_bytes])
                prev_tail = bytes(trailing[-overlap_bytes:])
            elif not is_last_chunk:
                prev_tail = bytes(trailing) if trailing else None
            else:
                if trailing:
                    yield bytes(trailing)

    def _stream_one(self,
                    text: str,
                    audio_reference: Optional[List[int]] = None,
                    reference_text: Optional[str] = None,
                    speaker_id: Optional[str] = None,
                    prebuffer_samples: Optional[int] = None,
                    **gen_kwargs) -> Iterator[bytes]:

        prompt = simple_text_to_tokens(
            text=text,
            tokenizer=self.tokenizer,
            return_decoded=True
        )

        logger.info(f"Final prompt before inference: {len(prompt)} chars")
        logger.debug(f"Full prompt: {prompt}")

        mapper = NeuCodecMapper(buffer_size=self.buffer_size)
        audio_buf = AudioBuffer(prebuffer_samples if prebuffer_samples is not None else self.prebuffer_samples)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[concurrent.futures.Future] = []

        def decode(tokens: List[int]) -> bytes:
            return self.codec.decode_tokens(tokens)

        def submit(tokens: List[int]):
            return executor.submit(decode, tokens) if executor else SyncFuture(decode(tokens))

        try:
            for token_text in self.transport.stream(prompt, **gen_kwargs):
                for n in extract_codebook_token_numbers(token_text):
                    tokens = mapper.feed_raw(n)
                    if tokens is not None:
                        pending.append(submit(tokens))

                    # Yield when we have enough pending
                    while len(pending) > 2:
                        result = audio_buf.process(pending.pop(0).result())
                        if result:
                            yield result

            # Flush remaining
            for fut in pending:
                result = audio_buf.process(fut.result())
                if result:
                    yield result

            # Flush any remaining tokens from mapper
            remaining_tokens = mapper.flush()
            if remaining_tokens:
                final_audio = decode(remaining_tokens)
                result = audio_buf.process(final_audio)
                if result:
                    yield result

            # Flush any prebuffered audio that never hit the threshold
            tail = audio_buf.flush()
            if tail:
                yield tail
        finally:
            if executor:
                executor.shutdown(wait=True)

    # ------------ ASYNC path ------------
    async def astream(self,
                      text: str,
                      audio_reference: Optional[List[int]] = None,
                      reference_text: Optional[str] = None,
                      speaker_id: Optional[str] = None,
                      chunk_size: Optional[int] = None,
                      buffer_ms: Optional[int] = None,
                      **gen_kwargs) -> AsyncIterator[bytes]:
        """Async stream the TTS output, automatically chunking long texts.

        For texts longer than chunk_size, splits at sentence boundaries
        and crossfades between chunks for smooth audio stitching.
        Streams audio progressively within each chunk — only holds back
        the last overlap_ms for crossfading with the next chunk.
        """
        max_chars = chunk_size or self.max_chunk_chars
        prebuf = int(self.codec.sample_rate * buffer_ms / 1000) if buffer_ms is not None else None
        chunks = chunk_text(text, max_len=max_chars)

        if len(chunks) <= 1:
            async for b in self._astream_one(text, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, prebuffer_samples=prebuf, **gen_kwargs):
                yield b
            return

        logger.info(f"Long text ({len(text)} chars) split into {len(chunks)} chunks")
        overlap_bytes = int(self.codec.sample_rate * self.crossfade_ms / 1000) * 2  # 2 bytes per sample
        prev_tail: Optional[bytes] = None

        for chunk_text_str in chunks:
            is_last_chunk = (chunk_text_str is chunks[-1])
            trailing = bytearray()

            async for b in self._astream_one(chunk_text_str, audio_reference=audio_reference, reference_text=reference_text, speaker_id=speaker_id, prebuffer_samples=prebuf, **gen_kwargs):
                trailing.extend(b)

                # For the first piece of the first non-first chunk, crossfade with prev_tail
                if prev_tail is not None:
                    head = bytes(trailing[:overlap_bytes]) if len(trailing) >= overlap_bytes else bytes(trailing)
                    if len(head) >= overlap_bytes:
                        blended = crossfade_pcm(prev_tail, head,
                                                overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                        yield blended
                        trailing = bytearray(trailing[overlap_bytes:])
                        prev_tail = None
                    # else: keep accumulating until we have enough for crossfade
                    continue

                # Stream the middle: yield everything except the last overlap_bytes
                if len(trailing) > overlap_bytes:
                    to_yield = bytes(trailing[:-overlap_bytes])
                    trailing = bytearray(trailing[-overlap_bytes:])
                    yield to_yield

            # After chunk finishes, handle any remaining crossfade that didn't have enough data
            if prev_tail is not None:
                # Chunk was very short, just crossfade what we have
                if trailing:
                    blended = crossfade_pcm(prev_tail, bytes(trailing),
                                            overlap_ms=self.crossfade_ms, sample_rate=self.codec.sample_rate)
                    yield blended
                else:
                    yield prev_tail
                prev_tail = None
                trailing = bytearray()

            # Hold back the tail for crossfading with the next chunk
            if not is_last_chunk and len(trailing) > overlap_bytes:
                yield bytes(trailing[:-overlap_bytes])
                prev_tail = bytes(trailing[-overlap_bytes:])
            elif not is_last_chunk:
                prev_tail = bytes(trailing) if trailing else None
            else:
                # Last chunk — yield everything
                if trailing:
                    yield bytes(trailing)


    async def _astream_one(self,
                           text: str,
                           audio_reference: Optional[List[int]] = None,
                           reference_text: Optional[str] = None,
                           speaker_id: Optional[str] = None,
                           prebuffer_samples: Optional[int] = None,
                           **gen_kwargs) -> AsyncIterator[bytes]:

        prompt = simple_text_to_tokens(
            text=text,
            tokenizer=self.tokenizer,
            return_decoded=True
        )

        logger.info(f"Final prompt before inference: {len(prompt)} chars")
        logger.debug(f"Full prompt: {prompt}")

        mapper = NeuCodecMapper(buffer_size=self.buffer_size)
        audio_buf = AudioBuffer(prebuffer_samples if prebuffer_samples is not None else self.prebuffer_samples)
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.concurrent_decode else None
        pending: List[asyncio.Task] = []

        def decode(tokens: List[int]) -> bytes:
            return self.codec.decode_tokens(tokens)

        async def submit_async(tokens: List[int]) -> bytes:
            if executor:
                return await loop.run_in_executor(executor, decode, tokens)
            else:
                return decode(tokens)

        try:
            async for token_text in self.transport.astream(prompt, **gen_kwargs):
                for n in extract_codebook_token_numbers(token_text):
                    tokens = mapper.feed_raw(n)
                    if tokens is not None:
                        pending.append(asyncio.create_task(submit_async(tokens)))

                    # Yield when we have enough pending
                    while len(pending) > 2:
                        result = audio_buf.process(await pending.pop(0))
                        if result:
                            yield result

            # Flush remaining
            for task in pending:
                result = audio_buf.process(await task)
                if result:
                    yield result

            # Flush any remaining tokens from mapper
            remaining_tokens = mapper.flush()
            if remaining_tokens:
                final_audio = decode(remaining_tokens)
                result = audio_buf.process(final_audio)
                if result:
                    yield result

            # Flush any prebuffered audio that never hit the threshold
            tail = audio_buf.flush()
            if tail:
                yield tail
        finally:
            if executor:
                executor.shutdown(wait=True)
