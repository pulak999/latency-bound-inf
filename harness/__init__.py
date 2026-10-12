"""Latency and throughput harness for edge inference.

One run = one engine, one model, one input stream. The harness fires requests,
records when each answer arrives, samples GPU power on the same clock, and
writes everything under one run directory so the report can be rebuilt from
the logs alone.

All timestamps are `time.monotonic()` seconds unless a name says otherwise.
"""
