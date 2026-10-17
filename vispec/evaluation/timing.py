"""Timing for the first authoritative output token, before text decoding."""

import time

import torch


class FirstTokenTimer:
    """Start immediately before generation; record token readiness once."""

    def __init__(self):
        self.started = time.perf_counter()
        self.seconds = None

    def mark(self, token):
        if self.seconds is None:
            # Waiting on the token's device includes asynchronous GPU work.
            if token.is_cuda:
                torch.cuda.synchronize(token.device)
            self.seconds = time.perf_counter() - self.started
