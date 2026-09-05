"""Record the window to a video file, without going via a disk full of frames.

A windowed run is a hundred minutes at sixty frames a second: 360 000 frames,
which as PNGs is gigabytes on disk for a video that ends up a few dozen
megabytes. So frames go straight down a pipe into ffmpeg and nothing is
written twice.

Two things make the result watchable rather than merely long. `every` keeps one
frame in N, so a hundred minutes of training becomes a few minutes of video
that still shows every generation. And with `--headless` the run is not capped
at sixty frames a second, so recording takes as long as the drawing takes
rather than as long as the training would have taken to watch.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pygame

logger = logging.getLogger(__name__)


class Recorder:
    """Pipes rendered frames into ffmpeg as they are drawn."""

    def __init__(
        self,
        path: Path,
        size: Tuple[int, int],
        fps: int = 30,
        every: int = 30,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise SystemExit(
                "Recording needs ffmpeg on the PATH. Install it with "
                "`brew install ffmpeg`, or drop --record."
            )
        self.every = max(1, every)
        self.path = path
        self.frames = 0
        self._seen = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        # `-pix_fmt yuv420p` because without it the file plays in ffplay and
        # nowhere else; h264 with an odd dimension is likewise unplayable, hence
        # the scale filter rounding both sides down to even.
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
                "-i", "-",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        logger.info("Recording every %d frames to %s", self.every, path)

    def capture(self, surface: pygame.Surface) -> None:
        """Offer a frame; one in `every` is actually written."""
        self._seen += 1
        if self._seen % self.every or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(pygame.image.tostring(surface, "RGB"))
            self.frames += 1
        except BrokenPipeError:  # ffmpeg died; carry on training regardless
            logger.error("ffmpeg stopped; the recording ends at %d frames", self.frames)
            self._proc.stdin = None

    def close(self) -> Optional[Path]:
        """Finish the file. Returns its path, or None if nothing was written."""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()
        if not self.frames:
            return None
        logger.info("Wrote %d frames to %s", self.frames, self.path)
        return self.path


__all__ = ["Recorder"]
