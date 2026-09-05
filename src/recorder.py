"""Record a run to a video, without a disk full of frames and without a slideshow.

A hundred minutes of training cannot become five minutes of video by keeping
one frame in twenty: the cars then jump a whole second between frames and the
motion — which is the only thing worth watching — is gone. Compression has to
come from leaving whole stretches out, not from thinning the ones kept.

So the run is filmed in clips. Every few generations, the first seconds of each
circuit are recorded at the full frame rate and played back at the same rate,
so the driving inside a clip is exactly what it looked like; between clips the
video cuts forward. The panel carries the generation number, so the cuts read
as progress rather than as glitches.

Frames go straight down a pipe into ffmpeg. Writing them as PNGs first would be
gigabytes on disk for a file that ends up a few dozen megabytes.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pygame

logger = logging.getLogger(__name__)

# Quality settings for a screen recording rather than for camera footage: thin
# lines and small text are what h264 destroys first, and they are most of the
# picture here. crf 16 is visually lossless for this material; `slow` costs
# encoder time that a headless run has to spare, since it is not waiting on a
# 60 fps display anyway.
_CRF = "16"
_PRESET = "slow"


class Recorder:
    """Pipes rendered frames into ffmpeg as they are drawn."""

    def __init__(self, path: Path, size: Tuple[int, int], fps: int = 60) -> None:
        if shutil.which("ffmpeg") is None:
            raise SystemExit(
                "Recording needs ffmpeg on the PATH. Install it with "
                "`brew install ffmpeg`, or drop --record."
            )
        self.path = path
        self.fps = fps
        self.frames = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
                "-i", "-",
                # h264 cannot encode an odd dimension, and without yuv420p the
                # file plays in ffplay and in almost nothing else.
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", _PRESET, "-crf", _CRF,
                "-pix_fmt", "yuv420p",
                # Lets a player seek and start mid-file rather than reading the
                # whole thing first.
                "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        logger.info("Recording to %s at %d fps", path, fps)

    def capture(self, surface: pygame.Surface) -> None:
        """Write one frame. The caller decides which frames are worth keeping."""
        if self._proc.stdin is None:
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
        size = self.path.stat().st_size / 1e6 if self.path.exists() else 0.0
        logger.info(
            "Video: %s — %d frames, %.0f seconds, %.0f MB",
            self.path, self.frames, self.frames / self.fps, size,
        )
        return self.path


__all__ = ["Recorder"]
