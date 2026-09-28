from pathlib import Path

import av


class VideoReader:
    def frames(self, path: Path, sample_rate=4):
        """Decode sequentially; presentation timestamps also support variable FPS."""
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("В файле нет видеопотока")
            next_time = 0.0
            for frame in container.decode(video=0):
                if frame.time is None:
                    continue
                time = float(frame.time)
                if time + 1e-6 < next_time:
                    continue
                yield time, frame.to_ndarray(format="bgr24")
                next_time = time + 1 / sample_rate
