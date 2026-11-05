"""Locate and hide subtitles that are already burned into video frames.

Detection needs no OCR or GPU. It samples frames across the video and looks for
a horizontal band with dense edges that recurs in the same place while its
contents change over the video. Static overlays such as logos keep the same
edges in every sample, so they are rejected.

Once the area is known, a second pass scans it through the whole video, so the
mask is on only while there is text to hide.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence

from PIL import Image, ImageChops, ImageFilter, ImageStat

from ..entities import VideoInfo
from ..utils.logger import setup_logger
from ..utils.video_utils import get_video_info

logger = setup_logger("burned_subs")

ACTION_NONE = "none"
ACTION_BLUR = "blur"
ACTION_BOX = "box"
ACTION_AVOID = "avoid"
ACTIONS = (ACTION_NONE, ACTION_BLUR, ACTION_BOX, ACTION_AVOID)

REGION_AUTO = "auto"

TIMING_DETECTED = "detected"
TIMING_ALWAYS = "always"
TIMINGS = (TIMING_DETECTED, TIMING_ALWAYS)

TimeRange = tuple[float, float]

_SAMPLE_COUNT = 24
_ANALYSIS_WIDTH = 640
_EDGE_THRESHOLD = 70
_MIN_ROW_DENSITY = 0.04
_ROW_DENSITY_OVER_MEDIAN = 2.0
_MIN_FRAME_RATIO = 0.25
_MIN_FRAMES = 3
_MAX_GAP_RATIO = 0.03
_MIN_BAND_RATIO = 0.025
_MAX_BAND_RATIO = 0.24
_MIN_COLUMN_DENSITY = 0.08
_MAX_WORD_GAP_RATIO = 0.04
_MIN_WIDTH_RATIO = 0.06
_STATIC_COMMON_RATIO = 0.7
_BOTTOM_SEARCH_START = 0.5
_TOP_SEARCH_END = 0.2
_AVOID_GAP_REFERENCE = 12
_MASK_GAP_REFERENCE = 8
_SCAN_FPS = 2
_SCAN_EDGE_THRESHOLD = 150
_MIN_ACTIVE_DENSITY = 0.005
_ACTIVE_PAD_SECONDS = 0.5
_ACTIVE_GAP_SECONDS = 1.0
# Keeps the FFmpeg command within the Windows command line limit.
_MAX_ACTIVE_RANGES = 400


@dataclass(frozen=True)
class Region:
    """A rectangle in video pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class FractionalRegion:
    """A rectangle expressed as fractions of the frame size."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class BurnedSubsPlan:
    """What the renderer must do to deal with existing burned subtitles."""

    region: Region
    pre_filter: Optional[str]
    margin_bottom: Optional[int]
    active_ranges: Optional[tuple[TimeRange, ...]] = None


def parse_region_values(spec: str) -> tuple[float, float, float, float]:
    """Parse "x,y,w,h" and check its shape, without knowing the video size.

    Raises:
        ValueError: if the text is not four non-negative numbers forming a
            non-empty rectangle.
    """
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Region must be 'x,y,w,h', got: {spec}")
    try:
        x, y, width, height = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Region must contain only numbers, got: {spec}") from exc
    if min(x, y, width, height) < 0 or width == 0 or height == 0:
        raise ValueError(f"Region must be non-negative with a non-zero size: {spec}")
    return x, y, width, height


def parse_region(spec: str, video_width: int, video_height: int) -> Region:
    """Convert "x,y,w,h" to pixels.

    All four values in 0-1 are read as fractions of the frame. Otherwise they
    are read as pixels.
    """
    x, y, width, height = parse_region_values(spec)
    if max(x, y, width, height) <= 1:
        return _to_pixels(
            FractionalRegion(x, y, width, height), video_width, video_height
        )
    return _clamp(Region(int(x), int(y), int(width), int(height)), video_width, video_height)


def _to_pixels(
    region: FractionalRegion, video_width: int, video_height: int
) -> Region:
    return _clamp(
        Region(
            round(region.x * video_width),
            round(region.y * video_height),
            round(region.width * video_width),
            round(region.height * video_height),
        ),
        video_width,
        video_height,
    )


def _clamp(region: Region, video_width: int, video_height: int) -> Region:
    """Fit a region inside the frame, on even coordinates for 4:2:0 video."""
    x = min(region.x, video_width - 2) & ~1
    y = min(region.y, video_height - 2) & ~1
    width = min(region.width, video_width - x)
    height = min(region.height, video_height - y)
    return Region(x, y, max(width & ~1, 2), max(height & ~1, 2))


def find_subtitle_region(frames: Sequence[Image.Image]) -> Optional[FractionalRegion]:
    """Find a recurring, changing text band in equally sized grayscale frames."""
    if len(frames) < _MIN_FRAMES:
        return None

    width, height = frames[0].size
    edge_maps = [_edge_map(frame) for frame in frames]
    row_profiles = [_row_profile(edges, height) for edges in edge_maps]
    hit_frames = [_text_rows(profile) for profile in row_profiles]

    required = max(_MIN_FRAMES, round(len(frames) * _MIN_FRAME_RATIO))
    recurring = [
        sum(row in hits for hits in hit_frames) >= required for row in range(height)
    ]

    bands = _bands(recurring, height)
    # Subtitles sit at the bottom far more often than at the top.
    bands.sort(key=lambda band: band[1], reverse=True)
    for top, bottom in bands:
        center = (top + bottom) / 2 / height
        if _TOP_SEARCH_END < center < _BOTTOM_SEARCH_START:
            continue
        frames_with_text = [
            edges
            for edges, hits in zip(edge_maps, hit_frames)
            if any(row in hits for row in range(top, bottom))
        ]
        if _is_static(frames_with_text, top, bottom, width):
            logger.debug(f"Ignoring static band at rows {top}-{bottom}")
            continue
        columns = _column_extent(frames_with_text, top, bottom, width)
        if columns is None:
            continue
        return _pad(columns, top, bottom, width, height)
    return None


def _edge_map(
    frame: Image.Image, threshold: int = _EDGE_THRESHOLD
) -> Image.Image:
    return frame.filter(ImageFilter.FIND_EDGES).point(
        lambda value: 255 if value > threshold else 0
    )


def _row_profile(edges: Image.Image, height: int) -> list[float]:
    """Return the share of edge pixels on each row."""
    return [value / 255 for value in edges.resize((1, height), Image.BOX).tobytes()]


def _text_rows(profile: list[float]) -> set[int]:
    median = sorted(profile)[len(profile) // 2]
    threshold = max(_MIN_ROW_DENSITY, median * _ROW_DENSITY_OVER_MEDIAN)
    # FIND_EDGES leaves artefacts on the outermost rows.
    return {
        row
        for row, density in enumerate(profile[:-2])
        if row >= 2 and density >= threshold
    }


def _bands(recurring: list[bool], height: int) -> list[tuple[int, int]]:
    """Merge recurring rows into bands, allowing a gap between two text lines."""
    max_gap = max(1, round(height * _MAX_GAP_RATIO))
    bands: list[list[int]] = []
    for row, is_recurring in enumerate(recurring):
        if not is_recurring:
            continue
        if bands and row - bands[-1][1] <= max_gap:
            bands[-1][1] = row + 1
        else:
            bands.append([row, row + 1])
    return [
        (top, bottom)
        for top, bottom in bands
        if _MIN_BAND_RATIO * height <= bottom - top <= _MAX_BAND_RATIO * height
    ]


def _is_static(
    edge_maps: list[Image.Image], top: int, bottom: int, width: int
) -> bool:
    """Tell a logo from subtitles by the edges every sample has in common.

    A logo's edges are in all samples. Subtitle lines differ, so few edges
    survive across them, even when neighbouring samples show the same line.
    """
    if len(edge_maps) < 2:
        return True
    crops = [edges.crop((0, top, width, bottom)) for edges in edge_maps]
    common = crops[0]
    for crop in crops[1:]:
        common = ImageChops.darker(common, crop)
    mean_density = sum(ImageStat.Stat(crop).mean[0] for crop in crops) / len(crops)
    if not mean_density:
        return True
    return ImageStat.Stat(common).mean[0] / mean_density >= _STATIC_COMMON_RATIO


def _column_extent(
    edge_maps: list[Image.Image], top: int, bottom: int, width: int
) -> Optional[tuple[int, int]]:
    """Return the left and right limits of the text across samples.

    Each sample contributes its largest run of dense columns, which keeps stray
    background edges out. The extremes are trimmed so one outlier cannot widen
    the area, while a long subtitle line still counts.
    """
    extents = []
    for edges in edge_maps:
        profile = edges.crop((0, top, width, bottom)).resize((width, 1), Image.BOX)
        extent = _largest_run(profile.tobytes(), width)
        if extent is not None:
            extents.append(extent)
    if not extents:
        return None
    trim = len(extents) // 10
    left = sorted(extent[0] for extent in extents)[trim]
    right = sorted((extent[1] for extent in extents), reverse=True)[trim]
    if right - left < _MIN_WIDTH_RATIO * width:
        return None
    return left, right


def _largest_run(profile: bytes, width: int) -> Optional[tuple[int, int]]:
    """Return the widest group of dense columns, allowing gaps between words."""
    threshold = _MIN_COLUMN_DENSITY * 255
    max_gap = round(width * _MAX_WORD_GAP_RATIO)
    runs: list[list[int]] = []
    for column, value in enumerate(profile):
        if value < threshold:
            continue
        if runs and column - runs[-1][1] <= max_gap:
            runs[-1][1] = column + 1
        else:
            runs.append([column, column + 1])
    if not runs:
        return None
    left, right = max(runs, key=lambda run: run[1] - run[0])
    return left, right


def _pad(
    columns: tuple[int, int], top: int, bottom: int, width: int, height: int
) -> FractionalRegion:
    left, right = columns
    pad_y = max(2, round((bottom - top) * 0.15))
    pad_x = round(width * 0.02)
    left = max(0, left - pad_x)
    right = min(width, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(height, bottom + pad_y)
    return FractionalRegion(
        left / width,
        top / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def _probe_video(video_path: str) -> VideoInfo:
    info = get_video_info(video_path)
    if info is None or not info.width or not info.height:
        raise ValueError(f"Could not read the video size of: {video_path}")
    return info


def _no_window_flag() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _sample_frames(
    video_path: str, video_width: int, video_height: int, duration: float
) -> list[Image.Image]:
    """Grab grayscale frames spread evenly across the video."""
    width = _ANALYSIS_WIDTH
    height = max(2, round(video_height * width / video_width)) & ~1
    frame_size = width * height
    frames = []
    for index in range(_SAMPLE_COUNT):
        timestamp = duration * (index + 0.5) / _SAMPLE_COUNT
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:{height},format=gray",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            creationflags=_no_window_flag(),
        )
        if result.returncode == 0 and len(result.stdout) >= frame_size:
            frames.append(
                Image.frombytes("L", (width, height), result.stdout[:frame_size])
            )
    return frames


def _strong_edge_density(frame: Image.Image) -> float:
    """Return the share of strong edges, which outlined text has and textures lack."""
    width, height = frame.size
    # FIND_EDGES leaves artefacts on the outermost pixels.
    edges = _edge_map(frame, _SCAN_EDGE_THRESHOLD).crop(
        (2, 2, width - 2, height - 2)
    )
    return ImageStat.Stat(edges).mean[0] / 255


def _scan_text_density(
    video_path: str, region: Region, video_width: int
) -> list[float]:
    """Measure a region's strong edges through the video at _SCAN_FPS."""
    scale = _ANALYSIS_WIDTH / video_width
    width = max(8, round(region.width * scale)) & ~1
    height = max(8, round(region.height * scale)) & ~1
    frame_size = width * height
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-an",
            "-sn",
            "-vf",
            f"fps={_SCAN_FPS},"
            f"crop={region.width}:{region.height}:{region.x}:{region.y},"
            f"scale={width}:{height},format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=_no_window_flag(),
    )
    assert process.stdout is not None
    densities = []
    with process.stdout:
        while len(data := process.stdout.read(frame_size)) == frame_size:
            densities.append(
                _strong_edge_density(Image.frombytes("L", (width, height), data))
            )
    if process.wait() != 0:
        raise RuntimeError(f"FFmpeg could not scan the video: {video_path}")
    return densities


def find_active_ranges(
    densities: Sequence[float], fps: float = _SCAN_FPS
) -> list[TimeRange]:
    """Turn per-sample text measurements into the time ranges to mask."""
    ranges: list[list[float]] = []
    for index, density in enumerate(densities):
        if density < _MIN_ACTIVE_DENSITY:
            continue
        start = max(0.0, index / fps - _ACTIVE_PAD_SECONDS)
        end = (index + 1) / fps + _ACTIVE_PAD_SECONDS
        if ranges and start - ranges[-1][1] <= _ACTIVE_GAP_SECONDS:
            ranges[-1][1] = end
        else:
            ranges.append([start, end])
    return _limit_ranges([(start, end) for start, end in ranges])


def _limit_ranges(
    ranges: list[TimeRange], limit: int = _MAX_ACTIVE_RANGES
) -> list[TimeRange]:
    """Merge across the smallest gaps, so the mask errs towards staying on."""
    if len(ranges) <= limit:
        return ranges
    gaps = [(ranges[i + 1][0] - ranges[i][1], i) for i in range(len(ranges) - 1)]
    kept_gaps = {index for _, index in sorted(gaps, reverse=True)[: limit - 1]}
    merged = [ranges[0]]
    for index, current in enumerate(ranges[1:]):
        if index in kept_gaps:
            merged.append(current)
        else:
            merged[-1] = (merged[-1][0], current[1])
    return merged


def detect_burned_subtitles(video_path: str) -> Optional[Region]:
    """Return the area of a video's burned subtitles, if there are any."""
    info = _probe_video(video_path)
    frames = _sample_frames(
        video_path, info.width, info.height, info.duration_seconds
    )
    found = find_subtitle_region(frames)
    if found is None:
        return None
    return _to_pixels(found, info.width, info.height)


def _enable_option(active_ranges: Optional[Sequence[TimeRange]]) -> str:
    """Build the filter option that limits an effect to some time ranges."""
    if active_ranges is None:
        return ""
    expression = "+".join(
        f"between(t,{start:.2f},{end:.2f})" for start, end in active_ranges
    )
    return f":enable='{expression}'"


def build_mask_filter(
    action: str,
    region: Region,
    active_ranges: Optional[Sequence[TimeRange]] = None,
) -> Optional[str]:
    """Build the FFmpeg filter that hides a region.

    The result has one video input and one video output, so it can lead both a
    -vf chain and a -filter_complex graph. With active_ranges, the region is
    hidden only during those times. Without them it is hidden throughout.
    """
    enable = _enable_option(active_ranges)
    if action == ACTION_BOX:
        return (
            f"drawbox=x={region.x}:y={region.y}:w={region.width}:h={region.height}"
            f":color=black:t=fill{enable}"
        )
    if action == ACTION_BLUR:
        luma_radius = max(2, min(region.height, region.width) // 3)
        chroma_radius = max(1, luma_radius // 2)
        # Without format, hardware-decoded NV12 frames lose their chroma in the
        # overlay and the area turns green.
        return (
            "format=yuv420p,split=2[vc_base][vc_mask];"
            f"[vc_mask]crop={region.width}:{region.height}:{region.x}:{region.y},"
            f"boxblur={luma_radius}:2:{chroma_radius}:2[vc_blur];"
            f"[vc_base][vc_blur]overlay={region.x}:{region.y}{enable}"
        )
    return None


def subtitle_margin_bottom(
    action: str, region: Region, video_height: int, reference_height: int = 720
) -> Optional[int]:
    """Return the bottom margin that places new subtitles for this action.

    The value uses the renderers' reference height. None keeps the style's own
    margin, which is right when the old subtitles are in the top half.
    """
    if region.y + region.height / 2 < video_height / 2:
        return None
    if action == ACTION_AVOID:
        distance = video_height - region.y
        gap = _AVOID_GAP_REFERENCE
    else:
        distance = video_height - region.bottom
        gap = _MASK_GAP_REFERENCE
    return round(distance * reference_height / video_height) + gap


def plan_burned_subs(
    video_path: str,
    action: str,
    region_spec: str,
    timing: str = TIMING_DETECTED,
) -> Optional[BurnedSubsPlan]:
    """Work out how to treat a video's burned subtitles.

    Returns None when the action is "none" or no text is found to hide.
    """
    if action == ACTION_NONE:
        return None
    info = _probe_video(video_path)
    if region_spec == REGION_AUTO:
        region = detect_burned_subtitles(video_path)
        if region is None:
            return None
    else:
        region = parse_region(region_spec, info.width, info.height)

    active_ranges = None
    if timing == TIMING_DETECTED and action in (ACTION_BLUR, ACTION_BOX):
        densities = _scan_text_density(video_path, region, info.width)
        active_ranges = tuple(find_active_ranges(densities))
        if not active_ranges:
            return None

    return BurnedSubsPlan(
        region=region,
        pre_filter=build_mask_filter(action, region, active_ranges),
        margin_bottom=subtitle_margin_bottom(action, region, info.height),
        active_ranges=active_ranges,
    )
