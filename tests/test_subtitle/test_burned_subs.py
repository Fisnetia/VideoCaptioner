"""Tests for detecting and hiding subtitles burned into video frames."""

import random
import shutil
import subprocess

import pytest
from PIL import Image, ImageDraw, ImageFont

from videocaptioner.cli.config import build_config
from videocaptioner.cli.validators import validate_burned_subs
from videocaptioner.core.subtitle.burned_subs import (
    ACTION_AVOID,
    ACTION_BLUR,
    ACTION_BOX,
    ACTION_NONE,
    Region,
    build_mask_filter,
    detect_burned_subtitles,
    find_subtitle_region,
    parse_region,
    parse_region_values,
    plan_burned_subs,
    subtitle_margin_bottom,
)
from videocaptioner.core.utils.video_utils import (
    _list_subtitle_codecs,
    _soft_subtitle_stream_args,
    add_subtitles,
)

WIDTH, HEIGHT = 640, 360
LINES = [
    "The quick brown fox jumps over",
    "Pack my box with five dozen jugs",
    "How vexingly quick daft zebras jump",
    "Sphinx of black quartz judge my vow",
    "Two driven jocks help fax my big quiz",
    "Five quacking zephyrs jolt my wax bed",
]

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def _background(seed: int) -> Image.Image:
    """A soft, changing picture without text."""
    rng = random.Random(seed)
    image = Image.new("L", (WIDTH, HEIGHT), rng.randint(40, 120))
    draw = ImageDraw.Draw(image)
    for _ in range(6):
        x, y = rng.randrange(WIDTH), rng.randrange(HEIGHT)
        size = rng.randint(30, 120)
        draw.ellipse((x, y, x + size, y + size), fill=rng.randint(30, 150))
    return image


def _with_text(frame: Image.Image, text: str, y: int) -> Image.Image:
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=22)
    left = (WIDTH - draw.textlength(text, font=font)) / 2
    draw.text((left, y), text, font=font, fill=255, stroke_width=2, stroke_fill=0)
    return frame


def _subtitled_frames(y: int, count: int = 24) -> list[Image.Image]:
    return [
        _with_text(_background(index), LINES[index % len(LINES)], y)
        for index in range(count)
    ]


class TestFindSubtitleRegion:
    def test_finds_bottom_subtitles(self):
        region = find_subtitle_region(_subtitled_frames(y=300))
        assert region is not None
        assert region.y <= 300 / HEIGHT <= region.y + region.height
        assert region.y + region.height <= 345 / HEIGHT
        assert region.width < 0.9

    def test_finds_subtitles_shown_in_a_third_of_frames(self):
        frames = [_background(index) for index in range(24)]
        for index in range(0, 24, 3):
            _with_text(frames[index], LINES[index % len(LINES)], 300)
        assert find_subtitle_region(frames) is not None

    def test_finds_subtitles_held_across_neighbouring_samples(self):
        frames = [
            _with_text(_background(index), LINES[index // 4], 300)
            for index in range(24)
        ]
        assert find_subtitle_region(frames) is not None

    def test_finds_two_line_subtitles_as_one_band(self):
        frames = _subtitled_frames(y=280)
        for index, frame in enumerate(frames):
            _with_text(frame, LINES[(index + 1) % len(LINES)], 310)
        region = find_subtitle_region(frames)
        assert region is not None
        assert region.y <= 282 / HEIGHT
        assert region.y + region.height >= 332 / HEIGHT

    def test_finds_top_subtitles(self):
        region = find_subtitle_region(_subtitled_frames(y=20))
        assert region is not None
        assert region.y < 0.15

    def test_prefers_bottom_band_when_both_exist(self):
        frames = _subtitled_frames(y=300)
        for index, frame in enumerate(frames):
            _with_text(frame, LINES[(index + 2) % len(LINES)], 20)
        region = find_subtitle_region(frames)
        assert region is not None
        assert region.y > 0.5

    def test_ignores_video_without_text(self):
        assert find_subtitle_region([_background(i) for i in range(24)]) is None

    def test_ignores_static_logo(self):
        frames = [
            _with_text(_background(index), "CHANNEL NEWS NETWORK", 300)
            for index in range(24)
        ]
        assert find_subtitle_region(frames) is None

    def test_ignores_letterbox_edges(self):
        frames = [_background(index) for index in range(24)]
        for frame in frames:
            ImageDraw.Draw(frame).rectangle((0, 300, WIDTH, HEIGHT), fill=0)
        assert find_subtitle_region(frames) is None

    def test_needs_several_frames(self):
        assert find_subtitle_region(_subtitled_frames(y=300, count=2)) is None


class TestParseRegion:
    def test_reads_fractions(self):
        assert parse_region("0,0.8,1,0.1", 1920, 1080) == Region(0, 864, 1920, 108)

    def test_reads_pixels(self):
        assert parse_region("100,900,1720,120", 1920, 1080) == Region(
            100, 900, 1720, 120
        )

    def test_clamps_to_frame(self):
        region = parse_region("0,1000,5000,5000", 1920, 1080)
        assert region.x + region.width <= 1920
        assert region.bottom <= 1080

    def test_uses_even_coordinates(self):
        region = parse_region("101,903,1001,123", 1920, 1080)
        assert not any(
            value % 2 for value in (region.x, region.y, region.width, region.height)
        )

    @pytest.mark.parametrize(
        "spec", ["", "1,2,3", "1,2,3,4,5", "a,b,c,d", "0,0,0,10", "-1,0,10,10"]
    )
    def test_rejects_bad_values(self, spec):
        with pytest.raises(ValueError):
            parse_region_values(spec)


class TestBuildMaskFilter:
    REGION = Region(100, 900, 1720, 120)

    def test_box(self):
        assert build_mask_filter(ACTION_BOX, self.REGION) == (
            "drawbox=x=100:y=900:w=1720:h=120:color=black:t=fill"
        )

    def test_blur_crops_the_region_and_overlays_it_back(self):
        result = build_mask_filter(ACTION_BLUR, self.REGION)
        assert "crop=1720:120:100:900" in result
        assert result.endswith("overlay=100:900")

    def test_blur_normalizes_hardware_frames_first(self):
        result = build_mask_filter(ACTION_BLUR, self.REGION)
        assert result.startswith("format=yuv420p,")

    @pytest.mark.parametrize("action", [ACTION_AVOID, ACTION_NONE])
    def test_other_actions_leave_the_picture_alone(self, action):
        assert build_mask_filter(action, self.REGION) is None


class TestSubtitleMarginBottom:
    def test_mask_places_new_subtitles_inside_the_area(self):
        region = Region(0, 900, 1920, 120)
        assert subtitle_margin_bottom(ACTION_BLUR, region, 1080) == 48

    def test_avoid_places_new_subtitles_above_the_area(self):
        region = Region(0, 900, 1920, 120)
        assert subtitle_margin_bottom(ACTION_AVOID, region, 1080) == 132

    @pytest.mark.parametrize("action", [ACTION_AVOID, ACTION_BLUR, ACTION_BOX])
    def test_top_subtitles_leave_the_margin_unchanged(self, action):
        region = Region(0, 40, 1920, 100)
        assert subtitle_margin_bottom(action, region, 1080) is None


class TestValidateBurnedSubs:
    @staticmethod
    def _config(**synthesize) -> dict:
        return build_config(
            cli_overrides={"synthesize": synthesize}, config_path=None
        )

    def test_default_needs_nothing(self):
        assert validate_burned_subs(self._config(subtitle_mode="soft"))

    def test_masking_needs_hard_subtitles(self):
        config = self._config(subtitle_mode="soft", burned_subs_action="blur")
        assert not validate_burned_subs(config)

    def test_rejects_bad_region(self):
        config = self._config(
            subtitle_mode="hard",
            burned_subs_action="box",
            burned_subs_region="1,2,3",
        )
        assert not validate_burned_subs(config)

    def test_accepts_manual_region(self):
        config = self._config(
            subtitle_mode="hard",
            burned_subs_action="box",
            burned_subs_region="0,0.8,1,0.15",
        )
        assert validate_burned_subs(config)


class TestSoftSubtitleStreamArgs:
    def test_new_track_comes_first_and_is_default(self, monkeypatch):
        monkeypatch.setattr(
            "videocaptioner.core.utils.video_utils._list_subtitle_codecs",
            lambda _: ["mov_text"],
        )
        args = _soft_subtitle_stream_args("video.mp4")
        assert args.index("1:0") < args.index("0:s:0")
        assert args[args.index("-disposition:s:0") + 1] == "default"
        assert args[args.index("-disposition:s:1") + 1] == "0"

    def test_drops_image_tracks(self, monkeypatch):
        monkeypatch.setattr(
            "videocaptioner.core.utils.video_utils._list_subtitle_codecs",
            lambda _: ["hdmv_pgs_subtitle", "subrip"],
        )
        args = _soft_subtitle_stream_args("video.mkv")
        assert "0:s:0" not in args
        assert "0:s:1" in args


def _srt(path, text: str) -> str:
    path.write_text(f"1\n00:00:00,000 --> 00:00:03,000\n{text}\n", encoding="utf-8")
    return str(path)


def _make_video(path, extra_filter: str = "") -> str:
    filters = "hue=H=2*PI*t/3"
    if extra_filter:
        filters = f"{filters},{extra_filter}"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "gradients=s=640x360:r=10:d=3:speed=0.1",
            "-f", "lavfi", "-i", "sine=d=3",
            "-vf", filters,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return str(path)


def _subtitle_text(video: str, track: int) -> str:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video, "-map", f"0:s:{track}", "-f", "srt", "-"],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    return result.stdout


@needs_ffmpeg
class TestWithFFmpeg:
    def test_soft_subtitle_is_kept_alongside_existing_track(self, tmp_path):
        video = _make_video(tmp_path / "plain.mp4")
        with_track = str(tmp_path / "with_track.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", video,
                "-i", _srt(tmp_path / "old.srt", "old track"),
                "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
                with_track,
            ],
            check=True,
            capture_output=True,
        )
        output = str(tmp_path / "output.mp4")

        add_subtitles(
            with_track,
            _srt(tmp_path / "new.srt", "new track"),
            output,
            soft_subtitle=True,
        )

        assert len(_list_subtitle_codecs(output)) == 2
        assert "new track" in _subtitle_text(output, 0)
        assert "old track" in _subtitle_text(output, 1)

    def test_plain_video_has_nothing_to_hide(self, tmp_path):
        video = _make_video(tmp_path / "plain.mp4")
        assert detect_burned_subtitles(video) is None
        assert plan_burned_subs(video, ACTION_BLUR, "auto") is None

    def test_manual_region_builds_a_plan(self, tmp_path):
        video = _make_video(tmp_path / "plain.mp4")
        plan = plan_burned_subs(video, ACTION_BOX, "0,0.8,1,0.15")
        assert plan.region == Region(0, 288, 640, 54)
        assert "drawbox" in plan.pre_filter
        assert plan.margin_bottom == 44
