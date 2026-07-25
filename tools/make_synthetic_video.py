"""Create a deterministic video where eight people enter one by one."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = ROOT / "assets/vision_demo/empty_entrance.png"
SPRITES = ROOT / "assets/vision_demo/people_sprite_rgba.png"
OUTPUT = ROOT / "assets/vision_demo/sequential_entry_8_people_4_masks.mp4"

FPS = 30
DURATION_SECONDS = 18
OUTPUT_SIZE = (1280, 720)
ENTRY_INTERVAL = 1.55
WALK_DURATION = 3.4


FINAL_POSITIONS = (
    (160, 345),
    (315, 330),
    (465, 350),
    (610, 320),
    (760, 340),
    (900, 325),
    (1035, 350),
    (1165, 335),
)


def split_sprites(sheet: Image.Image) -> list[Image.Image]:
    """Split the 4x2 sprite sheet and trim transparent padding."""
    people: list[Image.Image] = []
    cell_width, cell_height = sheet.width // 4, sheet.height // 2
    for row in range(2):
        for column in range(4):
            cell = sheet.crop(
                (
                    column * cell_width,
                    row * cell_height,
                    (column + 1) * cell_width,
                    (row + 1) * cell_height,
                )
            )
            bbox = cell.getchannel("A").getbbox()
            if bbox is None:
                raise ValueError(f"Empty sprite cell: row={row}, column={column}")
            people.append(cell.crop(bbox))
    return people


def smootherstep(value: float) -> float:
    """Quintic easing with zero velocity and acceleration at both ends."""
    value = min(1.0, max(0.0, value))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def make_frame(
    background: Image.Image,
    people: list[Image.Image],
    frame_number: int,
) -> np.ndarray:
    """Animate each person from the rear gate to a distinct foreground spot."""
    frame = background.copy()
    timestamp = frame_number / FPS

    for index, person in enumerate(people):
        start_time = 0.65 + index * ENTRY_INTERVAL
        raw_progress = min(1.0, max(0.0, (timestamp - start_time) / WALK_DURATION))
        progress = smootherstep(raw_progress)
        if timestamp < start_time:
            continue

        target_x, target_y = FINAL_POSITIONS[index]
        start_x = 640 + (index % 3 - 1) * 35
        start_y = 165
        x = int(start_x + (target_x - start_x) * progress)
        y = int(start_y + (target_y - start_y) * progress)

        # Perspective: a person grows naturally while approaching the camera.
        target_height = 335 + (index % 2) * 18
        height = int(195 + (target_height - 195) * progress)
        # Five soft steps: vertical body bounce and opposing lateral weight shift.
        step_phase = raw_progress * np.pi * 10.0
        motion_fade = np.sin(np.pi * raw_progress) if raw_progress < 1.0 else 0.0
        x += int(np.sin(step_phase + index * 0.7) * 5 * motion_fade)
        y -= int(abs(np.sin(step_phase)) * 4 * motion_fade)
        width = max(1, int(person.width * height / person.height))
        resized = person.resize((width, height), Image.Resampling.LANCZOS)
        angle = float(np.sin(step_phase + index * 0.7) * 1.15 * motion_fade)
        resized = resized.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        frame.alpha_composite(
            resized,
            (x - resized.width // 2, y - resized.height // 4),
        )

    return np.asarray(frame.convert("RGB"))


def main() -> None:
    background = Image.open(BACKGROUND).convert("RGBA")
    background = background.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    sprites = split_sprites(Image.open(SPRITES).convert("RGBA"))
    with imageio.get_writer(
        OUTPUT,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="warning",
    ) as writer:
        for frame_number in range(FPS * DURATION_SECONDS):
            writer.append_data(make_frame(background, sprites, frame_number))
    print(OUTPUT)


if __name__ == "__main__":
    main()
