"""Set-of-mark rendering of a screenshot's candidate UI elements.

A `tap_target` row asks the model to pick an element and names its candidates
`1..N` in `question.criteria` -- the numbers the agent's own element indices use
-- so the screenshot the row ships has to carry those same numbers on screen.
This module owns that one drawing step: a green box per candidate plus a white
chip holding its number. It is a PIL port of the look of the dataset's own
`add_ui_element_mark`, which needs cv2 and a protobuf accessibility forest; here
the input is a list of `(label, element)` pairs and a `PIL.Image`, so no image
library leaks into the row rules.

Marking is a pure function of `(image, elements)`: no randomness, no clock, no
environment reads, and the same pixels always encode to the same PNG bytes,
which is what content-addressed file names rely on. That determinism is only as
strong as the renderer under it, though: the label face is the font bundled with
Pillow, rasterised by FreeType and encoded by Pillow's PNG writer, so bumping
Pillow changes the bytes of an otherwise identical mark and every file name
derived from them. The module is pure, but not environment-independent.
"""

from PIL import Image, ImageDraw, ImageFont

from dohnuts.android_control_data import element_bounds

# The dataset scales its stroke with the screenshot because it draws into a
# frame that may itself be scaled. Our bounds already are screenshot pixels, so
# that scale is 1 and the dataset's `int(2 * sqrt(2))` collapses to this.
MARK_WIDTH = 2

# Breathing room between the label glyphs and the edge of their chip, which is
# also the offset of the chip from the box's top-left corner.
CHIP_PADDING = 2

MARK_COLOR = (0, 255, 0)
CHIP_COLOR = (255, 255, 255)
LABEL_COLOR = (0, 0, 0)

# The dataset draws a roughly 25-pixel label on a 2400-pixel-tall screen.
# Dividing the height by this keeps that visual scale; the floor keeps short
# screenshots legible.
FONT_HEIGHT_DIVISOR = 86
MIN_FONT_SIZE = 12


def label_font(image: Image.Image) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The label font for `image`: the default face at `image`'s scale."""
    size = max(MIN_FONT_SIZE, image.height // FONT_HEIGHT_DIVISOR)
    try:
        return ImageFont.load_default(size=size)
    except TypeError as error:
        # Belt and braces: `size=` has existed since Pillow 10.1 and dohnuts
        # requires 11, so this cannot fire on a supported install. It is here so
        # that an undeclared older Pillow fails loudly instead of drawing every
        # label at a scale nothing else was tuned against.
        raise RuntimeError(
            "ImageFont.load_default(size=...) is unavailable; set-of-mark rendering "
            "needs Pillow >= 10.1 (dohnuts requires Pillow >= 11)"
        ) from error


def mark_screenshot(image: Image.Image, candidates: list[tuple[str, object]]) -> Image.Image:
    """Return a NEW RGB copy of `image` with every candidate boxed and numbered.

    `candidates` is a list of `(label, element)` pairs, drawn in order; the label
    is the chip text, so the caller decides the numbering and it is the very key
    its criteria use. The copy is what lets callers keep using the screenshot
    they passed in; a non-RGB input is converted first. Only the pixels of the
    input matter: anything the source carried in `Image.info` — an ICC profile, a
    DPI, a timestamp — is dropped, because the PNG encoder would otherwise write
    it out and two screenshots with equal pixels would hash differently. That
    changes the bytes of every mark produced before it, so it belongs before the
    first conversion writes a file; names derived from marked bytes must never
    mix the two behaviours. Elements whose `bounds` are missing or malformed are
    skipped, so a partial list still marks exactly the elements it names. Nothing
    is drawn outside the image: PIL clips whatever runs past an edge, which is
    exactly the behaviour wanted for a box that touches one.
    """
    marked = image.convert("RGB")
    # `convert` copies `image.info` onto the copy and `save` writes what it finds
    # there, so the copy starts empty.
    marked.info.clear()
    draw = ImageDraw.Draw(marked)
    font = label_font(marked)
    for label, element in candidates:
        bounds = element_bounds(element)
        if bounds is None:
            continue
        x_min, y_min, x_max, y_max = (round(value) for value in bounds)
        draw.rectangle((x_min, y_min, x_max, y_max), outline=MARK_COLOR, width=MARK_WIDTH)
        # PIL anchors text boxes at the ascender line, so the chip is the
        # glyphs' own box grown by the padding rather than a fixed-size patch.
        # `textbbox` reports its far corner one pixel past the last glyph pixel
        # while `rectangle` takes the corners themselves, hence the step back.
        # The chip is filled before the label so it never covers the glyphs.
        text_at = (x_min + CHIP_PADDING, y_min + CHIP_PADDING)
        left, top, right, bottom = draw.textbbox(text_at, label, font=font)
        chip = (
            left - CHIP_PADDING,
            top - CHIP_PADDING,
            right + CHIP_PADDING - 1,
            bottom + CHIP_PADDING - 1,
        )
        draw.rectangle(chip, fill=CHIP_COLOR)
        draw.text(text_at, label, fill=LABEL_COLOR, font=font)
    return marked
