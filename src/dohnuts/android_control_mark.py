"""Set-of-mark rendering of a screenshot's candidate UI elements.

An element row asks the model to pick the next UI element and names its
candidates `r0..r{N-1}` in `question.criteria`, so the screenshot the row ships
has to carry those same numbers on screen. This module owns that one drawing
step: a green box per candidate plus a white chip holding its number. It is a
PIL port of the look of the dataset's own `add_ui_element_mark`, which needs cv2
and a protobuf accessibility forest; here the input is the plain element dicts
`dohnuts.android_control_data.extract_elements` returns and a `PIL.Image`, so no
image library leaks into the row rules.

Marking is a pure function of `(image, elements)`: no randomness, no clock, no
environment, so equal inputs always give equal pixels and therefore
byte-identical PNGs, which is what content-addressed file names rely on.
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
        # `load_default` grew its `size` argument in Pillow 10.1. Without it the
        # labels would silently come out at a different scale, so fail here
        # rather than fall back to a look nothing else was tuned against.
        raise RuntimeError(
            "ImageFont.load_default(size=...) is unavailable; set-of-mark rendering "
            "needs Pillow >= 10.1 (dohnuts requires Pillow >= 11)"
        ) from error


def mark_screenshot(image: Image.Image, elements: list[dict]) -> Image.Image:
    """Return a NEW RGB copy of `image` with every candidate element boxed and numbered.

    The copy is what lets callers keep using the screenshot they passed in; a
    non-RGB input is converted first. Elements are drawn in list order and
    numbered by their position in it, which is the numbering the `r{position}`
    criteria keys use: an element's own `index` field is ignored, so a list that
    was filtered or reordered still gets the numbers the row names. Elements
    whose `bounds` are missing or malformed are skipped and keep their slot, so
    that alignment holds even for a partial list. Nothing is drawn outside the
    image: PIL clips whatever runs past an edge, which is exactly the behaviour
    wanted for a box that touches one.
    """
    marked = image.convert("RGB")
    draw = ImageDraw.Draw(marked)
    font = label_font(marked)
    for position, element in enumerate(elements):
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
        label = str(position)
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
