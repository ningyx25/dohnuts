"""Set-of-mark rendering: determinism, drawn marks, and caller-supplied labels.

Everything here is synthetic — screenshots are built with PIL in memory — so the
tests stay independent of the example-data corpus and of the row rules.
"""

from io import BytesIO

from PIL import Image, ImageDraw

from dohnuts.android_control_mark import (
    CHIP_COLOR,
    CHIP_PADDING,
    LABEL_COLOR,
    MARK_COLOR,
    label_font,
    mark_screenshot,
)
from dohnuts.mobile_jev_prompt import Element

# Small images keep the property tests quick. The chip tests need a screen tall
# enough for the label to reach its production size (2400 // 86 = 27 pixels):
# at the 12-pixel floor a 200-pixel-tall image gets, a "1" has no solid black
# pixel at all, so the glyph assertions there would be vacuous.
SMALL = (100, 200)
SCREEN = (100, 2400)

BACKGROUND = (0, 0, 255)
BOXES = [(10, 20, 60, 80), (10, 100, 60, 160)]

# A stand-in for the ICC profile every real screenshot carries.
ICC_PROFILE = bytes(range(256)) * 2

# How far past a box's top-left corner to look for its chip. The chip starts at
# the corner itself and is a few dozen pixels across at the largest label size.
CHIP_SEARCH = 60


def element(bounds, index=0, text="", content_description=""):
    """One element, exactly as `mobile_jev_prompt.summarize_observation` builds it."""
    return Element(
        id=str(index),
        bounds=tuple(bounds) if bounds is not None else None,
        text=text,
        label=content_description,
    )


def labelled(elements, labels=None):
    """The `(label, element)` pairs `mark_screenshot` draws."""
    if labels is None:
        labels = [str(position) for position in range(len(elements))]
    return list(zip(labels, elements))


def png_bytes(image):
    """`image` as PNG bytes, the form the row builder hashes for a file name."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def chip_box(marked, corner, span=CHIP_SEARCH):
    """The white chip at `corner` as `(left, top, right, bottom)`, or None.

    The screenshot's background is never white, so the chip is simply the white
    region in the window that starts at the corner; the box returned is the
    inclusive-exclusive pixel box PIL's `crop` wants. `span` narrows the window
    when a neighbouring chip could otherwise fall inside it.
    """
    x_min, y_min = corner
    width, height = marked.size
    pixels = marked.load()
    found = [
        (x, y)
        for y in range(y_min, min(height, y_min + span))
        for x in range(x_min, min(width, x_min + span))
        if pixels[x, y] == CHIP_COLOR
    ]
    if not found:
        return None
    return (
        min(x for x, _ in found),
        min(y for _, y in found),
        max(x for x, _ in found) + 1,
        max(y for _, y in found) + 1,
    )


def ink(image):
    """`image`'s dark pixels as an L bitmap cropped to them, for glyph compare.

    Cropping to the ink is what makes two renderings of the same glyph
    comparable: the chip and a reference label are drawn at different offsets,
    but the glyph itself rasterises identically.
    """
    mask = image.convert("L").point(lambda value: 255 if value < 128 else 0)
    bounds = mask.getbbox()
    assert bounds is not None, "expected the image to contain a glyph"
    return mask.crop(bounds).tobytes()


def glyph_mask(label, font):
    """The dark pixels of `label` drawn directly with `font`, cropped to them."""
    probe = Image.new("RGB", (200, 200), CHIP_COLOR)
    ImageDraw.Draw(probe).text((10, 10), label, fill=LABEL_COLOR, font=font)
    return ink(probe)


def black_pixels(image):
    """How many pixels of `image` are exactly the label colour."""
    colors = image.getcolors(maxcolors=1 << 16)
    assert colors is not None
    return sum(count for count, color in colors if color == LABEL_COLOR)


def marked_pair(size=SMALL):
    """The two-box screenshot and the result of marking it."""
    elements = [element(bounds, index=position) for position, bounds in enumerate(BOXES)]
    return mark_screenshot(Image.new("RGB", size, BACKGROUND), labelled(elements))


def test_marking_the_same_input_twice_gives_byte_identical_pngs():
    # Content-addressed file names hash these bytes, so the whole pipeline
    # depends on a second run of the same input landing on the same PNG.
    image = Image.new("RGB", SMALL, BACKGROUND)
    elements = labelled([element(bounds, index=position) for position, bounds in enumerate(BOXES)])
    assert png_bytes(mark_screenshot(image, elements)) == png_bytes(
        mark_screenshot(image, elements)
    )
    marked = mark_screenshot(image, elements)
    assert png_bytes(marked) == png_bytes(marked)
    # The marks really are in there: the marked image differs from the source.
    assert png_bytes(marked) != png_bytes(image)


def test_marks_depend_on_the_pixels_and_not_on_the_source_metadata():
    # Real screenshots are RGBA and carry an ICC profile. `convert` copies
    # `Image.info` onto the copy and `save` writes what it finds there, so an
    # identical-pixels screenshot would encode to different bytes — and land on
    # a different content-addressed name — purely because of its metadata.
    plain = Image.new("RGBA", SMALL, (10, 20, 30, 255))
    tagged = plain.copy()
    tagged.info["icc_profile"] = ICC_PROFILE
    tagged.info["dpi"] = (72, 72)
    elements = labelled([element(BOXES[0])])
    marked = mark_screenshot(tagged, elements)
    assert png_bytes(marked) == png_bytes(mark_screenshot(plain, elements))
    assert marked.info == {}
    # The sources keep their own metadata; only the copy is stripped.
    assert tagged.info == {"icc_profile": ICC_PROFILE, "dpi": (72, 72)}
    assert tagged.tobytes() == plain.tobytes()


def test_the_marks_use_the_literal_colours():
    marked = marked_pair(SCREEN)
    pixels = marked.load()
    _x_min, y_min, x_max, y_max = BOXES[0]
    assert pixels[x_max, (y_min + y_max) // 2] == (0, 255, 0)
    chip = chip_box(marked, BOXES[0][:2])
    assert chip is not None
    # The chip is white and the glyph inside it is black.
    assert pixels[chip[0], chip[1]] == (255, 255, 255)
    colors = marked.crop(chip).getcolors(maxcolors=1 << 16)
    assert colors is not None
    assert (255, 255, 255) in [color for _count, color in colors]
    assert (0, 0, 0) in [color for _count, color in colors]


def test_every_box_is_outlined_in_green_with_a_two_pixel_stroke():
    marked = marked_pair()
    pixels = marked.load()
    for x_min, y_min, x_max, y_max in BOXES:
        chip = chip_box(marked, (x_min, y_min))
        assert chip is not None, "expected a chip at the box's top-left corner"
        # The chip covers the corner, so the two edges it sits on are checked
        # past it; the far edges are never covered and pin the geometry.
        assert all(pixels[x, y_min] == MARK_COLOR for x in range(chip[2], x_max + 1))
        assert all(pixels[x_min, y] == MARK_COLOR for y in range(chip[3], y_max + 1))
        assert all(pixels[x, y_max] == MARK_COLOR for x in range(x_min, x_max + 1))
        assert all(pixels[x_max, y] == MARK_COLOR for y in range(y_min, y_max + 1))
        # The stroke is two pixels wide and lies inside the box, not outside.
        middle = (y_min + y_max) // 2
        assert pixels[x_max, middle] == MARK_COLOR
        assert pixels[x_max - 1, middle] == MARK_COLOR
        assert pixels[x_max - 2, middle] == BACKGROUND
        assert pixels[x_min, middle] == MARK_COLOR
        assert pixels[x_min + 1, middle] == MARK_COLOR
        assert pixels[x_min + 2, middle] == BACKGROUND


def test_box_corners_carry_a_white_chip_with_a_black_label():
    marked = marked_pair(SCREEN)
    font = label_font(marked)
    ascent, _descent = font.getmetrics()
    masks = []
    for position, (x_min, y_min, _x_max, _y_max) in enumerate(BOXES):
        chip = chip_box(marked, (x_min, y_min))
        assert chip is not None
        glyph = font.getbbox(str(position))
        # The chip hangs on the corner: its left edge on the box's, its top a
        # padding below, then the glyphs' ascent.
        assert chip[0] == x_min
        assert CHIP_PADDING <= chip[1] - y_min <= CHIP_PADDING + ascent
        # It is the glyph box plus the padding on all four sides.
        assert chip[2] - chip[0] == glyph[2] - glyph[0] + 2 * CHIP_PADDING
        assert chip[3] - chip[1] == glyph[3] - glyph[1] + 2 * CHIP_PADDING
        crop = marked.crop(chip)
        assert black_pixels(crop) >= 1, "the label glyph should have black pixels"
        masks.append(ink(crop))
    # Two elements, two numbers, two different pictures of them, and each chip
    # is the glyph of its own position.
    assert masks[0] != masks[1]
    assert masks[0] == glyph_mask("0", font)
    assert masks[1] == glyph_mask("1", font)


def test_chip_labels_are_exactly_what_the_caller_supplied():
    image = Image.new("RGB", SCREEN, BACKGROUND)
    pairs = labelled([element(BOXES[0], index=7), element(BOXES[1], index=3)], ["9", "3"])
    marked = mark_screenshot(image, pairs)
    font = label_font(marked)
    # The label is the criteria key the row offers, so neither the element's own
    # index nor its position in the list may leak into the drawing.
    assert ink(marked.crop(chip_box(marked, BOXES[0][:2]))) == glyph_mask("9", font)
    assert ink(marked.crop(chip_box(marked, BOXES[1][:2]))) == glyph_mask("3", font)
    assert ink(marked.crop(chip_box(marked, BOXES[0][:2]))) != glyph_mask("3", font)


def test_three_digit_labels_render_and_widen_their_chip():
    # Candidate lists run to MAX_CANDIDATES, so labels reach "127".
    image = Image.new("RGB", (128 * 30, 200), BACKGROUND)
    elements = [element((position * 30 + 2, 20, position * 30 + 26, 60)) for position in range(128)]
    marked = mark_screenshot(image, labelled(elements))
    font = label_font(marked)
    chip = chip_box(marked, (elements[-1].bounds[0], 20), span=28)
    assert chip is not None
    glyph = font.getbbox("127")
    single = font.getbbox("7")
    assert chip[2] - chip[0] == glyph[2] - glyph[0] + 2 * CHIP_PADDING
    assert chip[2] - chip[0] > single[2] - single[0] + 2 * CHIP_PADDING
    crop = marked.crop(chip)
    assert ink(crop) == glyph_mask("127", font)
    assert ink(crop) != glyph_mask("7", font)


def test_the_input_image_is_never_mutated():
    image = Image.new("RGB", SMALL, BACKGROUND)
    before = image.tobytes()
    marked = mark_screenshot(image, labelled([element(BOXES[0])]))
    assert image.tobytes() == before
    assert image.mode == "RGB"
    assert marked is not image
    assert marked.size == image.size


def test_non_rgb_input_is_returned_as_a_new_rgb_image():
    grey = Image.new("L", SMALL, 90)
    before = grey.tobytes()
    marked = mark_screenshot(grey, labelled([element(BOXES[0])]))
    assert marked.mode == "RGB"
    assert grey.mode == "L"
    assert grey.tobytes() == before

    rgba = Image.new("RGBA", SMALL, (10, 20, 30, 255))
    assert mark_screenshot(rgba, labelled([element(BOXES[0])])).mode == "RGB"
    assert rgba.mode == "RGBA"
    assert rgba.tobytes() == Image.new("RGBA", SMALL, (10, 20, 30, 255)).tobytes()


def test_no_elements_returns_the_original_pixels():
    image = Image.new("RGB", SMALL, BACKGROUND)
    marked = mark_screenshot(image, [])
    assert marked.mode == "RGB"
    assert marked.tobytes() == image.tobytes()

    grey = Image.new("L", SMALL, 42)
    assert mark_screenshot(grey, []).tobytes() == grey.convert("RGB").tobytes()


def test_elements_without_usable_bounds_are_skipped():
    image = Image.new("RGB", SMALL, BACKGROUND)
    elements = [
        {"index": 0, "text": "", "content_description": ""},  # no `bounds` key
        {"bounds": None},
        {"bounds": [1, 2, 3]},
        {"bounds": [0, 0, "x", 10]},
        {"bounds": [0, 0, 10, 10, 20]},
        {"bounds": [10, 10, 5, 5]},  # inverted: PIL would reject the rectangle
        {"bounds": [10, 10, 10, 10]},  # degenerate
        {"bounds": [float("inf"), 0, 10, 10]},  # `round` would raise
        {"bounds": [float("nan"), 0, 10, 10]},
        None,
        element(BOXES[1]),  # the only one that can be drawn
    ]
    marked = mark_screenshot(image, labelled(elements))
    pixels = marked.load()
    for x in range(0, 11):
        assert pixels[x, 0] == BACKGROUND
        assert pixels[x, 10] == BACKGROUND
    for y in range(0, 11):
        assert pixels[0, y] == BACKGROUND
        assert pixels[10, y] == BACKGROUND
    assert chip_box(marked, BOXES[0][:2]) is None
    # A skipped element draws nothing; the survivor keeps the label it was given.
    font = label_font(marked)
    chip = chip_box(marked, BOXES[1][:2])
    assert ink(marked.crop(chip)) == glyph_mask("10", font)


def test_boxes_running_off_the_edge_are_clipped():
    image = Image.new("RGB", SMALL, BACKGROUND)
    marked = mark_screenshot(
        image, labelled([element((-20, -20, 30, 30)), element((60, 150, 500, 400))])
    )
    pixels = marked.load()
    assert pixels[30, 30] == MARK_COLOR
    assert pixels[60, 199] == MARK_COLOR
