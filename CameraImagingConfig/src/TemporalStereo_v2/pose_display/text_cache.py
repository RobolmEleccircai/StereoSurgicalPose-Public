"""Bounded glyph-tile cache for video overlays; full report rendering stays exact."""
from functools import lru_cache
from PIL import Image, ImageDraw


@lru_cache(maxsize=256)
def tile(text, font, fill, anchor, stroke_width, stroke_fill):
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    box = probe.textbbox((0, 0), text, font=font, anchor=anchor, stroke_width=stroke_width)
    left, top, right, bottom = box
    image = Image.new('RGBA', (max(1, right-left), max(1, bottom-top)))
    ImageDraw.Draw(image).text((-left, -top), text, font=font, fill=fill, anchor=anchor,
                              stroke_width=stroke_width, stroke_fill=stroke_fill)
    return image, box


@lru_cache(maxsize=256)
def text_box(text, font, anchor, stroke_width):
    return ImageDraw.Draw(Image.new('L', (1, 1))).textbbox(
        (0, 0), text, font=font, anchor=anchor, stroke_width=stroke_width)


class CachedTextDraw:
    """ImageDraw-compatible subset used by the approved compact renderer."""
    def __init__(self, canvas):
        self.canvas = canvas
        self.draw = ImageDraw.Draw(canvas)

    def __getattr__(self, name):
        return getattr(self.draw, name)

    def textbbox(self, xy, text, font, anchor=None, stroke_width=0):
        box = text_box(text, font, anchor, stroke_width)
        return tuple(value+xy[i % 2] for i, value in enumerate(box))

    def text(self, xy, text, font, fill, anchor=None, stroke_width=0, stroke_fill=None):
        image, box = tile(text, font, fill, anchor, stroke_width, stroke_fill)
        self.canvas.paste(image, (int(xy[0]+box[0]), int(xy[1]+box[1])), image)
