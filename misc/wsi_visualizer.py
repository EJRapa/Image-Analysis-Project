from tiatoolbox.wsicore.wsireader import WSIReader
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SLIDE_IDS = ["104S", "129S", "184B", "205B"]   # add more if you want multiple different WSIs
TRAINING_ROOT_DIR = Path("/bulk/rapae/BMED6460/wsi_project/training_data/wsitils")
WSI_FOLDER_NAME = Path("images")


def get_wsi_thumbnail(wsi_path, thumb_size=(320, 320)):
    reader = WSIReader.open(wsi_path)
    thumb = reader.slide_thumbnail()
    thumb = np.asarray(thumb)

    if thumb.shape[-1] == 4:
        img = Image.fromarray(thumb, mode="RGBA")
    else:
        img = Image.fromarray(thumb).convert("RGBA")

    img = img.resize(thumb_size, Image.Resampling.LANCZOS)
    return img


def warp_slide(img, slant=90, inset=35):
    """
    Make the slide look like your sketch:
    - right edge nearly vertical
    - left side slanted
    - top/bottom edges angled down-left
    """
    w, h = img.size

    # output quadrilateral corners in destination image
    # order: UL, LL, LR, UR
    quad = (
        slant, 0,          # upper-left pushed right
        0, h - inset,      # lower-left pushed left/down look
        w - slant, h,      # lower-right
        w, inset           # upper-right
    )

    warped = img.transform(
        (w, h),
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC
    )
    return warped


def make_stack(images, out_path="wsi_stack.png"):
    # warp all slides first
    warped = [warp_slide(im, slant=110, inset=45) for im in images]

    w, h = warped[0].size

    # transparent canvas
    canvas_w = w + 160
    canvas_h = h + 160
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # back-to-front offsets, like your sketch
    base_x = 70
    base_y = 40
    dx = 18   # shift right for deeper slides
    dy = -12  # shift upward for deeper slides

    # draw back first, front last
    n = len(warped)
    for i, im in enumerate(reversed(warped)):
        x = base_x + i * dx
        y = base_y + i * dy
        canvas.alpha_composite(im, (x, y))

    canvas.save(out_path)
    return canvas


def main():
    images = []
    for sid in SLIDE_IDS:
        wsi_path = TRAINING_ROOT_DIR / WSI_FOLDER_NAME / f"{sid}.tif"
        images.append(get_wsi_thumbnail(wsi_path, thumb_size=(320, 320)))

    # repeat same image if you just want to test the stack look
    if len(images) == 1:
        images = [images[0], images[0], images[0]]

    result = make_stack(images, out_path="wsi_stack_transparent.png")


if __name__ == "__main__":
    main()