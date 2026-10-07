"""The probe preserves processor geometry and destroys fine spatial detail."""

import pytest
from PIL import Image

from va_opd.images import pixelate_image


def test_pixelation_is_bilinear_downsample_then_nearest_upsample():
    image = Image.new("RGB", (43, 27))
    image.putdata([(x * 5 % 256, y * 9 % 256, (x + y) * 3 % 256) for y in range(27) for x in range(43)])
    before = image.tobytes()
    actual = pixelate_image(image)
    expected = image.resize((4, 2), Image.Resampling.BILINEAR).resize((43, 27), Image.Resampling.NEAREST)
    assert actual.tobytes() == expected.tobytes()
    assert actual.size == image.size and actual.mode == image.mode
    assert image.tobytes() == before
    assert actual.tobytes() != before
    assert len(set(actual.tobytes()[i : i + 3] for i in range(0, len(actual.tobytes()), 3))) <= 8


@pytest.mark.parametrize("size", [(1, 1), (1, 9), (9, 1), (4, 3)])
def test_small_images_still_have_one_downsampled_pixel(size):
    image = Image.new("RGB", size, (12, 34, 56))
    result = pixelate_image(image)
    assert result.size == size and result.getpixel((0, 0)) == (12, 34, 56)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_modes_and_identity_ratio(mode):
    image = Image.new(mode, (11, 17))
    result = pixelate_image(image, ratio=1)
    assert result is not image
    assert result.mode == mode and result.tobytes() == image.tobytes()


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, True, "0.1", float("nan"), float("inf")])
def test_invalid_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        pixelate_image(Image.new("RGB", (10, 10)), ratio)
