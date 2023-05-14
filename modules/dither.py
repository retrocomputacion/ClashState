from . import palette as Palette
import numpy as np
from .palette import CCIR_LUMINOSITY
from hitherdither.ordered.bayer import I
from hitherdither.ordered.yliluoma._utils import color_compare

######################################
# Custom ordered dither
# code derived from hitherdither
# custom Bayer matrixes taken from
# Project One

def B(m):
    """Get the Bayer matrix with side of length ``n``.
    Will only work if ``n`` is a power of 2.
    Reference: http://caca.zoy.org/study/part2.html
    :param int n: Power of 2 side length of matrix.
    :return: The Bayer matrix.
    """
    return (1 + m) / (1 + (m.shape[0] * m.shape[1]))

def custom_dithering(image, palette:Palette, thresholds, type=0):
    """Render the image using the ordered Bayer matrix dithering pattern.
    :param :class:`PIL.Image` image: The image to apply
        Bayer ordered dithering to.
    :param :class:`~hitherdither.colour.Palette` palette: The palette to use.
    :param thresholds: Thresholds to apply dithering at.
    :param int order: Custom matrix type.
    :return:  The Bayer matrix dithered PIL image of type "P"
        using the input palette.
    """
    dMatrix = np.asarray([
        [[4,1],[2,3]],
        [[1,13,4,16],[9,5,12,7],[3,15,2,14],[11,8,10,8]],
        [[1,2,3,4],[9,10,11,12],[5,6,7,8],[13,14,15,16]],
        [[1,9,4,12],[5,13,6,14],[3,11,2,10],[7,15,8,16]],
        [[10,1,12,6],[4,9,3,15],[14,2,13,7],[8,11,5,16]],
        [[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],
        [12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
        [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],
        [15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]]], dtype=object)


    bayer_matrix = B(np.asarray(dMatrix[type]))
    ni = np.array(image, "uint8")
    thresholds = np.array(thresholds, "uint8")
    xx, yy = np.meshgrid(range(ni.shape[1]), range(ni.shape[0]))
    xx %= bayer_matrix.shape[0]
    yy %= bayer_matrix.shape[1]
    factor_threshold_matrix = np.expand_dims(bayer_matrix[yy, xx], axis=2) * thresholds
    new_image = ni + factor_threshold_matrix
    return palette.create_PIL_png_from_rgb_array(new_image)

##############################################
# Yliluoma's algo 1 copied from hitherdither
##############################################

def _get_mixing_plan_matrix(palette, order=8):
    mixing_matrix = []
    colours = {}
    colour_component_distances = []

    nn = order * order
    for i in range(len(palette)):
        for j in range(i, len(palette)):
            for ratio in range(0, nn):
                if i == j and ratio != 0:
                    break
                # Determine the two component colors.
                c_mix = _colour_combine(palette, i, j, ratio / nn)
                hex_colour = palette.rgb2hex(*c_mix.tolist())
                colours[hex_colour] = (i, j, ratio / nn)
                mixing_matrix.append(c_mix)

                c1 = np.array(palette[i], "int")
                c2 = np.array(palette[j], "int")
                cmpval = (
                    color_compare(c1, c2)
                    * 0.1
                    * (np.abs((ratio / float(nn)) - 0.5) + 0.5)
                )
                colour_component_distances.append(cmpval)

    mixing_matrix = np.array(mixing_matrix)
    colour_component_distances = np.array(colour_component_distances)

    # for c in mixing_matrix:
    #     assert palette.rgb2hex(*c.tolist()) in colours

    return mixing_matrix, colours, colour_component_distances


def _colour_combine(palette, i, j, ratio):
    c1, c2 = np.array(palette[i], "int"), np.array(palette[j], "int")
    return np.array(c1 + ratio * (c2 - c1), "uint8")


def _improved_mixing_error_fcn(
    colour, mixing_matrix, colour_component_distances, luma_mat=None
):
    """Compares two colours using the Psychovisual model.

    The simplest way to adjust the psychovisual model is to
    add some code that considers the difference between the
    two pixel values that are being mixed in the dithering
    process, and penalizes combinations that differ too much.

    Wikipedia has an entire article about the topic of comparing
    two color values. Most of the improved color comparison
    functions are based on the CIE colorspace, but simple
    improvements can be done in the RGB space too. Such a simple
    improvement is shown below. We might call this RGBL, for
    luminance-weighted RGB.

    :param :class:`numpy.ndarray` colour: The colour to estimate error to.
    :param :class:`numpy.ndarray` mixing_matrix: The rgb
        values of mixed colours.
    :param :class:`numpy.ndarray` colour_component_distances: The colour
        distance of the mixed colours.
    :return: :class:`numpy.ndarray`

    """
    colour = np.array(colour, "int")
    if luma_mat is None:
        luma_mat = mixing_matrix.dot(CCIR_LUMINOSITY / 255000)
    luma_colour = colour.dot(CCIR_LUMINOSITY) / (255000)
    luma_diff_squared = (luma_mat - luma_colour) ** 2
    diff_colour_squared = ((colour - mixing_matrix) / 255) ** 2
    cmpvals = ((diff_colour_squared.dot(CCIR_LUMINOSITY) / 1000)*0.75)+luma_diff_squared+colour_component_distances
    #cmpvals *= 0.75
    #cmpvals += luma_diff_squared
    #cmpvals += colour_component_distances
    return cmpvals


def yliluomas_1_ordered_dithering(image, palette, progress, order=8):
    """A dithering method that weighs in color combinations of palette.

    N.B. tri-tone dithering is not implemented.

    :param :class:`PIL.Image` image: The image to apply
        Bayer ordered dithering to.
    :param :class:`~hitherdither.colour.Palette` palette: The palette to use.
    :param list(int) progress: Progress value, passed 'by reference'
    :param int order: The Bayer matrix size to use.
    :return:  The dithered PIL image of type "P" using the input palette.

    """
    bayer_matrix = I(order, transposed=True) / 64.0
    ni = np.array(image, "uint8")
    xx, yy = np.meshgrid(range(ni.shape[1]), range(ni.shape[0]))
    factor_matrix = bayer_matrix[yy % order, xx % order]

    # Prepare all precalculated mixed colours and their respective
    mixing_matrix, colour_map, colour_component_distances = _get_mixing_plan_matrix(
        palette
    )
    mixing_matrix = np.array(mixing_matrix, "int")
    luma_mat = mixing_matrix.dot(CCIR_LUMINOSITY / 1000.0 / 255.0)

    steps = (0.5-progress[0])/(ni.shape[1]*ni.shape[0])

    color_matrix = np.zeros(ni.shape[:2], dtype="uint8")
    for x, y in zip(np.nditer(xx), np.nditer(yy)):
        min_index = np.argmin(
            _improved_mixing_error_fcn(
                ni[y, x, :], mixing_matrix, colour_component_distances, luma_mat
            )
        )
        closest_mix_colour = mixing_matrix[min_index, :].tolist()
        closest_mix_hexcolour = palette.rgb2hex(*closest_mix_colour)
        plan = colour_map.get(closest_mix_hexcolour)
        color_matrix[y, x] = plan[1] if (factor_matrix[y, x] < plan[-1]) else plan[0]
        progress[0]+=steps

    return palette.create_PIL_png_from_closest_colour(color_matrix)

