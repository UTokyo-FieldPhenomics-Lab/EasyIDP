import numpy as np
import colorsys

def distinguish_color(num):
    dist_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + np.random.rand() * 10
        l = 50 + np.random.rand() * 10
        r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
        dist_colors.append((r, g, b))
        i += step

    return dist_colors