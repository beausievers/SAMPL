from scipy.interpolate import interp1d


def get_update(dip_center, dip_width, y_min, y_max):
    """
    Function for different reweighting of weak vs. strong graph edges

    Update function with symmetrical dip.

    Arguments
    ---------
    dip_center : float [0, 1]
    dip_width : float [0, 1]
    y_min : float
    y_max : float

    Returns
    -------
    function [0, 1] -> [y_min, y_max]
    """
    dip_half_width = dip_width / 2
    xp = [0, dip_center - dip_half_width, dip_center, dip_center + dip_half_width, 1]
    fp = [0, 0, y_min, 0, y_max]
    return interp1d(xp, fp)
