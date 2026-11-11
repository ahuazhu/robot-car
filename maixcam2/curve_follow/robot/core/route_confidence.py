"""One continuous confidence model and its normalized speed envelope."""
import math
import numpy as np


def innovation_likelihood(distance_m, heading_cos, scale_m):
    position=1. / (1. + (max(0.,float(distance_m))/scale_m)**2)
    direction=max(0.,min(1.,(float(heading_cos)+1.)/2.))**2
    return position*direction


def confidence_profile(quality, sigma, agreement, surface, position_scale):
    localization=1. / (1. + (np.asarray(sigma)/position_scale)**2)
    return np.clip(np.asarray(quality)*localization*agreement*surface,0.,1.)


def confidence_speed_factor(confidence):
    confidence=float(confidence)
    if not math.isfinite(confidence) or not 0<=confidence<=1:
        raise ValueError('Centerline confidence must be within 0..1')
    # Preserve a crawl reference for low confidence. Physical stops remain
    # independent constraints; confidence alone never requests zero speed.
    return max(.1,min(1.,confidence/.8))
