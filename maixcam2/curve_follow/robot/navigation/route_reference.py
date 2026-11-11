"""Local pose on a retained route, independent of forward-pursuit feasibility."""
import math
import numpy as np
from robot.core.geometry import arc, differential, support_projection


def route_reference(path, origin, maximum_distance):
    points=np.asarray(path,float)
    if (points.ndim!=2 or points.shape[1:]!=(2,) or len(points)<5
            or not np.isfinite(points).all()):
        return None
    points=points-np.asarray(origin,float)
    distance,position=support_projection(np.zeros((1,2)),points)
    if not 0<=distance[0]<=maximum_distance:
        return None
    stations=arc(points)
    index=int(np.clip(np.searchsorted(stations,position[0],side='right')-1,0,len(points)-2))
    span=stations[index+1]-stations[index]
    if span<=0:return None
    fraction=float(np.clip((position[0]-stations[index])/span,0.,1.))
    point=points[index]+fraction*(points[index+1]-points[index])
    tangent,_=differential(points)
    direction=(1-fraction)*tangent[index]+fraction*tangent[index+1]
    return dict(heading_rad=math.atan2(direction[1],direction[0]),
        bearing_rad=math.atan2(point[1],point[0]),distance_m=float(distance[0]),point_m=point.tolist())
