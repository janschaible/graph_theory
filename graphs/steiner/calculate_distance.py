from math import radians, sin, cos, sqrt, atan2

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km (Earth radius)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
