def get_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    return inter / (area1 + area2 - inter + 1e-6)


def get_ios(box_small, box_large):
    x1, y1, x2, y2 = box_small
    x3, y3, x4, y4 = box_large
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_s = (x2 - x1) * (y2 - y1)
    return inter / (area_s + 1e-6)


def get_union_box(box1, box2):
    return (
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    )

def capped_union(old_box, new_box, max_growth=1.3):
    ux1, uy1, ux2, uy2 = get_union_box(old_box, new_box)

    ox1, oy1, ox2, oy2 = old_box
    ow, oh = ox2 - ox1, oy2 - oy1

    uw, uh = ux2 - ux1, uy2 - uy1

    if uw > ow * max_growth or uh > oh * max_growth:
        return old_box

    return (ux1, uy1, ux2, uy2)
