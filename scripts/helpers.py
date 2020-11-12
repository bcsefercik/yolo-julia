def coco_to_yolo_bbox(coco_bbox, width, height):

    bbox = [0, 0, 0, 0]
    bbox[0] = coco_bbox[0]/width
    bbox[1] = coco_bbox[1]/height
    bbox[2] = coco_bbox[2]/width
    bbox[3] = coco_bbox[3]/height
    bbox[0] += bbox[2] / 2
    bbox[1] += bbox[3] / 2
    bbox = list(map(lambda x: round(x, 6), bbox))

    return bbox
