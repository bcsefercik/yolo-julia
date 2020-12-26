PARAM_IOU_T = 0.20  # iou training threshold

PARAM_GIOU = 3.54  # giou loss gain
PARAM_CLS = 37.4  # cls loss gain
PARAM_OBJ = 64.3 * (416/320)  # obj loss gain (*=img_size/320 if img_size != 320)


# hyp = {'giou': 3.54,  # giou loss gain
#        'cls': 37.4,  # cls loss gain
#        'cls_pw': 1.0,  # cls BCELoss positive_weight
#        'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
#        'obj_pw': 1.0,  # obj BCELoss positive_weight
#        'iou_t': 0.20,  # iou training threshold
#        'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
#        'lrf': 0.0005,  # final learning rate (with cos scheduler)
#        'momentum': 0.937,  # SGD momentum
#        'weight_decay': 0.0005,  # optimizer weight decay
#        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
#        'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
#        'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
#        'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
#        'degrees': 1.98 * 0,  # image rotation (+/- deg)
#        'translate': 0.05 * 0,  # image translation (+/- fraction)
#        'scale': 0.05 * 0,  # image scale (+/- gain)
#        'shear': 0.641 * 0}  # image shear (+/- deg)