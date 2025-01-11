from medpy import metric
import numpy as np

def calculate_metric_percase(pred, gt):
    max_num = np.amax(gt)
    values_list = []
    for i in range(1, max_num+1):
        p, g = pred==i, gt==i
        p[p > 0] = 1
        g[g > 0] = 1
        dc = metric.binary.dc(p, g)
        jc = metric.binary.jc(p, g)
        hd = metric.binary.hd95(p, g)
        asd = metric.binary.asd(p, g)
        values_list.extend([dc, jc, hd, asd])
    return values_list