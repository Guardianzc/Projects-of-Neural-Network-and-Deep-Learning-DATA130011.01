
def area_cal(coord):
    return (coord[2] - coord[0]) * (coord[3]-coord[1])

def union_cal(pred, label):
    lens = min(pred[2], label[2]) - max(pred[0], label[0])
    wide = min(pred[3], label[3]) - max(pred[1], label[1])
    return (lens*wide)

def overlap(preds, labels):
    '''
    input：pred, label
    pred: [[x1,y1,x2,y2], [x1,y1,x2,y2], ....., ....]
    label: [[x1,y1,x2,y2], [x1,y1,x2,y2], ....., ....]
    '''

    label_length = len(labels)
    match = 0
    area_dict = {}

    for label in labels:
        area_dict[tuple(label)] = area_cal(label)
    for pred in preds:
        area_dict[tuple(pred)] = area_cal(pred)

    for label in labels:
        
        match_pred = None
        for pred in preds:
            if (pred[0] > label[2]) or (pred[1] > label[3]) or (label[0] > pred[2]) or (label[0] > pred[2]):
                continue
            
            union_area = union_cal(pred, label)

            if union_area / (area_dict[tuple(label)] + area_dict[tuple(pred)] - union_area) >= 0.5:
                match += 1
                preds.remove(pred)
                break
    return match

if __name__ == "__main__":
    a = [[1,1,5,3]]
    b = [[2,1,5,3]]
    print(area_cal(a[0]))
    print(union_cal(a[0], b[0]))
    print(overlap(a, b))