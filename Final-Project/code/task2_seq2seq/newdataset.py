import os
from PIL import Image
import matplotlib.pyplot as plt
import tqdm 
def get_imagepath(root):
    image_paths = list(map(lambda x: os.path.join(root, x), [t for t in os.listdir(root) if (('jpg' in t) and ('nii' not in t))]))
    return image_paths

def get_landmarkpath(root):
    landmark_paths = list(map(lambda x: os.path.join(root, x), [t for t in os.listdir(root) if (('txt' in t) and ('nii' not in t))]))
    return landmark_paths



if __name__ == "__main__":
    landmark_path = '../data/train/gt/'
    image_path = '../data/train/img/'
    result_path_image = '../data/result/image'
    result_path_gt = '../data/result/gt'
    image_files = list(sorted(os.listdir(image_path)))
    label_files = list(sorted(os.listdir(landmark_path)))
    count = 0
    count_files = 0
    length = len(image_files)
    
    for i in tqdm.tqdm(range(length)):
        img_path = os.path.join(image_path, image_files[i])
        label_path = os.path.join(landmark_path, label_files[i])
        image = Image.open(img_path)
        landmarks = open(label_path,'r',encoding='utf8').readlines()
        for landmark in landmarks:
            coord = landmark.split(',')
            for j in range(8):
                coord[j] = int(coord[j])
            x1 = min(coord[0], coord[2], coord[4], coord[6])
            x2 = max(coord[0], coord[2], coord[4], coord[6])
            y1 = min(coord[1], coord[3], coord[5], coord[7])
            y2 = max(coord[1], coord[3], coord[5], coord[7])
            #x1, y1, x2, y2 = landmark.split(',')
            crop = image.crop((x1,y1,x2,y2))
            label = ','.join(coord[9:])
            try:
                label = label.rstrip('\n')
            except:
                pass
            image_name = result_path_image + '/' + str(count) + image_files[i]
            label_name = result_path_gt + '/' + str(count) + label_files[i]
            count += 1
            if label == '':
                a = 1
                pass
            '''
            try:
                crop.save(image_name)
                with open(label_name, 'w',encoding='utf8') as f:
                    f.write(label)
            except:
                print(count)
                with open('error.txt', 'a') as f:
                    f.write(image_name)
                    f.write(label)
            '''
        count_files += 1

