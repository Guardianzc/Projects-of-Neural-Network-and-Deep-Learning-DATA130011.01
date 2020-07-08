import os
result_path_gt = '../data/result/gt'
files = list(sorted(os.listdir(result_path_gt)))
maxn = 0
for f in files:
    with open(result_path_gt+'/'+f,'r',encoding = 'utf8') as f:
        text = f.readlines()
    if text!=[]:
        maxn = max(maxn,len(text[0]))
print(maxn)