import os
import sys
sys.path.append(os.getcwd())
from itertools import chain

# processedFile = open('./wider_face_processed_val_gt.txt','w')
# annoFile = "./wider_face_val_bbx_gt.txt"


processed_ = './wider_face_processed_train_gt.txt'
annoFile = "./wider_face_train_bbx_gt.txt"
final_file = './wider_train_gt.txt'
def frist_step():
    processedFile = open(processed_,'w')
    with open(annoFile,'r') as f:
        lines = f.readlines()
    res=""

    for line in lines:
        line = line.strip()
        if "/" in line:
            imgPath = line
            # processedFile.write(imgPath+' ')
        if len(line.split(' '))==1 and " .jpg " not in line:
            nums = str(line.split(' ')[0])
            # processedFile.write(' ')
        else:
            # res = imgPath+' '+line 
            # print(imgPath+'00000'+line)
            # print(line.type)
            processedFile.write(imgPath+' '+line+'\n')
            # print(res)
        
        # processedFile.write('\n')

def second_step():
    with open(processed_,'r') as f:
        lines = f.readlines()

    tmp = ''
    anno_dic = {} 
    for line in lines:
        arrayLine = line.strip().split(' ')

        if arrayLine[0] not in anno_dic:
            anno_dic[arrayLine[0]] = []
        tempBbox=[arrayLine[1],arrayLine[2],str(int(arrayLine[1])+int(arrayLine[3])),str(int(arrayLine[2])+int(arrayLine[4]))]
        anno_dic[arrayLine[0]].append(tempBbox)
    
    f = open(final_file,'w')
    for key, value in anno_dic.items():
        f.write(str(key)+' '+str(" ".join(list(chain.from_iterable(value))))+"\n")
        
if __name__=="__main__":
    # frist_step()
    second_step()