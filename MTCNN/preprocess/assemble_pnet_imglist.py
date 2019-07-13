import os
import sys
sys.path.append(os.getcwd())
import assemble
postive_file = '../datasets/train/12/pos_12.txt'
negative_file = '../datasets/train/12/neg_12.txt'
part_file = '../datasets/train/12/part_12.txt'

imglist_filename = '../datasets/train/12/train_12.txt'

if __name__=="__main__":
    anno_list=[]
    
    anno_list.extend([postive_file,negative_file,part_file])
    chose_count = assemble.assemble_(imglist_filename,anno_list)
    print("total sample is %d was assemble"%chose_count)
    