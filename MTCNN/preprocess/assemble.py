import os
import numpy as np 

def assemble_(out_file,input_file_list):

    if len(input_file_list)==0:
        return 
    
    if os.path.exists(out_file):
        os.remove(out_file)

    for file in input_file_list:
        with open(file ,'r') as f:
            lines = f.readlines()

        base_num = 250000

        if len(lines) > base_num*3:
            idx_keep = np.random.choice(len(lines),size=base_num*3,replace=True)
        elif len(lines) > 10000:
            idx_keep = np.random.choice(len(lines), size=len(lines), replace=True)
        else:
            idx_keep = np.arange(len(lines))
            np.random.shuffle(idx_keep)
        chose_count = 0
        with open(out_file,'a+') as f:
            for idx in idx_keep:
                f.write(lines[idx])
                chose_count+=1
                
        print("%s was processed"%file)
        
    return chose_count
