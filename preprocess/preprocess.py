#======================================================================
#
#					Copyright (C) 2020, Buzhen Huang
#						   All rights reserved
#-----------------------------------------------------------------------
#	filename : preprocess.py
#	description : Transform from original SMPL model(http://smpl.is.tue.mpg)
#   to txt format. 
#   We provide this software for research purposes only. 
#
#                              created by Buzhen Huang at  03/17/2020
#======================================================================
import numpy as np
import os
import pickle

class SMPLModel():
    def __init__(self, model_path='./model/SMPL_NEUTRAL.pkl'):
        super(SMPLModel, self).__init__()

        self.output2d = []
        self.output3d = []

        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')

        self.output2d.append(['J_regressor',np.array(params['J_regressor'].todense())])
        self.output2d.append(['weights',params['weights']])
        self.output2d.append(['v_template',params['v_template']])
        self.output2d.append(['kintree_table',params['kintree_table']])
        self.output2d.append(['faces',params['f']])

        self.output3d.append(['shapedirs',params['shapedirs']])
        self.output3d.append(['posedirs',params['posedirs']])

    def write_params(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        print('---transform parameters---')
        for item in self.output2d:
            f = open(os.path.join(path, item[0]+'.txt'), 'w')
            for line in item[1]:
                for p in line:
                    if item[0] == 'faces':
                        f.write(str(p+1)+' ')
                    else:
                        f.write(str(p)+' ')
            f.close()

        for item in self.output3d:
            f = open(os.path.join(path, item[0]+'.txt'), 'w')
            for line in item[1]:
                for i in line:
                    for j in i:
                        f.write(str(j)+' ')
            f.close()

if __name__ == "__main__":

    model_path = './model/SMPL_MALE.pkl' # pkl model
    output = '../SMPL/SMPL_MALE'

    model = SMPLModel(model_path=model_path)
    model.write_params(output)
