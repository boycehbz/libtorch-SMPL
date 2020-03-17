/*======================================================================
*
*					Copyright (C) 2020, Buzhen Huang
*						   All rights reserved
*-----------------------------------------------------------------------
*
*		filename : main.cpp
*		description : demo of Libtorch-SMPL. We provide this software
*		for research purposes only. 
*		Original SMPL model: http://smpl.is.tue.mpg.
*
*                              created by Buzhen Huang at  03/17/2020
*======================================================================*/
#include <iostream>

#ifdef _DEBUG
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "c10.lib")
#else
#pragma comment(lib, "torch.lib")
#pragma comment(lib, "c10.lib")
#endif // _DEBUG

#include "SMPL.h"

using namespace std;


void main(int argc, char** argv)
{
	// init
	SMPL model;
	model.load_model("../SMPL/SMPL_MALE");
	cout << "load SMPL model" << endl;

	// model parameters
	torch::Tensor shape = torch::zeros({ 1, 10 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor pose = torch::zeros({ 1, 72 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor trans = torch::zeros({ 1, 3 }, torch::TensorOptions().dtype(torch::kFloat32));

	model.pose = pose;
	model.shape = shape;
	model.trans = trans;
	model.update(false);

	const string output = "SMPL_MALE.obj";
	model.write_smpl(output);

}




