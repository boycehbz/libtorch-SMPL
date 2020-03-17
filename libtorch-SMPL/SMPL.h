/*======================================================================
*
*					Copyright (C) 2020, Buzhen Huang
*						   All rights reserved
*-----------------------------------------------------------------------
*
*		filename : SMPL.h
*		description : header file of Libtorch-SMPL. We provide this software
*		for research purposes only.
*		Original SMPL model: http://smpl.is.tue.mpg.
*
*                              created by Buzhen Huang at  03/17/2020
*======================================================================*/
#ifndef __SMPL_
#define __SMPL_

#include <iostream>
#include "torch/script.h"

class SMPL
{
public:
	void write_smpl(const std::string& file);
	void load_model(const std::string& model_dir);
	void update(bool smplify);

public:
	torch::Tensor shape = torch::zeros({ 1, 10 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor pose = torch::zeros({ 1, 72 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor trans = torch::zeros({ 1, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor smpl_verts = torch::zeros({ 6890, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor smpl_joints = torch::zeros({ 24, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor faces = torch::zeros({ 13776, 3 }, torch::TensorOptions().dtype(torch::kInt32));
	float scale = 1.;

private:
	void read_2dparam(torch::Tensor& param, const std::string& file);
	void read_3dparam(torch::Tensor& param, const std::string& file);
	void read_2dparam_int(torch::Tensor& param, const std::string& file);
	int parent(int child);
	void rodrigues(torch::Tensor thetas, torch::Tensor& lRs);
	void _lR2G(torch::Tensor& lRs, torch::Tensor& J, torch::Tensor& G);
	torch::Tensor with_zeros(torch::Tensor x);
	torch::Tensor pack(torch::Tensor x);
private:
	torch::Tensor J_regressor = torch::zeros({ 24, 6890 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor weights = torch::zeros({ 6890, 24 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor posedirs = torch::zeros({ 6890, 3, 207 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor shapedirs = torch::zeros({ 6890, 3, 10 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor v_template = torch::zeros({ 6890, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
	torch::Tensor kintree_table = torch::zeros({ 2, 24 }, torch::TensorOptions().dtype(torch::kInt32));


};



#endif // !_SMPL_
