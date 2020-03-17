/*======================================================================
*
*					Copyright (C) 2020, Buzhen Huang
*						   All rights reserved
*-----------------------------------------------------------------------
*
*		filename : SMPL.cpp
*		description : implementation of Libtorch-SMPL. We provide this software
*		for research purposes only.
*		Original SMPL model: http://smpl.is.tue.mpg.
*
*                              created by Buzhen Huang at  03/17/2020
*======================================================================*/
#include <iostream>
#include "SMPL.h"
#include "torch/script.h"

using namespace torch;
using namespace std;


void SMPL::read_2dparam(Tensor& param, const string& file)
{
    ifstream myfile(file);
    assert(myfile.is_open());

    float j_ele;

    auto foo_a = param.accessor<float, 2>();

    for (int i = 0; i < foo_a.size(0); i++)
    {
        for (int j = 0; j < foo_a.size(1); j++)
        {
            myfile >> j_ele;
            foo_a[i][j] = j_ele;
        }
    }
}

void SMPL::read_2dparam_int(Tensor& param, const string& file)
{
    ifstream myfile(file);
    assert(myfile.is_open());

    int j_ele;

    auto foo_a = param.accessor<int, 2>();

    for (int i = 0; i < foo_a.size(0); i++)
    {
        for (int j = 0; j < foo_a.size(1); j++)
        {
            myfile >> j_ele;
            foo_a[i][j] = j_ele;
        }
    }
}

void SMPL::read_3dparam(Tensor& param, const string& file)
{
    ifstream myfile(file);
    assert(myfile.is_open());

    float j_ele;

    auto foo_a = param.accessor<float, 3>();

    for (int i = 0; i < foo_a.size(0); i++)
    {
        for (int j = 0; j < foo_a.size(1); j++)
        {
            for (int k = 0; k < foo_a.size(2); k++)
            {
                myfile >> j_ele;
                foo_a[i][j][k] = j_ele;
            }
        }
    }
}


void SMPL::write_smpl(const string& file)
{
    auto foo_a = smpl_verts.accessor<float, 2>();
    auto foo_f = faces.accessor<int, 2>();

    ofstream outfile(file, ios::trunc);

    for (int i = 0; i < foo_a.size(0); i++)
    {
        outfile << "v " << foo_a[i][0] << " " << foo_a[i][1] << " "
            << foo_a[i][2] << endl;
    };

    for (int i = 0; i < foo_f.size(0); i++)
    {
        outfile << "f " << foo_f[i][0] << " " << foo_f[i][1] << " "
            << foo_f[i][2] << endl;
    };
}

void SMPL::load_model(const string& model_dir)
{
    const string J_reg_dir = model_dir + "/J_regressor.txt";
    const string weig_dir = model_dir + "/weights.txt";
    const string posed_dir = model_dir + "/posedirs.txt";
    const string v_tem_dir = model_dir + "/v_template.txt";
    const string faces_dir = model_dir + "/faces.txt";
    const string shaped_dir = model_dir + "/shapedirs.txt";
    const string kintree_dir = model_dir + "/kintree_table.txt";

    read_2dparam(J_regressor, J_reg_dir);
    read_2dparam(weights, weig_dir);
    read_2dparam(v_template, v_tem_dir);
    read_3dparam(posedirs, posed_dir);
    read_3dparam(shapedirs, shaped_dir);
    read_2dparam_int(faces, faces_dir);
    read_2dparam_int(kintree_table, kintree_dir);
}

void SMPL::update(bool smplify)
{
    auto v_shaped = shapedirs.matmul(shape.permute({ 1,0 }));// + v_template;
    v_shaped = v_shaped.reshape({ 6890,3 });
    v_shaped += v_template;

    v_shaped = v_shaped * scale;

    auto J = J_regressor.matmul(v_shaped);

    Tensor lRs = torch::zeros({ 24, 3, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor G = torch::zeros({ 24, 4, 4 }, torch::TensorOptions().dtype(torch::kFloat32));
    rodrigues(pose, lRs);

    _lR2G(lRs, J, G);

    //(1) Pose shape blending (SMPL formula(9))

    Tensor v_posed = torch::zeros({ 6890, 3 }, torch::TensorOptions().dtype(torch::kFloat32));;
    if (smplify)
    {
        v_posed = v_shaped;
    }
    else
    {
        vector<Tensor> R_cube_t;

        for (int i = 1; i < lRs.size(0); i++)
        {
            R_cube_t.push_back(lRs[i]);
        }

        auto R_cube = torch::stack({ R_cube_t }, 0);
        Tensor I_cube = torch::eye({ 3 }, torch::TensorOptions().dtype(torch::kFloat32)) + \
            torch::zeros({ R_cube.size(0), 3, 3 }, torch::TensorOptions().dtype(torch::kFloat32));

        auto lrotmin = (R_cube - I_cube).reshape({ -1,1 });

        v_posed = torch::matmul(posedirs, lrotmin).squeeze(2);
        v_posed += v_shaped;
    }

    //(2) Skinning (W)

    auto T = torch::tensordot(G, weights,0,1);
    T = T.permute({ 2,0,1 });

    Tensor ones = torch::ones({ v_posed.size(0), 1 }, torch::TensorOptions().dtype(torch::kFloat32));
    auto rest_shape_h = torch::cat({ v_posed, ones }, 1);
    rest_shape_h = rest_shape_h.reshape({ -1,4,1 });


    auto v = torch::matmul(T, rest_shape_h);
    v = v.reshape({ -1,4 }).permute({1,0});

    vector<Tensor> vert;

    for (int i = 0; i < 3; i++)
    {
        vert.push_back(v[i]);
    }

    auto verts = torch::stack({ vert }, 0);
    verts = verts.permute({ 1,0 });

    auto t_trans = trans.reshape({ 1, 3 });

    smpl_verts = verts + t_trans;

    auto joints = torch::tensordot(smpl_verts, J_regressor, 0, 1);
    smpl_joints = joints.permute({ 1,0 });

}

Tensor SMPL::with_zeros(Tensor x)
{

    Tensor ones = torch::zeros({ 1,4 }, torch::TensorOptions().dtype(torch::kFloat32));
    ones[0][3] = 1.0;
    Tensor ret = torch::cat({ x, ones }, 0);

    return ret;
}

Tensor SMPL::pack(Tensor x)
{
    Tensor zeros43 = torch::zeros({ x.size(0), 4, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
    Tensor ret = torch::cat({ zeros43, x }, 2);

    return ret;
}


void SMPL::_lR2G(Tensor& lRs, Tensor& J, Tensor& G)
{
    lRs = lRs.reshape({ 24, 3, 3 });

    vector<Tensor> results;

    auto temp_R = lRs[0];
    auto temp_J = J[0].reshape({ 3,1 });
    auto transform = torch::cat({ temp_R,temp_J }, 1);
    transform = with_zeros(transform);
    results.push_back(transform);

    for (int i = 1; i < kintree_table.size(1); i++)
    {
        temp_R = lRs[i];
        temp_J = J[i] - J[parent(i)];
        temp_J = temp_J.reshape({ 3,1 });
        transform = torch::cat({ temp_R,temp_J }, 1);
        transform = with_zeros(transform);
        transform = torch::matmul(results[parent(i)],transform );

        results.push_back(transform);

    }

    auto stacked = torch::stack({ results }, 0);

    
    Tensor zero = torch::zeros({ 24, 1 }, torch::TensorOptions().dtype(torch::kFloat32));
    auto deformed = torch::cat({ J, zero }, 1);
    deformed = deformed.reshape({ 24,4,1 });
    auto deformed_joint = torch::matmul(stacked, deformed);

    auto temp = pack(deformed_joint);

    G = stacked - temp;

}

void SMPL::rodrigues(Tensor r, Tensor& lRs)
{
    r = r.view({ -1, 3 });
    auto eps = r.clone().normal_(0, 1e-8);
    auto temp = r + eps;
    auto theta = torch::norm(temp, (1,2), true).reshape({ -1,1 });

    int theta_dims = theta.size(0);

    auto r_hat = r / theta; //24*3

    auto r_hatT = r_hat.clone().permute({ 1,0 });

    auto first_col = r_hatT[0].reshape({ 24,1 });
    auto second_col = r_hatT[1].reshape({ 24,1 });
    auto third_col = r_hatT[2].reshape({ 24,1 });

    auto cos = torch::cos(theta);

    auto z_stick = torch::zeros({ theta_dims,1 }, torch::TensorOptions().dtype(torch::kFloat32)); //24*1

    auto m = torch::stack(
        { z_stick, -third_col, second_col, third_col, z_stick,
            -first_col, -second_col, first_col, z_stick }, (1));

    m = m.reshape({-1,3,3});

    auto i_cube = (torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(0) \
        + torch::zeros({ theta_dims, 3, 3 }, torch::TensorOptions().dtype(torch::kFloat32)));


    r_hat = r_hat.view({ 24,1,3 });
    auto A = r_hat.permute({ 0,2,1 });
    auto dot = torch::matmul(A, r_hat);

    

    cos = cos.view({ 24,1,1 });
    theta = theta.view({ 24,1,1 });

    lRs = cos * i_cube + (1 - cos) * dot + torch::sin(theta) * m;
    lRs = lRs.reshape({-1, 3, 3 });
}


int SMPL::parent(int child)
{
    switch (child)
    {
    case 1:
        return 0;
    case 2:
        return 0;
    case 3:
        return 0;
    case 4:
        return 1;
    case 5:
        return 2;
    case 6:
        return 3;
    case 7:
        return 4;
    case 8:
        return 5;
    case 9:
        return 6;
    case 10:
        return 7;
    case 11:
        return 8;
    case 12:
        return 9;
    case 13:
        return 9;
    case 14:
        return 9;
    case 15:
        return 12;
    case 16:
        return 13;
    case 17:
        return 14;
    case 18:
        return 16;
    case 19:
        return 17;
    case 20:
        return 18;
    case 21:
        return 19;
    case 22:
        return 20;
    case 23:
        return 21;
    default:
        break;
    }

}

