#include <iostream>

#include "core/util/pose3d.h"

int main()
{
    std::cout << "Start test.\n";

// Rotation
// 1) initialize test.
    SO3 R_mat;
    R_mat <<  0.9987510,-0.0009691, 0.0499554,
              0.0010291, 0.9999988,-0.0011739,
             -0.0499542, 0.0012239, 0.9987507;


    Rotation3 rot(R_mat);

    std::cout << rot.R() << std::endl;
    std::cout << rot.q() << std::endl;
    std::cout << rot.determinant() << std::endl;


// 2) Inverse test
    SO3 Rinv_mat = R_mat.transpose();
    Rotation3 rot_inv;
    rot_inv << rot.inverse();
    

    std::cout << "Rinv_mat:\n" << Rinv_mat << std::endl;
    std::cout << "det: " << Rinv_mat.determinant() << std::endl;

    std::cout << "Rinv:\n" << rot_inv << std::endl;
    std::cout << "det: " << rot_inv.determinant() << std::endl;

    rot_inv *= rot;
    std::cout << "Rinv:\n" << rot_inv <<std::endl;
    std::cout << "det: " << rot_inv.determinant() << std::endl;



// Pose3d
// 1) initialize test
    Position3 t;
    t << 0.1, 0.2, 0.3;

    Rotation3 rot2(R_mat);

    Pose3D pose;
    pose << (rot2,t);

    std::cout << pose << std::endl;
    std::cout << pose.inverse() << std::endl;

    std::cout << pose.rotation().determinant() << std::endl;

// 2) 
    Rotation3 drot;
    float angle = 2.0*3.1415926535897932384626433832795028841971693*0.00001;
    so3 w(angle,0,0);

    drot << w;

    std::cout << drot.R() << std::endl;
    Rotation3 rot_tmp = rot;
    for(int i = 0; i < 100000; ++i){
        rot_tmp *= drot;

        std::cout << i << "-th rot mult:\n";
        std::cout << rot_tmp.R() << std::endl;
        std::cout << rot_tmp.q() << std::endl;
        std::cout << "determinant: " << rot_tmp.determinant() << std::endl;
    }


        std::cout << "original rot mult:\n";
        std::cout << rot.R() << std::endl;
        std::cout << rot.q() << std::endl;

    Rotation3 rot_diff = rot * rot_tmp.inverse();
    std::cout << rot_diff.q() << std::endl;
// 0.306186 -0.250001  0.918557
//  0.883883  0.433012 -0.176776
// -0.353552  0.866024  0.353554
    std::cout << "End test.\n";
    return 0;
};