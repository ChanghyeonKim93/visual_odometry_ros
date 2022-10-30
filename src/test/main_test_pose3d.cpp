#include <iostream>

#include "util/pose3d.h"

int main()
{
    std::cout << "Start test.\n";

// Rotation
// 1) initialize test.
    SO3 R_mat;
    R_mat << 0.306185853,-0.250000803, 0.918557021,
             0.8838825,  0.433011621, -0.176776249,
            -0.35355216, 0.866024084,  0.353553866;

    std::cout << "R_mat:\n" << R_mat << std::endl;

    Rotation3 rot;
    rot.initByRotation(R_mat);
    
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
    


// 2) 


    std::cout << "End test.\n";
    return 0;
};