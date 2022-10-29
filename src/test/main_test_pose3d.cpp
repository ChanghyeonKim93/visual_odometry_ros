#include <iostream>

#include "util/pose3d.h"

int main()
{
    std::cout << "Start test.\n";
    Rotation3 R;

    std::cout << R.R() << std::endl;
    std::cout << R.q() << std::endl;
    std::cout << R.determinant() << std::endl;

    std::cout << "End test.\n";
    return 0;
};