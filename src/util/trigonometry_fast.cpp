#include "util/trigonometry_fast.h"

namespace trigonometry{
    float acos(float x) {
        float negate = float(x < 0);
        x = abs(x);
        float ret = -0.0187293;
        ret = ret * x;
        ret = ret + 0.0742610;
        ret = ret * x;
        ret = ret - 0.2121144;
        ret = ret * x;
        ret = ret + 1.5707288;
        ret = ret * std::sqrt(1.0-x);
        ret = ret - 2 * negate * ret;
        return negate * 3.14159265358979 + ret;
    };

    float asin(float x) {
        float negate = float(x < 0);
        x = abs(x);
        float ret = -0.0187293;
        ret *= x;
        ret += 0.0742610;
        ret *= x;
        ret -= 0.2121144;
        ret *= x;
        ret += 1.5707288;
        ret = 3.14159265358979*0.5 - std::sqrt(1.0 - x)*ret;
        return ret - 2 * negate * ret;
    };
};