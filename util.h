#ifndef UTIL_H
#define UTIL_H


auto relu = [](const double x) -> double {
    return x < 0.0 ? 0.0 : x;
};

auto relu_capped = [](const double x) -> double {
    if(x < 0.0)
        return 0.0;
    else if(x < 1.0)
        return x;
    else
        return 1.0;
};

#endif
