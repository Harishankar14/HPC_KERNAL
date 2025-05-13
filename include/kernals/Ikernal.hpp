#ifndef IKERNEL_HPP
#define IKERNEL_HPP

#include <vector>

class IKernel {
public:
    virtual void execute(const std::vector<std::vector<int>>& A,const std::vector<std::vector<int>>& B,std::vector<std::vector<int>>& C) = 0;
    virtual const char* name() const = 0;
    virtual const char* version() const = 0;
    virtual ~IKernel() = default;
};

#endif // IKERNEL_HPP
