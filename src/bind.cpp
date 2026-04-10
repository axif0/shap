#include <nanobind/nanobind.h>
#include "kernel.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(_ext, m)
{
    m.doc() = "Internal C++ implementation of SHAP algorithms";
    m.def(
        "_exp_val",
        &kernel::exp_val,
        "nsamples_run"_a,
        "nsamples_added"_a,
        "D"_a,
        "N"_a,
        "weights"_a,
        "y"_a,
        "ey"_a);
}
