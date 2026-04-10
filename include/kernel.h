#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace kernel
{
    nb::tuple exp_val(
        int nsamples_run,
        int nsamples_added,
        int D,
        int N,
        nb::ndarray<nb::numpy, double, nb::ndim<1>> weights,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> y,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> ey);
}
