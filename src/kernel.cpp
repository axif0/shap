#include "kernel.h"

namespace kernel
{
    nb::tuple exp_val(
        int nsamples_run,
        int nsamples_added,
        int D,
        int N,
        nb::ndarray<nb::numpy, double, nb::ndim<1>> weights,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> y,
        nb::ndarray<nb::numpy, double, nb::ndim<2>> ey)
    {
        auto weights_view = weights.view<double, nb::ndim<1>>();
        auto y_view = y.view<double, nb::ndim<2>>();
        auto ey_view = ey.view<double, nb::ndim<2>>();

        if (D < 0 || N < 0 || nsamples_run < 0 || nsamples_added < 0)
        {
            throw nb::value_error("D, N, nsamples_run, and nsamples_added must be non-negative");
        }
        if (nsamples_run > nsamples_added)
        {
            throw nb::value_error("nsamples_run cannot be greater than nsamples_added");
        }
        if (weights_view.shape(0) != static_cast<size_t>(N))
        {
            throw nb::value_error("weights length does not match N");
        }
        if (y_view.shape(1) != static_cast<size_t>(D))
        {
            throw nb::value_error("y second dimension does not match D");
        }
        if (ey_view.shape(1) != static_cast<size_t>(D))
        {
            throw nb::value_error("ey second dimension does not match D");
        }
        const size_t required_y_rows = static_cast<size_t>(nsamples_added) * static_cast<size_t>(N);
        if (y_view.shape(0) < required_y_rows)
        {
            throw nb::value_error("y first dimension is smaller than nsamples_added * N");
        }
        if (ey_view.shape(0) < static_cast<size_t>(nsamples_added))
        {
            throw nb::value_error("ey first dimension is smaller than nsamples_added");
        }

        for (int i = nsamples_run; i < nsamples_added; ++i)
        {
            for (int k = 0; k < D; ++k)
            {
                double value = 0.0;
                for (int j = 0; j < N; ++j)
                {
                    value += y_view(i * N + j, k) * weights_view(j);
                }
                ey_view(i, k) = value;
            }
            nsamples_run += 1;
        }

        return nb::make_tuple(ey, nsamples_run);
    }
}
