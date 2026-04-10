#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>

namespace nb = nanobind;

namespace {

void compute_grey_code_row_values(
    nb::ndarray<double, nb::numpy, nb::c_contig> row_values,
    nb::ndarray<bool, nb::numpy, nb::c_contig> mask,
    nb::ndarray<int64_t, nb::numpy, nb::c_contig> inds,
    nb::ndarray<double, nb::numpy, nb::c_contig> outputs,
    nb::ndarray<double, nb::numpy, nb::c_contig> shapley_coeff,
    nb::ndarray<int64_t, nb::numpy, nb::c_contig> extended_delta_indexes,
    int64_t noop_code
) {
    if (row_values.ndim() < 1 || outputs.ndim() < 1) {
        throw nb::value_error("row_values and outputs must have at least 1 dimension.");
    }
    if (mask.ndim() != 1 || inds.ndim() != 1 || shapley_coeff.ndim() != 1 || extended_delta_indexes.ndim() != 1) {
        throw nb::value_error("mask, inds, shapley_coeff, and extended_delta_indexes must be 1D arrays.");
    }

    const size_t m = inds.shape(0);
    const size_t num_masks = extended_delta_indexes.shape(0);
    if (outputs.shape(0) != num_masks) {
        throw nb::value_error("outputs.shape[0] must match extended_delta_indexes.shape[0].");
    }
    if (row_values.shape(0) != mask.shape(0)) {
        throw nb::value_error("row_values.shape[0] must match mask.shape[0].");
    }
    if (shapley_coeff.shape(0) != m) {
        throw nb::value_error("shapley_coeff length must match inds length.");
    }

    int64_t trailing_size = 1;
    for (size_t d = 1; d < row_values.ndim(); ++d) {
        trailing_size *= row_values.shape(d);
    }
    int64_t outputs_trailing_size = 1;
    for (size_t d = 1; d < outputs.ndim(); ++d) {
        outputs_trailing_size *= outputs.shape(d);
    }
    if (trailing_size != outputs_trailing_size) {
        throw nb::value_error("row_values and outputs trailing dimensions must match.");
    }

    auto *row_values_ptr = row_values.data();
    auto *mask_ptr = mask.data();
    auto *inds_ptr = inds.data();
    auto *outputs_ptr = outputs.data();
    auto *coeff_ptr = shapley_coeff.data();
    auto *extended_ptr = extended_delta_indexes.data();

    int64_t set_size = 0;
    for (size_t i = 0; i < num_masks; ++i) {
        const int64_t delta_ind = extended_ptr[i];
        if (delta_ind != noop_code) {
            if (delta_ind < 0 || static_cast<size_t>(delta_ind) >= mask.shape(0)) {
                throw nb::value_error("extended_delta_indexes contains out-of-bounds feature index.");
            }
            mask_ptr[delta_ind] = !mask_ptr[delta_ind];
            if (mask_ptr[delta_ind]) {
                set_size += 1;
            } else {
                set_size -= 1;
            }
        }

        if (set_size < 0 || set_size > static_cast<int64_t>(m)) {
            throw nb::value_error("Invalid set_size derived from extended_delta_indexes and noop_code.");
        }
        const int64_t on_coeff_index = set_size == 0 ? static_cast<int64_t>(m) - 1 : set_size - 1;
        const double on_coeff = coeff_ptr[on_coeff_index];
        double off_coeff = 0.0;
        if (set_size < static_cast<int64_t>(m)) {
            off_coeff = coeff_ptr[set_size];
        }

        const double *out_ptr = outputs_ptr + static_cast<int64_t>(i) * trailing_size;
        for (size_t j_idx = 0; j_idx < m; ++j_idx) {
            const int64_t j = inds_ptr[j_idx];
            if (j < 0 || static_cast<size_t>(j) >= row_values.shape(0)) {
                throw nb::value_error("inds contains out-of-bounds feature index.");
            }
            double *rv_ptr = row_values_ptr + j * trailing_size;
            if (mask_ptr[j]) {
                for (int64_t t = 0; t < trailing_size; ++t) {
                    rv_ptr[t] += out_ptr[t] * on_coeff;
                }
            } else {
                for (int64_t t = 0; t < trailing_size; ++t) {
                    rv_ptr[t] -= out_ptr[t] * off_coeff;
                }
            }
        }
    }
}

} 

NB_MODULE(_explainers, m) {
    m.def(
        "compute_grey_code_row_values",
        &compute_grey_code_row_values,
        nb::arg("row_values"),
        nb::arg("mask"),
        nb::arg("inds"),
        nb::arg("outputs"),
        nb::arg("shapley_coeff"),
        nb::arg("extended_delta_indexes"),
        nb::arg("noop_code")
    );
}
