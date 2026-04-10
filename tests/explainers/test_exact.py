"""Unit tests for the Exact explainer."""

import pickle
import time

import numpy as np
import pytest
from conftest import compare_numpy_outputs_against_baseline

import shap

from . import common


def _compute_grey_code_row_values_python(
    row_values: np.ndarray,
    mask: np.ndarray,
    inds: np.ndarray,
    outputs: np.ndarray,
    shapley_coeff: np.ndarray,
    extended_delta_indexes: np.ndarray,
    noop_code: int,
) -> None:
    """Reference implementation used to validate the C++ extension path."""
    set_size = 0
    num_inds = len(inds)
    for i in range(2**num_inds):
        delta_ind = extended_delta_indexes[i]
        if delta_ind != noop_code:
            mask[delta_ind] = ~mask[delta_ind]
            if mask[delta_ind]:
                set_size += 1
            else:
                set_size -= 1

        on_coeff = shapley_coeff[set_size - 1]
        off_coeff = shapley_coeff[set_size] if set_size < num_inds else 0.0
        out = outputs[i]
        for j in inds:
            if mask[j]:
                row_values[j] += out * on_coeff
            else:
                row_values[j] -= out * off_coeff


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_interactions():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_interactions_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict_proba, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Partition(data), data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker_single_value():
    model, data = common.basic_xgboost_scenario(2)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_auto_masker_minimal():
    model, data = common.basic_xgboost_scenario(2)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(
        shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Partition(data), data
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_single_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(shap.explainers.ExactExplainer, model.predict, shap.maskers.Independent(data), data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_tabular_multi_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    return common.test_additivity(
        shap.explainers.ExactExplainer, model.predict_proba, shap.maskers.Independent(data), data
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(shap.explainers.ExactExplainer, model.predict, data, data)


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_no_model_or_masker():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(
        shap.explainers.ExactExplainer,
        model.predict,
        data,
        data,
        model_saver=False,
        masker_saver=False,
        model_loader=lambda _: model.predict,
        masker_loader=lambda _: data,
    )


@compare_numpy_outputs_against_baseline(func_file=__file__)
def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    return common.test_serialization(
        shap.explainers.ExactExplainer, model.predict, data, data, model_saver=pickle.dump, model_loader=pickle.load
    )


def test_multi_output_with_non_varying_features():
    """Test 2D code path when some features don't vary from background.

    This reproduces a bug in compute_grey_code_row_values_2d where the inner
    loop iterates over rv.shape(0) and indexes rv(rvi, ...) instead of
    iterating over inds.shape(0) and indexing rv(inds(rvi), ...).
    The bug is invisible when all features vary (inds == [0,1,...,M-1]),
    but causes wrong results when only a subset varies.
    """
    # 4 features, multi-output model
    # Background: single sample so we can control exactly which features vary
    background = np.array([[0.0, 1.0, 2.0, 3.0]])

    # Simple linear multi-output model: returns [sum_of_features, 2*sum_of_features]
    def model(X):
        s = X.sum(axis=1)
        return np.column_stack([s, 2 * s])

    # Test sample: features 0 and 2 match the background, features 1 and 3 differ
    # So inds should be [1, 3] (only 2 of 4 features vary)
    test_x = np.array([[0.0, 5.0, 2.0, 7.0]])

    explainer = shap.explainers.ExactExplainer(model, background)
    shap_values = explainer(test_x)

    # Additivity check: base_values + sum(shap_values) == model prediction
    pred = model(test_x)
    reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, pred, atol=1e-10)

    # Non-varying features (0 and 2) should have zero SHAP values
    np.testing.assert_allclose(shap_values.values[0, 0, :], 0.0, atol=1e-10)
    np.testing.assert_allclose(shap_values.values[0, 2, :], 0.0, atol=1e-10)

    # Varying features (1 and 3) should have non-zero SHAP values
    assert np.any(np.abs(shap_values.values[0, 1, :]) > 1e-10)
    assert np.any(np.abs(shap_values.values[0, 3, :]) > 1e-10)


@pytest.mark.parametrize("n_outputs", [1, 3])
def test_compute_grey_code_row_values_cpp_matches_reference(n_outputs: int):
    """C++ implementation matches the original Python algorithm."""
    from shap._explainers import compute_grey_code_row_values

    rng = np.random.default_rng(0)
    num_features = 8
    num_varying = 5
    inds = np.array([0, 2, 3, 5, 7], dtype=np.int64)
    num_masks = 2**num_varying
    noop_code = -1

    outputs = rng.normal(size=(num_masks, n_outputs)).astype(np.float64)
    coeff = np.array([1.0 / (num_varying * (num_varying - 1))] * num_varying, dtype=np.float64)
    extended_delta_indexes = np.array(
        [noop_code, 7, 5, 2, 3, 0, 5, 7, 2, 3, 0, 7, 5, 2, 3, 0, 7, 5, 2, 3, 0, 7, 5, 2, 3, 0, 7, 5, 2, 3, 0, 7],
        dtype=np.int64,
    )

    rv_cpp = np.zeros((num_features, n_outputs), dtype=np.float64)
    rv_ref = np.zeros((num_features, n_outputs), dtype=np.float64)
    mask_cpp = np.zeros(num_features, dtype=bool)
    mask_ref = np.zeros(num_features, dtype=bool)

    compute_grey_code_row_values(rv_cpp, mask_cpp, inds, outputs, coeff, extended_delta_indexes, noop_code)
    _compute_grey_code_row_values_python(rv_ref, mask_ref, inds, outputs, coeff, extended_delta_indexes, noop_code)

    np.testing.assert_allclose(rv_cpp, rv_ref, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(mask_cpp, mask_ref)


@pytest.mark.xslow
def test_compute_grey_code_row_values_cpp_perf_vs_numba():
    """Performance guardrail: C++ should stay close to Numba speed."""
    numba = pytest.importorskip("numba")
    from shap._explainers import compute_grey_code_row_values

    @numba.njit(cache=True)
    def numba_impl(row_values, mask, inds, outputs, shapley_coeff, extended_delta_indexes, noop_code):
        set_size = 0
        num_inds = len(inds)
        for i in range(2**num_inds):
            delta_ind = extended_delta_indexes[i]
            if delta_ind != noop_code:
                mask[delta_ind] = ~mask[delta_ind]
                if mask[delta_ind]:
                    set_size += 1
                else:
                    set_size -= 1

            on_coeff = shapley_coeff[set_size - 1]
            off_coeff = shapley_coeff[set_size] if set_size < num_inds else 0.0
            out = outputs[i]
            for j in inds:
                if mask[j]:
                    row_values[j] += out * on_coeff
                else:
                    row_values[j] -= out * off_coeff

    rng = np.random.default_rng(1)
    num_features = 16
    num_varying = 12
    num_masks = 2**num_varying
    noop_code = -1
    inds = np.arange(num_varying, dtype=np.int64)
    outputs = rng.normal(size=(num_masks, 4)).astype(np.float64)
    coeff = np.linspace(0.01, 0.2, num_varying, dtype=np.float64)
    extended_delta_indexes = rng.choice(np.append(inds, noop_code), size=num_masks).astype(np.int64)

    # Warm up JIT before measuring.
    rv_warm = np.zeros((num_features, 4), dtype=np.float64)
    mask_warm = np.zeros(num_features, dtype=np.bool_)
    numba_impl(rv_warm, mask_warm, inds, outputs, coeff, extended_delta_indexes, noop_code)

    loops = 8
    t0 = time.perf_counter()
    for _ in range(loops):
        rv = np.zeros((num_features, 4), dtype=np.float64)
        mask = np.zeros(num_features, dtype=np.bool_)
        compute_grey_code_row_values(rv, mask, inds, outputs, coeff, extended_delta_indexes, noop_code)
    cpp_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(loops):
        rv = np.zeros((num_features, 4), dtype=np.float64)
        mask = np.zeros(num_features, dtype=np.bool_)
        numba_impl(rv, mask, inds, outputs, coeff, extended_delta_indexes, noop_code)
    numba_time = time.perf_counter() - t1

    # Keep a tolerance for platform/compiler variability.
    assert cpp_time <= numba_time * 1.25, (cpp_time, numba_time)


def test_compute_grey_code_row_values_cpp_stress_memory_safety():
    """Stress-call C++ path repeatedly to catch shape/indexing memory errors."""
    from shap._explainers import compute_grey_code_row_values

    rng = np.random.default_rng(2)
    noop_code = -1
    for _ in range(500):
        num_features = int(rng.integers(4, 16))
        num_varying = int(rng.integers(2, min(10, num_features)))
        inds = np.sort(rng.choice(np.arange(num_features), size=num_varying, replace=False)).astype(np.int64)
        num_masks = 2**num_varying
        out_dim = int(rng.integers(1, 5))

        row_values = np.zeros((num_features, out_dim), dtype=np.float64)
        mask = np.zeros(num_features, dtype=np.bool_)
        outputs = rng.normal(size=(num_masks, out_dim)).astype(np.float64)
        coeff = np.linspace(0.05, 0.25, num_varying, dtype=np.float64)
        allowed = np.append(inds, noop_code)
        extended_delta_indexes = rng.choice(allowed, size=num_masks).astype(np.int64)

        compute_grey_code_row_values(row_values, mask, inds, outputs, coeff, extended_delta_indexes, noop_code)
        assert np.isfinite(row_values).all()


def test_compute_grey_code_row_values_cpp_invalid_index_raises():
    """Invalid index inputs should fail with a Python exception, not memory corruption."""
    from shap._explainers import compute_grey_code_row_values

    row_values = np.zeros((4, 2), dtype=np.float64)
    mask = np.zeros(4, dtype=np.bool_)
    inds = np.array([0, 5], dtype=np.int64)  # out of bounds
    outputs = np.zeros((4, 2), dtype=np.float64)
    coeff = np.array([0.5, 0.5], dtype=np.float64)
    extended_delta_indexes = np.array([-1, 1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError):
        compute_grey_code_row_values(row_values, mask, inds, outputs, coeff, extended_delta_indexes, -1)
