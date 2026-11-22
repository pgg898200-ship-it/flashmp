// Zero-copy CPython 3.10+ C-API extension for flashmp (PEP 3118 buffer protocol).
// Depends ONLY on <Python.h> and flashmp.hpp — zero NumPy C-API dependency.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "flashmp.hpp"

namespace {

enum class BufKind {
    Float64,
    Int64,
    Int32,
};

static const char* kind_name(BufKind kind) {
    switch (kind) {
        case BufKind::Float64: return "float64";
        case BufKind::Int64:   return "int64";
        case BufKind::Int32:   return "int32";
    }
    return "unknown";
}

static std::size_t kind_align(BufKind kind) {
    switch (kind) {
        case BufKind::Float64: return alignof(double);
        case BufKind::Int64:   return alignof(int64_t);
        case BufKind::Int32:   return alignof(int32_t);
    }
    return 1;
}

static bool check_format_matches(const char* fmt, Py_ssize_t itemsize, BufKind kind) {
    if (!fmt || fmt[0] == '\0') {
        return false;
    }
    const char* p = fmt;
    if (*p == '@' || *p == '=') {
        ++p;
    } else if (*p == '<') {
        if constexpr (std::endian::native != std::endian::little) {
            return false;
        }
        ++p;
    } else if (*p == '>' || *p == '!') {
        if constexpr (std::endian::native != std::endian::big) {
            return false;
        }
        ++p;
    }
    // Require a single native scalar format specifier (rejects complex 'Zd', strings '4w', etc.).
    if (p[0] == '\0' || p[1] != '\0') {
        return false;
    }
    const char code = p[0];
    switch (kind) {
        case BufKind::Float64:
            return itemsize == static_cast<Py_ssize_t>(sizeof(double)) && code == 'd';
        case BufKind::Int64:
            if (itemsize != static_cast<Py_ssize_t>(sizeof(int64_t))) {
                return false;
            }
            if (code == 'q') {
                return true;
            }
            if constexpr (sizeof(long) == sizeof(int64_t)) {
                if (code == 'l') {
                    return true;
                }
            }
            if constexpr (sizeof(int) == sizeof(int64_t)) {
                if (code == 'i') {
                    return true;
                }
            }
            return false;
        case BufKind::Int32:
            if (itemsize != static_cast<Py_ssize_t>(sizeof(int32_t))) {
                return false;
            }
            if (code == 'i') {
                return true;
            }
            if constexpr (sizeof(long) == sizeof(int32_t)) {
                if (code == 'l') {
                    return true;
                }
            }
            return false;
    }
    return false;
}

// RAII PEP 3118 buffer guard. MUST be acquired and destroyed with the GIL held.
struct BufferGuard {
    Py_buffer view{};
    bool active = false;
    int64_t length = 0;

    BufferGuard() = default;
    BufferGuard(const BufferGuard&) = delete;
    BufferGuard& operator=(const BufferGuard&) = delete;

    ~BufferGuard() {
        release();
    }

    void release() {
        if (active) {
            PyBuffer_Release(&view);
            active = false;
            length = 0;
        }
    }

    bool acquire_1d(PyObject* obj, BufKind kind, bool writable, const char* name) {
        release();
        if (obj == nullptr || obj == Py_None) {
            PyErr_Format(PyExc_TypeError, "Argument '%s' must be a %s buffer, not None", name, kind_name(kind));
            return false;
        }
        int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
        if (writable) {
            flags |= PyBUF_WRITABLE;
        }
        if (PyObject_GetBuffer(obj, &view, flags) != 0) {
            return false;
        }
        active = true;

        if (!check_format_matches(view.format, view.itemsize, kind)) {
            const char* got_fmt = view.format ? view.format : "<null>";
            Py_ssize_t got_sz = view.itemsize;
            release();
            PyErr_Format(
                PyExc_TypeError,
                "Argument '%s' must be a native C-contiguous %s buffer (got format='%s', itemsize=%zd)",
                name,
                kind_name(kind),
                got_fmt,
                got_sz
            );
            return false;
        }
        if (view.ndim != 1) {
            int ndim = view.ndim;
            release();
            PyErr_Format(PyExc_ValueError, "Argument '%s' must be a 1D buffer (got ndim=%d)", name, ndim);
            return false;
        }
        if (view.itemsize <= 0 || (view.len % view.itemsize) != 0) {
            release();
            PyErr_Format(PyExc_ValueError, "Argument '%s' has inconsistent byte length", name);
            return false;
        }
        length = static_cast<int64_t>(view.len / view.itemsize);
        if (length > 0) {
            std::size_t align = kind_align(kind);
            if ((reinterpret_cast<std::uintptr_t>(view.buf) % align) != 0) {
                release();
                PyErr_Format(
                    PyExc_ValueError,
                    "Argument '%s' buffer is not %zu-byte aligned for %s",
                    name,
                    align,
                    kind_name(kind)
                );
                return false;
            }
        }
        return true;
    }

    bool acquire_1d_optional(PyObject* obj, BufKind kind, bool writable, const char* name) {
        release();
        if (obj == nullptr || obj == Py_None) {
            return true;
        }
        return acquire_1d(obj, kind, writable, name);
    }

    // Accepts either a 2D (rows, cols) C-contiguous buffer OR a flat 1D buffer of length rows * cols.
    bool acquire_2d(
        PyObject* obj,
        BufKind kind,
        bool writable,
        int64_t rows,
        int64_t cols,
        const char* name
    ) {
        release();
        if (obj == nullptr || obj == Py_None) {
            PyErr_Format(PyExc_TypeError, "Argument '%s' must be a 2D %s buffer, not None", name, kind_name(kind));
            return false;
        }
        int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
        if (writable) {
            flags |= PyBUF_WRITABLE;
        }
        if (PyObject_GetBuffer(obj, &view, flags) != 0) {
            return false;
        }
        active = true;

        if (!check_format_matches(view.format, view.itemsize, kind)) {
            const char* got_fmt = view.format ? view.format : "<null>";
            Py_ssize_t got_sz = view.itemsize;
            release();
            PyErr_Format(
                PyExc_TypeError,
                "Argument '%s' must be a native C-contiguous %s buffer (got format='%s', itemsize=%zd)",
                name,
                kind_name(kind),
                got_fmt,
                got_sz
            );
            return false;
        }
        if (view.itemsize <= 0 || (view.len % view.itemsize) != 0) {
            release();
            PyErr_Format(PyExc_ValueError, "Argument '%s' has inconsistent byte length", name);
            return false;
        }
        if (rows < 0 || cols < 0 || (rows > 0 && cols > std::numeric_limits<int64_t>::max() / rows)) {
            release();
            PyErr_Format(PyExc_OverflowError, "Argument '%s' dimensions overflow int64", name);
            return false;
        }
        const int64_t total = static_cast<int64_t>(view.len / view.itemsize);
        const int64_t expected = rows * cols;
        if (view.ndim == 2) {
            if (!view.shape || view.shape[0] != rows || view.shape[1] != cols) {
                Py_ssize_t s0 = view.shape ? view.shape[0] : -1;
                Py_ssize_t s1 = view.shape ? view.shape[1] : -1;
                release();
                PyErr_Format(
                    PyExc_ValueError,
                    "Argument '%s' has shape (%zd, %zd), expected (%lld, %lld)",
                    name,
                    s0,
                    s1,
                    static_cast<long long>(rows),
                    static_cast<long long>(cols)
                );
                return false;
            }
        } else if (view.ndim == 1) {
            if (total != expected) {
                release();
                PyErr_Format(
                    PyExc_ValueError,
                    "Argument '%s' has length %lld, expected %lld (%lld * %lld)",
                    name,
                    static_cast<long long>(total),
                    static_cast<long long>(expected),
                    static_cast<long long>(rows),
                    static_cast<long long>(cols)
                );
                return false;
            }
        } else {
            int ndim = view.ndim;
            release();
            PyErr_Format(PyExc_ValueError, "Argument '%s' must be 2D or flat 1D (got ndim=%d)", name, ndim);
            return false;
        }
        length = total;
        if (length > 0) {
            std::size_t align = kind_align(kind);
            if ((reinterpret_cast<std::uintptr_t>(view.buf) % align) != 0) {
                release();
                PyErr_Format(
                    PyExc_ValueError,
                    "Argument '%s' buffer is not %zu-byte aligned for %s",
                    name,
                    align,
                    kind_name(kind)
                );
                return false;
            }
        }
        return true;
    }

    template <typename T>
    T* data() const {
        return (active && length > 0) ? static_cast<T*>(view.buf) : nullptr;
    }

    template <typename T>
    const T* cdata() const {
        return (active && length > 0) ? static_cast<const T*>(view.buf) : nullptr;
    }
};

static bool parse_strict_i64(PyObject* obj, const char* name, int64_t& out) {
    if (PyBool_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "Argument '%s' must be an integer, not bool", name);
        return false;
    }
    PyObject* idx = PyNumber_Index(obj);
    if (!idx) {
        return false;
    }
    long long val = PyLong_AsLongLong(idx);
    Py_DECREF(idx);
    if (val == -1 && PyErr_Occurred()) {
        return false;
    }
    out = static_cast<int64_t>(val);
    return true;
}

static bool parse_strict_int(PyObject* obj, const char* name, int& out) {
    int64_t v = 0;
    if (!parse_strict_i64(obj, name, v)) {
        return false;
    }
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        PyErr_Format(PyExc_OverflowError, "Argument '%s' out of C int range", name);
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_ez_arg(PyObject* obj, int64_t& out) {
    if (obj == nullptr || obj == Py_None) {
        out = -1;
        return true;
    }
    if (!parse_strict_i64(obj, "ez", out)) {
        return false;
    }
    if (out < -1) {
        PyErr_Format(PyExc_ValueError, "ez must be >= -1 or None, got %lld", static_cast<long long>(out));
        return false;
    }
    return true;
}

static bool parse_strict_double(PyObject* obj, const char* name, double& out) {
    if (PyFloat_CheckExact(obj)) {
        out = PyFloat_AS_DOUBLE(obj);
        return true;
    }
    if (PyBool_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "Argument '%s' must be a real number, not bool", name);
        return false;
    }
    double val = PyFloat_AsDouble(obj);
    if (val == -1.0 && PyErr_Occurred()) {
        return false;
    }
    out = val;
    return true;
}

static bool check_exact_len(const BufferGuard& g, int64_t expected, const char* name) {
    if (g.active && g.length != expected) {
        PyErr_Format(
            PyExc_ValueError,
            "Output buffer '%s' has length %lld, expected %lld",
            name,
            static_cast<long long>(g.length),
            static_cast<long long>(expected)
        );
        return false;
    }
    return true;
}

static bool check_min_len(const BufferGuard& g, int64_t min_expected, const char* name) {
    if (g.active && g.length < min_expected) {
        PyErr_Format(
            PyExc_ValueError,
            "Output buffer '%s' has length %lld, need at least %lld",
            name,
            static_cast<long long>(g.length),
            static_cast<long long>(min_expected)
        );
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Module-level batch functions
// ---------------------------------------------------------------------------

static PyObject* py_version(PyObject* /*self*/, PyObject* /*args*/) {
    const char* v = fmp_version();
    return PyUnicode_FromString(v ? v : "0.2.0");
}

static PyObject* py_simd_isa(PyObject* /*self*/, PyObject* /*args*/) {
    const char* isa = fmp_simd_isa();
    return PyUnicode_FromString(isa ? isa : "SCALAR");
}

// fast_self_join(buf, m, ez, threads, out_mp, out_idx, out_lmp=None, out_lidx=None, out_rmp=None, out_ridx=None)
static PyObject* py_fast_self_join(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static char k_buf[] = "buf";
    static char k_m[] = "m";
    static char k_ez[] = "ez";
    static char k_threads[] = "threads";
    static char k_out_mp[] = "out_mp";
    static char k_out_idx[] = "out_idx";
    static char k_out_lmp[] = "out_lmp";
    static char k_out_lidx[] = "out_lidx";
    static char k_out_rmp[] = "out_rmp";
    static char k_out_ridx[] = "out_ridx";
    static char* kwlist[] = {
        k_buf, k_m, k_ez, k_threads, k_out_mp, k_out_idx,
        k_out_lmp, k_out_lidx, k_out_rmp, k_out_ridx, nullptr
    };

    PyObject* buf_obj = nullptr;
    PyObject* m_obj = nullptr;
    PyObject* ez_obj = nullptr;
    PyObject* threads_obj = nullptr;
    PyObject* out_mp_obj = nullptr;
    PyObject* out_idx_obj = nullptr;
    PyObject* out_lmp_obj = Py_None;
    PyObject* out_lidx_obj = Py_None;
    PyObject* out_rmp_obj = Py_None;
    PyObject* out_ridx_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOO|OOOO:fast_self_join", kwlist,
            &buf_obj, &m_obj, &ez_obj, &threads_obj, &out_mp_obj, &out_idx_obj,
            &out_lmp_obj, &out_lidx_obj, &out_rmp_obj, &out_ridx_obj)) {
        return nullptr;
    }

    int64_t m = 0;
    int64_t ez = -1;
    int threads = 0;
    if (!parse_strict_i64(m_obj, "m", m) ||
        !parse_ez_arg(ez_obj, ez) ||
        !parse_strict_int(threads_obj, "threads", threads)) {
        return nullptr;
    }
    if (m < 3) {
        PyErr_Format(PyExc_ValueError, "Subsequence length m must be >= 3, got %lld", static_cast<long long>(m));
        return nullptr;
    }

    BufferGuard t_buf, mp_buf, idx_buf, lmp_buf, lidx_buf, rmp_buf, ridx_buf;
    if (!t_buf.acquire_1d(buf_obj, BufKind::Float64, false, "buf")) {
        return nullptr;
    }
    const int64_t N = t_buf.length;
    if (N < m) {
        PyErr_Format(
            PyExc_ValueError,
            "Time series length N=%lld must be >= m=%lld",
            static_cast<long long>(N),
            static_cast<long long>(m)
        );
        return nullptr;
    }
    const int64_t n = N - m + 1;

    if (!mp_buf.acquire_1d(out_mp_obj, BufKind::Float64, true, "out_mp") ||
        !idx_buf.acquire_1d(out_idx_obj, BufKind::Int64, true, "out_idx") ||
        !lmp_buf.acquire_1d_optional(out_lmp_obj, BufKind::Float64, true, "out_lmp") ||
        !lidx_buf.acquire_1d_optional(out_lidx_obj, BufKind::Int64, true, "out_lidx") ||
        !rmp_buf.acquire_1d_optional(out_rmp_obj, BufKind::Float64, true, "out_rmp") ||
        !ridx_buf.acquire_1d_optional(out_ridx_obj, BufKind::Int64, true, "out_ridx")) {
        return nullptr;
    }

    if (!check_exact_len(mp_buf, n, "out_mp") ||
        !check_exact_len(idx_buf, n, "out_idx") ||
        !check_exact_len(lmp_buf, n, "out_lmp") ||
        !check_exact_len(lidx_buf, n, "out_lidx") ||
        !check_exact_len(rmp_buf, n, "out_rmp") ||
        !check_exact_len(ridx_buf, n, "out_ridx")) {
        return nullptr;
    }

    const double* t_ptr = t_buf.cdata<double>();
    double* mp_ptr = mp_buf.data<double>();
    int64_t* idx_ptr = idx_buf.data<int64_t>();
    double* lmp_ptr = lmp_buf.data<double>();
    int64_t* lidx_ptr = lidx_buf.data<int64_t>();
    double* rmp_ptr = rmp_buf.data<double>();
    int64_t* ridx_ptr = ridx_buf.data<int64_t>();

    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_self_join(t_ptr, N, m, ez, threads, mp_ptr, idx_ptr, lmp_ptr, lidx_ptr, rmp_ptr, ridx_ptr);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_self_join failed with error code %d", rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

// fast_ab_join(buf_a, buf_b, m, threads, out_mp_a, out_idx_a, out_mp_b=None, out_idx_b=None)
static PyObject* py_fast_ab_join(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static char k_buf_a[] = "buf_a";
    static char k_buf_b[] = "buf_b";
    static char k_m[] = "m";
    static char k_threads[] = "threads";
    static char k_out_mp_a[] = "out_mp_a";
    static char k_out_idx_a[] = "out_idx_a";
    static char k_out_mp_b[] = "out_mp_b";
    static char k_out_idx_b[] = "out_idx_b";
    static char* kwlist[] = {
        k_buf_a, k_buf_b, k_m, k_threads, k_out_mp_a, k_out_idx_a, k_out_mp_b, k_out_idx_b, nullptr
    };

    PyObject* buf_a_obj = nullptr;
    PyObject* buf_b_obj = nullptr;
    PyObject* m_obj = nullptr;
    PyObject* threads_obj = nullptr;
    PyObject* out_mp_a_obj = nullptr;
    PyObject* out_idx_a_obj = nullptr;
    PyObject* out_mp_b_obj = Py_None;
    PyObject* out_idx_b_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOO|OO:fast_ab_join", kwlist,
            &buf_a_obj, &buf_b_obj, &m_obj, &threads_obj,
            &out_mp_a_obj, &out_idx_a_obj, &out_mp_b_obj, &out_idx_b_obj)) {
        return nullptr;
    }

    int64_t m = 0;
    int threads = 0;
    if (!parse_strict_i64(m_obj, "m", m) ||
        !parse_strict_int(threads_obj, "threads", threads)) {
        return nullptr;
    }
    if (m < 3) {
        PyErr_Format(PyExc_ValueError, "Subsequence length m must be >= 3, got %lld", static_cast<long long>(m));
        return nullptr;
    }

    const bool has_mp_b = (out_mp_b_obj != nullptr && out_mp_b_obj != Py_None);
    const bool has_idx_b = (out_idx_b_obj != nullptr && out_idx_b_obj != Py_None);
    if (has_mp_b != has_idx_b) {
        PyErr_SetString(PyExc_ValueError, "out_mp_b and out_idx_b must either both be provided or both be None");
        return nullptr;
    }

    BufferGuard a_buf, b_buf, mp_a_buf, idx_a_buf, mp_b_buf, idx_b_buf;
    if (!a_buf.acquire_1d(buf_a_obj, BufKind::Float64, false, "buf_a") ||
        !b_buf.acquire_1d(buf_b_obj, BufKind::Float64, false, "buf_b")) {
        return nullptr;
    }
    const int64_t NA = a_buf.length;
    const int64_t NB = b_buf.length;
    if (NA < m) {
        PyErr_Format(PyExc_ValueError, "Series A length %lld must be >= m=%lld", static_cast<long long>(NA), static_cast<long long>(m));
        return nullptr;
    }
    if (NB < m) {
        PyErr_Format(PyExc_ValueError, "Series B length %lld must be >= m=%lld", static_cast<long long>(NB), static_cast<long long>(m));
        return nullptr;
    }
    const int64_t nA = NA - m + 1;
    const int64_t nB = NB - m + 1;

    if (!mp_a_buf.acquire_1d(out_mp_a_obj, BufKind::Float64, true, "out_mp_a") ||
        !idx_a_buf.acquire_1d(out_idx_a_obj, BufKind::Int64, true, "out_idx_a") ||
        !mp_b_buf.acquire_1d_optional(out_mp_b_obj, BufKind::Float64, true, "out_mp_b") ||
        !idx_b_buf.acquire_1d_optional(out_idx_b_obj, BufKind::Int64, true, "out_idx_b")) {
        return nullptr;
    }

    if (!check_exact_len(mp_a_buf, nA, "out_mp_a") ||
        !check_exact_len(idx_a_buf, nA, "out_idx_a") ||
        !check_exact_len(mp_b_buf, nB, "out_mp_b") ||
        !check_exact_len(idx_b_buf, nB, "out_idx_b")) {
        return nullptr;
    }

    const double* a_ptr = a_buf.cdata<double>();
    const double* b_ptr = b_buf.cdata<double>();
    double* mp_a_ptr = mp_a_buf.data<double>();
    int64_t* idx_a_ptr = idx_a_buf.data<int64_t>();
    double* mp_b_ptr = mp_b_buf.data<double>();
    int64_t* idx_b_ptr = idx_b_buf.data<int64_t>();

    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_ab_join(a_ptr, NA, b_ptr, NB, m, threads, mp_a_ptr, idx_a_ptr, mp_b_ptr, idx_b_ptr);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_ab_join failed with error code %d", rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

// fast_leak_scan(ref_buf, qry_buf, m, corr_thresh, check_neg, threads, out_corr, out_idx) -> hit_count
static PyObject* py_fast_leak_scan(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static char k_ref[] = "ref_buf";
    static char k_qry[] = "qry_buf";
    static char k_m[] = "m";
    static char k_thresh[] = "corr_thresh";
    static char k_neg[] = "check_neg";
    static char k_threads[] = "threads";
    static char k_out_corr[] = "out_corr";
    static char k_out_idx[] = "out_idx";
    static char* kwlist[] = {
        k_ref, k_qry, k_m, k_thresh, k_neg, k_threads, k_out_corr, k_out_idx, nullptr
    };

    PyObject* ref_obj = nullptr;
    PyObject* qry_obj = nullptr;
    PyObject* m_obj = nullptr;
    PyObject* thresh_obj = nullptr;
    PyObject* neg_obj = nullptr;
    PyObject* threads_obj = nullptr;
    PyObject* out_corr_obj = nullptr;
    PyObject* out_idx_obj = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOOO:fast_leak_scan", kwlist,
            &ref_obj, &qry_obj, &m_obj, &thresh_obj, &neg_obj, &threads_obj,
            &out_corr_obj, &out_idx_obj)) {
        return nullptr;
    }

    int64_t m = 0;
    double corr_thresh = 0.0;
    int threads = 0;
    if (!parse_strict_i64(m_obj, "m", m) ||
        !parse_strict_double(thresh_obj, "corr_thresh", corr_thresh) ||
        !parse_strict_int(threads_obj, "threads", threads)) {
        return nullptr;
    }
    int check_neg = PyObject_IsTrue(neg_obj);
    if (check_neg < 0) {
        return nullptr;
    }
    if (m < 3) {
        PyErr_Format(PyExc_ValueError, "Subsequence length m must be >= 3, got %lld", static_cast<long long>(m));
        return nullptr;
    }
    if (!std::isfinite(corr_thresh)) {
        PyErr_SetString(PyExc_ValueError, "corr_thresh must be finite");
        return nullptr;
    }

    BufferGuard ref_buf, qry_buf, corr_buf, idx_buf;
    if (!ref_buf.acquire_1d(ref_obj, BufKind::Float64, false, "ref_buf") ||
        !qry_buf.acquire_1d(qry_obj, BufKind::Float64, false, "qry_buf")) {
        return nullptr;
    }
    const int64_t N_ref = ref_buf.length;
    const int64_t N_qry = qry_buf.length;
    if (N_ref < m) {
        PyErr_Format(PyExc_ValueError, "ref length (%lld) must be >= m (%lld)", static_cast<long long>(N_ref), static_cast<long long>(m));
        return nullptr;
    }
    if (N_qry < m) {
        PyErr_Format(PyExc_ValueError, "qry length (%lld) must be >= m (%lld)", static_cast<long long>(N_qry), static_cast<long long>(m));
        return nullptr;
    }
    const int64_t n_qry = N_qry - m + 1;

    if (!corr_buf.acquire_1d(out_corr_obj, BufKind::Float64, true, "out_corr") ||
        !idx_buf.acquire_1d(out_idx_obj, BufKind::Int64, true, "out_idx")) {
        return nullptr;
    }
    if (!check_exact_len(corr_buf, n_qry, "out_corr") ||
        !check_exact_len(idx_buf, n_qry, "out_idx")) {
        return nullptr;
    }

    const double* ref_ptr = ref_buf.cdata<double>();
    const double* qry_ptr = qry_buf.cdata<double>();
    double* corr_ptr = corr_buf.data<double>();
    int64_t* idx_ptr = idx_buf.data<int64_t>();
    int64_t hit_count = 0;

    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_leak_scan(
        ref_ptr, N_ref, qry_ptr, N_qry, m, corr_thresh, check_neg, threads,
        corr_ptr, idx_ptr, &hit_count
    );
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_leak_scan failed with error code %d", rc);
        return nullptr;
    }
    return PyLong_FromLongLong(hit_count);
}

// fast_mstump(buf_2d, D, N, m, ez, threads, out_mp, out_idx)
static PyObject* py_fast_mstump(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static char k_buf[] = "buf_2d";
    static char k_D[] = "D";
    static char k_N[] = "N";
    static char k_m[] = "m";
    static char k_ez[] = "ez";
    static char k_threads[] = "threads";
    static char k_out_mp[] = "out_mp";
    static char k_out_idx[] = "out_idx";
    static char* kwlist[] = {
        k_buf, k_D, k_N, k_m, k_ez, k_threads, k_out_mp, k_out_idx, nullptr
    };

    PyObject* buf_obj = nullptr;
    PyObject* D_obj = nullptr;
    PyObject* N_obj = nullptr;
    PyObject* m_obj = nullptr;
    PyObject* ez_obj = nullptr;
    PyObject* threads_obj = nullptr;
    PyObject* out_mp_obj = nullptr;
    PyObject* out_idx_obj = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOOO:fast_mstump", kwlist,
            &buf_obj, &D_obj, &N_obj, &m_obj, &ez_obj, &threads_obj,
            &out_mp_obj, &out_idx_obj)) {
        return nullptr;
    }

    int64_t D = 0;
    int64_t N = 0;
    int64_t m = 0;
    int64_t ez = -1;
    int threads = 0;
    if (!parse_strict_i64(D_obj, "D", D) ||
        !parse_strict_i64(N_obj, "N", N) ||
        !parse_strict_i64(m_obj, "m", m) ||
        !parse_ez_arg(ez_obj, ez) ||
        !parse_strict_int(threads_obj, "threads", threads)) {
        return nullptr;
    }
    if (D < 1) {
        PyErr_Format(PyExc_ValueError, "D must be >= 1, got %lld", static_cast<long long>(D));
        return nullptr;
    }
    if (m < 3) {
        PyErr_Format(PyExc_ValueError, "Subsequence length m must be >= 3, got %lld", static_cast<long long>(m));
        return nullptr;
    }
    if (N < m) {
        PyErr_Format(PyExc_ValueError, "Series length N=%lld must be >= m=%lld", static_cast<long long>(N), static_cast<long long>(m));
        return nullptr;
    }
    if (D > std::numeric_limits<int64_t>::max() / N) {
        PyErr_SetString(PyExc_OverflowError, "D * N overflows int64");
        return nullptr;
    }
    const int64_t n = N - m + 1;

    BufferGuard t_buf, mp_buf, idx_buf;
    if (!t_buf.acquire_2d(buf_obj, BufKind::Float64, false, D, N, "buf_2d") ||
        !mp_buf.acquire_2d(out_mp_obj, BufKind::Float64, true, D, n, "out_mp") ||
        !idx_buf.acquire_2d(out_idx_obj, BufKind::Int64, true, D, n, "out_idx")) {
        return nullptr;
    }

    const double* t_ptr = t_buf.cdata<double>();
    double* mp_ptr = mp_buf.data<double>();
    int64_t* idx_ptr = idx_buf.data<int64_t>();

    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_mstump(t_ptr, D, N, m, ez, threads, mp_ptr, idx_ptr);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_mstump failed with error code %d", rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

// fast_floss_cac(idx_buf, L, out_cac)
static PyObject* py_fast_floss_cac(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static char k_idx[] = "idx_buf";
    static char k_L[] = "L";
    static char k_out[] = "out_cac";
    static char* kwlist[] = {k_idx, k_L, k_out, nullptr};

    PyObject* idx_obj = nullptr;
    PyObject* L_obj = nullptr;
    PyObject* out_obj = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO:fast_floss_cac", kwlist, &idx_obj, &L_obj, &out_obj)) {
        return nullptr;
    }

    int64_t L = 0;
    if (!parse_strict_i64(L_obj, "L", L)) {
        return nullptr;
    }
    if (L < 1) {
        L = 1;
    }

    BufferGuard idx_buf, cac_buf;
    if (!idx_buf.acquire_1d(idx_obj, BufKind::Int64, false, "idx_buf") ||
        !cac_buf.acquire_1d(out_obj, BufKind::Float64, true, "out_cac")) {
        return nullptr;
    }
    const int64_t n = idx_buf.length;
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "idx_buf must be non-empty");
        return nullptr;
    }
    if (!check_exact_len(cac_buf, n, "out_cac")) {
        return nullptr;
    }

    const int64_t* idx_ptr = idx_buf.cdata<int64_t>();
    double* cac_ptr = cac_buf.data<double>();

    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_floss_cac(idx_ptr, n, L, cac_ptr);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_floss_cac failed with error code %d", rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

// fast_auto_window(buf, min_m=8, max_m=512) -> int
static PyObject* py_fast_auto_window(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    static char k_buf[] = "buf";
    static char k_min[] = "min_m";
    static char k_max[] = "max_m";
    static char* kwlist[] = {k_buf, k_min, k_max, nullptr};

    PyObject* buf_obj = nullptr;
    PyObject* min_obj = nullptr;
    PyObject* max_obj = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO:fast_auto_window", kwlist, &buf_obj, &min_obj, &max_obj)) {
        return nullptr;
    }

    int64_t min_m = 8;
    int64_t max_m = 512;
    if (min_obj != nullptr && !parse_strict_i64(min_obj, "min_m", min_m)) {
        return nullptr;
    }
    if (max_obj != nullptr && !parse_strict_i64(max_obj, "max_m", max_m)) {
        return nullptr;
    }

    BufferGuard t_buf;
    if (!t_buf.acquire_1d(buf_obj, BufKind::Float64, false, "buf")) {
        return nullptr;
    }
    if (t_buf.length <= 0) {
        PyErr_SetString(PyExc_ValueError, "buf must be non-empty");
        return nullptr;
    }

    const double* t_ptr = t_buf.cdata<double>();
    const int64_t N = t_buf.length;
    int64_t res = 3;
    Py_BEGIN_ALLOW_THREADS
    res = fmp_auto_window(t_ptr, N, min_m, max_m);
    Py_END_ALLOW_THREADS

    return PyLong_FromLongLong(res);
}

// ---------------------------------------------------------------------------
// FastStream extension type wrapping fmp_stream_t*
// ---------------------------------------------------------------------------

struct FastStreamObject {
    PyObject_HEAD
    fmp_stream_t* ptr;
    int64_t m;
    int64_t window;
    int64_t excl_zone;
    double alpha;
    int egress;
    int track_full_mp;
    int in_use;
};

static bool stream_require_ready(FastStreamObject* self) {
    if (self->in_use) {
        PyErr_SetString(PyExc_RuntimeError, "FastStream is concurrently in use by another thread");
        return false;
    }
    if (!self->ptr) {
        PyErr_SetString(PyExc_RuntimeError, "FastStream has been closed");
        return false;
    }
    return true;
}

static void FastStream_dealloc(FastStreamObject* self) {
    if (self->ptr) {
        fmp_stream_destroy(self->ptr);
        self->ptr = nullptr;
    }
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static int FastStream_init(FastStreamObject* self, PyObject* args, PyObject* kwargs) {
    static char k_m[] = "m";
    static char k_window[] = "window";
    static char k_ez[] = "ez";
    static char k_egress[] = "egress";
    static char k_alpha[] = "alpha";
    static char k_track[] = "track_full_mp";
    static char* kwlist[] = {k_m, k_window, k_ez, k_egress, k_alpha, k_track, nullptr};

    PyObject* m_obj = nullptr;
    PyObject* window_obj = nullptr;
    PyObject* ez_obj = Py_None;
    PyObject* egress_obj = Py_True;
    PyObject* alpha_obj = nullptr;
    PyObject* track_obj = Py_True;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|OOOOO:FastStream", kwlist,
            &m_obj, &window_obj, &ez_obj, &egress_obj, &alpha_obj, &track_obj)) {
        return -1;
    }

    int64_t m = 0;
    int64_t window = 4096;
    int64_t ez = -1;
    double alpha = 0.05;
    if (!parse_strict_i64(m_obj, "m", m)) {
        return -1;
    }
    if (window_obj != nullptr && !parse_strict_i64(window_obj, "window", window)) {
        return -1;
    }
    if (!parse_ez_arg(ez_obj, ez)) {
        return -1;
    }
    int egress = PyObject_IsTrue(egress_obj);
    if (egress < 0) {
        return -1;
    }
    if (alpha_obj != nullptr && !parse_strict_double(alpha_obj, "alpha", alpha)) {
        return -1;
    }
    int track_full_mp = PyObject_IsTrue(track_obj);
    if (track_full_mp < 0) {
        return -1;
    }

    if (m < 3) {
        PyErr_Format(PyExc_ValueError, "m must be >= 3, got %lld", static_cast<long long>(m));
        return -1;
    }
    if (window < 4) {
        PyErr_Format(PyExc_ValueError, "window must be >= 4, got %lld", static_cast<long long>(window));
        return -1;
    }
    if (!std::isfinite(alpha) || !(alpha > 0.0 && alpha < 1.0)) {
        PyErr_SetString(PyExc_ValueError, "alpha must be finite and in (0, 1)");
        return -1;
    }
    if (self->in_use) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot re-initialize FastStream while in use");
        return -1;
    }
    if (self->ptr) {
        fmp_stream_destroy(self->ptr);
        self->ptr = nullptr;
    }

    fmp_stream_t* s = fmp_stream_create(m, window, ez, egress, alpha, track_full_mp);
    if (!s) {
        PyErr_SetString(PyExc_RuntimeError, "fmp_stream_create failed to allocate native stream state");
        return -1;
    }

    self->ptr = s;
    self->m = m;
    self->window = window;
    self->excl_zone = (ez < 0) ? fmp_default_excl_zone(m) : ez;
    self->alpha = alpha;
    self->egress = egress;
    self->track_full_mp = track_full_mp;
    self->in_use = 0;
    return 0;
}

static PyObject* FastStream_init_data(FastStreamObject* self, PyObject* arg) {
    if (!stream_require_ready(self)) {
        return nullptr;
    }
    BufferGuard buf;
    if (!buf.acquire_1d(arg, BufKind::Float64, false, "buf")) {
        return nullptr;
    }
    if (buf.length == 0) {
        Py_RETURN_NONE;
    }

    fmp_stream_t* s = self->ptr;
    const double* ptr = buf.cdata<double>();
    const int64_t n = buf.length;
    int rc = 0;

    self->in_use = 1;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_stream_init(s, ptr, n);
    Py_END_ALLOW_THREADS
    self->in_use = 0;

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_stream_init failed with error code %d", rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

// Hot-path single tick update (METH_O — zero tuple parsing, zero ctypes marshalling).
static PyObject* FastStream_update(FastStreamObject* self, PyObject* arg) {
    if (__builtin_expect(self->in_use != 0, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "FastStream is concurrently in use by another thread");
        return nullptr;
    }
    if (__builtin_expect(self->ptr == nullptr, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "FastStream has been closed");
        return nullptr;
    }

    double x = 0.0;
    if (__builtin_expect(PyFloat_CheckExact(arg), 1)) {
        x = PyFloat_AS_DOUBLE(arg);
    } else if (PyBool_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Tick value x must be a real number, not bool");
        return nullptr;
    } else {
        x = PyFloat_AsDouble(arg);
        if (x == -1.0 && PyErr_Occurred()) {
            return nullptr;
        }
    }

    fmp_tick_result_t res{};
    int rc = fmp_stream_update(self->ptr, x, &res);
    if (__builtin_expect(rc != 0, 0)) {
        PyErr_Format(PyExc_RuntimeError, "fmp_stream_update failed with error code %d", rc);
        return nullptr;
    }

    PyObject* o0 = PyLong_FromLongLong(res.gid);
    PyObject* o1 = PyFloat_FromDouble(res.left_dist);
    PyObject* o2 = PyLong_FromLongLong(res.left_idx);
    PyObject* o3 = PyFloat_FromDouble(res.discord_z);
    PyObject* o4 = PyFloat_FromDouble(res.p_value);
    PyObject* o5 = PyFloat_FromDouble(res.e_value);
    PyObject* o6 = PyBool_FromLong(res.is_alert ? 1 : 0);
    PyObject* o7 = PyLong_FromLong(res.n_repaired);

    if (__builtin_expect(!o0 || !o1 || !o2 || !o3 || !o4 || !o5 || !o6 || !o7, 0)) {
        Py_XDECREF(o0);
        Py_XDECREF(o1);
        Py_XDECREF(o2);
        Py_XDECREF(o3);
        Py_XDECREF(o4);
        Py_XDECREF(o5);
        Py_XDECREF(o6);
        Py_XDECREF(o7);
        return nullptr;
    }

    PyObject* tup = PyTuple_New(8);
    if (__builtin_expect(!tup, 0)) {
        Py_DECREF(o0);
        Py_DECREF(o1);
        Py_DECREF(o2);
        Py_DECREF(o3);
        Py_DECREF(o4);
        Py_DECREF(o5);
        Py_DECREF(o6);
        Py_DECREF(o7);
        return nullptr;
    }

    PyTuple_SET_ITEM(tup, 0, o0);
    PyTuple_SET_ITEM(tup, 1, o1);
    PyTuple_SET_ITEM(tup, 2, o2);
    PyTuple_SET_ITEM(tup, 3, o3);
    PyTuple_SET_ITEM(tup, 4, o4);
    PyTuple_SET_ITEM(tup, 5, o5);
    PyTuple_SET_ITEM(tup, 6, o6);
    PyTuple_SET_ITEM(tup, 7, o7);
    return tup;
}

// update_batch(xs_buf, out_ld=None, out_li=None, out_ev=None, out_al=None)
static PyObject* FastStream_update_batch(FastStreamObject* self, PyObject* args, PyObject* kwargs) {
    static char k_xs[] = "xs_buf";
    static char k_ld[] = "out_ld";
    static char k_li[] = "out_li";
    static char k_ev[] = "out_ev";
    static char k_al[] = "out_al";
    static char* kwlist[] = {k_xs, k_ld, k_li, k_ev, k_al, nullptr};

    PyObject* xs_obj = nullptr;
    PyObject* ld_obj = Py_None;
    PyObject* li_obj = Py_None;
    PyObject* ev_obj = Py_None;
    PyObject* al_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|OOOO:update_batch", kwlist,
            &xs_obj, &ld_obj, &li_obj, &ev_obj, &al_obj)) {
        return nullptr;
    }
    if (!stream_require_ready(self)) {
        return nullptr;
    }

    BufferGuard xs_buf, ld_buf, li_buf, ev_buf, al_buf;
    if (!xs_buf.acquire_1d(xs_obj, BufKind::Float64, false, "xs_buf") ||
        !ld_buf.acquire_1d_optional(ld_obj, BufKind::Float64, true, "out_ld") ||
        !li_buf.acquire_1d_optional(li_obj, BufKind::Int64, true, "out_li") ||
        !ev_buf.acquire_1d_optional(ev_obj, BufKind::Float64, true, "out_ev") ||
        !al_buf.acquire_1d_optional(al_obj, BufKind::Int32, true, "out_al")) {
        return nullptr;
    }

    const int64_t count = xs_buf.length;
    if (!check_min_len(ld_buf, count, "out_ld") ||
        !check_min_len(li_buf, count, "out_li") ||
        !check_min_len(ev_buf, count, "out_ev") ||
        !check_min_len(al_buf, count, "out_al")) {
        return nullptr;
    }
    if (count == 0) {
        Py_RETURN_NONE;
    }

    fmp_stream_t* s = self->ptr;
    const double* xs_ptr = xs_buf.cdata<double>();
    double* ld_ptr = ld_buf.data<double>();
    int64_t* li_ptr = li_buf.data<int64_t>();
    double* ev_ptr = ev_buf.data<double>();
    int32_t* al_ptr = al_buf.data<int32_t>();

    int rc = 0;
    self->in_use = 1;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_stream_update_batch(s, xs_ptr, count, ld_ptr, li_ptr, ev_ptr, al_ptr);
    Py_END_ALLOW_THREADS
    self->in_use = 0;

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_stream_update_batch failed with error code %d", rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

// snapshot(relative_indices, out_mp, out_idx, out_lmp=None, out_lidx=None) -> active_len
static PyObject* FastStream_snapshot(FastStreamObject* self, PyObject* args, PyObject* kwargs) {
    static char k_rel[] = "relative_indices";
    static char k_mp[] = "out_mp";
    static char k_idx[] = "out_idx";
    static char k_lmp[] = "out_lmp";
    static char k_lidx[] = "out_lidx";
    static char* kwlist[] = {k_rel, k_mp, k_idx, k_lmp, k_lidx, nullptr};

    PyObject* rel_obj = nullptr;
    PyObject* mp_obj = nullptr;
    PyObject* idx_obj = nullptr;
    PyObject* lmp_obj = Py_None;
    PyObject* lidx_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO|OO:snapshot", kwlist,
            &rel_obj, &mp_obj, &idx_obj, &lmp_obj, &lidx_obj)) {
        return nullptr;
    }
    if (!stream_require_ready(self)) {
        return nullptr;
    }

    int rel = PyObject_IsTrue(rel_obj);
    if (rel < 0) {
        return nullptr;
    }

    BufferGuard mp_buf, idx_buf, lmp_buf, lidx_buf;
    if (!mp_buf.acquire_1d(mp_obj, BufKind::Float64, true, "out_mp") ||
        !idx_buf.acquire_1d(idx_obj, BufKind::Int64, true, "out_idx") ||
        !lmp_buf.acquire_1d_optional(lmp_obj, BufKind::Float64, true, "out_lmp") ||
        !lidx_buf.acquire_1d_optional(lidx_obj, BufKind::Int64, true, "out_lidx")) {
        return nullptr;
    }

    const int64_t active = fmp_stream_active_len(self->ptr);
    if (!check_min_len(mp_buf, active, "out_mp") ||
        !check_min_len(idx_buf, active, "out_idx") ||
        !check_min_len(lmp_buf, active, "out_lmp") ||
        !check_min_len(lidx_buf, active, "out_lidx")) {
        return nullptr;
    }
    if (active == 0) {
        return PyLong_FromLongLong(0);
    }

    const fmp_stream_t* s = self->ptr;
    double* mp_ptr = mp_buf.data<double>();
    int64_t* idx_ptr = idx_buf.data<int64_t>();
    double* lmp_ptr = lmp_buf.data<double>();
    int64_t* lidx_ptr = lidx_buf.data<int64_t>();

    int rc = 0;
    self->in_use = 1;
    Py_BEGIN_ALLOW_THREADS
    rc = fmp_stream_snapshot(s, rel, mp_ptr, idx_ptr, lmp_ptr, lidx_ptr);
    Py_END_ALLOW_THREADS
    self->in_use = 0;

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError, "fmp_stream_snapshot failed with error code %d", rc);
        return nullptr;
    }
    return PyLong_FromLongLong(active);
}

static PyObject* FastStream_active_len(FastStreamObject* self, PyObject* /*ignored*/) {
    if (!stream_require_ready(self)) {
        return nullptr;
    }
    return PyLong_FromLongLong(fmp_stream_active_len(self->ptr));
}

static PyObject* FastStream_gid_min(FastStreamObject* self, PyObject* /*ignored*/) {
    if (!stream_require_ready(self)) {
        return nullptr;
    }
    return PyLong_FromLongLong(fmp_stream_gid_min(self->ptr));
}

static PyObject* FastStream_close(FastStreamObject* self, PyObject* /*ignored*/) {
    if (self->in_use) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot close FastStream while concurrently in use");
        return nullptr;
    }
    if (self->ptr) {
        fmp_stream_destroy(self->ptr);
        self->ptr = nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* FastStream_enter(FastStreamObject* self, PyObject* /*ignored*/) {
    if (!stream_require_ready(self)) {
        return nullptr;
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
}

static PyObject* FastStream_exit(FastStreamObject* self, PyObject* /*args*/) {
    return FastStream_close(self, nullptr);
}

static PyObject* FastStream_get_m(PyObject* obj, void* /*closure*/) {
    return PyLong_FromLongLong(reinterpret_cast<FastStreamObject*>(obj)->m);
}
static PyObject* FastStream_get_window(PyObject* obj, void* /*closure*/) {
    return PyLong_FromLongLong(reinterpret_cast<FastStreamObject*>(obj)->window);
}
static PyObject* FastStream_get_excl_zone(PyObject* obj, void* /*closure*/) {
    return PyLong_FromLongLong(reinterpret_cast<FastStreamObject*>(obj)->excl_zone);
}
static PyObject* FastStream_get_alpha(PyObject* obj, void* /*closure*/) {
    return PyFloat_FromDouble(reinterpret_cast<FastStreamObject*>(obj)->alpha);
}
static PyObject* FastStream_get_egress(PyObject* obj, void* /*closure*/) {
    return PyBool_FromLong(reinterpret_cast<FastStreamObject*>(obj)->egress);
}
static PyObject* FastStream_get_track_full_mp(PyObject* obj, void* /*closure*/) {
    return PyBool_FromLong(reinterpret_cast<FastStreamObject*>(obj)->track_full_mp);
}
static PyObject* FastStream_get_closed(PyObject* obj, void* /*closure*/) {
    return PyBool_FromLong(reinterpret_cast<FastStreamObject*>(obj)->ptr == nullptr ? 1 : 0);
}

static PyGetSetDef FastStream_getset[] = {
    {"m", FastStream_get_m, nullptr, "Subsequence length m.", nullptr},
    {"window", FastStream_get_window, nullptr, "Sliding window capacity in subsequences.", nullptr},
    {"excl_zone", FastStream_get_excl_zone, nullptr, "Resolved exclusion zone.", nullptr},
    {"alpha", FastStream_get_alpha, nullptr, "E-process significance level.", nullptr},
    {"egress", FastStream_get_egress, nullptr, "Whether sliding-window egress is enabled.", nullptr},
    {"track_full_mp", FastStream_get_track_full_mp, nullptr, "Whether full-window right-MP is tracked.", nullptr},
    {"closed", FastStream_get_closed, nullptr, "True if native stream resources have been released.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

static PyMethodDef FastStream_methods[] = {
    {"init", reinterpret_cast<PyCFunction>(FastStream_init_data), METH_O,
     "Ingest an initial contiguous aligned float64 buffer (releases GIL)."},
    {"update", reinterpret_cast<PyCFunction>(FastStream_update), METH_O,
     "Ingest one float tick and return 8-tuple (gid, left_dist, left_idx, discord_z, p_value, e_value, is_alert, n_repaired)."},
    {"update_batch", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(FastStream_update_batch)),
     METH_VARARGS | METH_KEYWORDS,
     "Ingest a batch float64 buffer into preallocated output buffers (releases GIL)."},
    {"snapshot", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(FastStream_snapshot)),
     METH_VARARGS | METH_KEYWORDS,
     "Copy active window Full-MP and optional Left-MP into preallocated output buffers (releases GIL)."},
    {"active_len", reinterpret_cast<PyCFunction>(FastStream_active_len), METH_NOARGS,
     "Return number of active subsequences currently in the stream."},
    {"gid_min", reinterpret_cast<PyCFunction>(FastStream_gid_min), METH_NOARGS,
     "Return minimum active global subsequence ID."},
    {"close", reinterpret_cast<PyCFunction>(FastStream_close), METH_NOARGS,
     "Release native C++ stream resources (idempotent)."},
    {"__enter__", reinterpret_cast<PyCFunction>(FastStream_enter), METH_NOARGS,
     "Enter context manager."},
    {"__exit__", reinterpret_cast<PyCFunction>(FastStream_exit), METH_VARARGS,
     "Exit context manager and close native stream."},
    {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject FastStreamType;

static void init_fast_stream_type() {
    FastStreamType = {};
    FastStreamType.ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
    FastStreamType.tp_name = "flashmp._fast.FastStream";
    FastStreamType.tp_doc = "Zero-overhead C-API wrapper around RingMP native fmp_stream_t.";
    FastStreamType.tp_basicsize = sizeof(FastStreamObject);
    FastStreamType.tp_itemsize = 0;
    FastStreamType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    FastStreamType.tp_new = PyType_GenericNew;
    FastStreamType.tp_init = reinterpret_cast<initproc>(FastStream_init);
    FastStreamType.tp_dealloc = reinterpret_cast<destructor>(FastStream_dealloc);
    FastStreamType.tp_methods = FastStream_methods;
    FastStreamType.tp_getset = FastStream_getset;
}

static PyMethodDef module_methods[] = {
    {"version", py_version, METH_NOARGS, "Return native libflashmp version string."},
    {"simd_isa", py_simd_isa, METH_NOARGS, "Return compiled SIMD instruction set identifier."},
    {"fast_self_join", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(py_fast_self_join)),
     METH_VARARGS | METH_KEYWORDS,
     "Zero-copy self-join into preallocated writable buffers (releases GIL)."},
    {"fast_ab_join", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(py_fast_ab_join)),
     METH_VARARGS | METH_KEYWORDS,
     "Zero-copy AB-join into preallocated writable buffers (releases GIL)."},
    {"fast_leak_scan", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(py_fast_leak_scan)),
     METH_VARARGS | METH_KEYWORDS,
     "Zero-copy affine-invariant leakage scan into preallocated buffers (releases GIL); returns hit_count."},
    {"fast_mstump", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(py_fast_mstump)),
     METH_VARARGS | METH_KEYWORDS,
     "Zero-copy multivariate mSTUMP into preallocated 2D/flat buffers (releases GIL)."},
    {"fast_floss_cac", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(py_fast_floss_cac)),
     METH_VARARGS | METH_KEYWORDS,
     "Zero-copy O(n) FLOSS Corrected Arc Curve into preallocated float64 buffer (releases GIL)."},
    {"fast_auto_window", reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)(void)>(py_fast_auto_window)),
     METH_VARARGS | METH_KEYWORDS,
     "Zero-copy causal autocorrelation window selector (releases GIL)."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef fast_module = {
    PyModuleDef_HEAD_INIT,
    "_fast",
    "Zero-copy PEP 3118 CPython C-API bindings for flashmp (GIL-releasing batch & sub-microsecond streaming).",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

} // namespace

PyMODINIT_FUNC PyInit__fast(void) {
    init_fast_stream_type();
    if (PyType_Ready(&FastStreamType) < 0) {
        return nullptr;
    }

    PyObject* mod = PyModule_Create(&fast_module);
    if (!mod) {
        return nullptr;
    }

    Py_INCREF(&FastStreamType);
    if (PyModule_AddObject(mod, "FastStream", reinterpret_cast<PyObject*>(&FastStreamType)) < 0) {
        Py_DECREF(&FastStreamType);
        Py_DECREF(mod);
        return nullptr;
    }
    return mod;
}
