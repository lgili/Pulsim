#pragma once

// =============================================================================
// Pulsim — kernel side of the external device ABI (add-external-device-abi)
// =============================================================================
//
// Loads a model library, validates its descriptor, CHECKS its mathematical
// contract at registration (the model is untrusted as far as correctness
// goes) and evaluates it during a run with a finiteness check that costs one
// comparison per call. Every refusal names the device, the property and the
// numbers. Not stable API: the stable contract is pulsim_xdev.h.

#include "pulsim/device/device.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/xdev/pulsim_xdev.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <type_traits>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pulsim::xdev {

/// A refused external model or instance (a std::invalid_argument, so the
/// bindings raise ValueError).
struct ContractError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// ---- the library ------------------------------------------------------------
/// One loaded library; closed when the last model of it is released.
class Library {
public:
    explicit Library(const std::string& given) : path_(absolute_file_(given)) {
        const std::string& path = path_;
#if defined(_WIN32)
        // UTF-8 path → wide (MultiByteToWideChar), so non-ASCII paths load
        const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring w(n > 0 ? static_cast<std::size_t>(n) : 0u, L'\0');
        if (n > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
        // the model's folder and the system folders only: no CWD / PATH planting
        handle_ = reinterpret_cast<void*>(LoadLibraryExW(
            w.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
        if (!handle_) {
            throw ContractError("external device library '" + path
                                + "': LoadLibrary failed (error "
                                + std::to_string(GetLastError()) + ")");
        }
        get_ = reinterpret_cast<pulsim_xdev_get_fn>(reinterpret_cast<void*>(
            GetProcAddress(reinterpret_cast<HMODULE>(handle_), PULSIM_XDEV_ENTRY)));
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            const char* e = dlerror();
            throw ContractError("external device library '" + path + "': "
                                + (e ? e : "dlopen failed"));
        }
        get_ = reinterpret_cast<pulsim_xdev_get_fn>(dlsym(handle_, PULSIM_XDEV_ENTRY));
#endif
        if (!get_) {
            close_();
            throw ContractError("external device library '" + path
                                + "' exports no '" PULSIM_XDEV_ENTRY
                                "': not a Pulsim external device library");
        }
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    ~Library() { close_(); }

    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] const pulsim_xdev_descriptor* get(std::uint32_t i) const noexcept {
        return get_(i);
    }

private:
    /// The library is loaded by absolute path of an existing regular file
    /// only: a bare name would be searched for (LD_LIBRARY_PATH, PATH, CWD).
    static std::string absolute_file_(const std::string& given) {
        std::error_code ec;
        const std::filesystem::path p(std::u8string(given.begin(), given.end()));   // UTF-8 on every platform
        if (!std::filesystem::is_regular_file(p, ec)) {
            throw ContractError("external device library '" + given
                                + "': not an existing file (a library is loaded only "
                                  "from the path you name, never searched for)");
        }
        const auto a = std::filesystem::absolute(p, ec);
        if (ec) return given;
        const auto u = a.u8string();
        return std::string(u.begin(), u.end());
    }
    void close_() noexcept {
        if (!handle_) return;
#if defined(_WIN32)
        FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
        handle_ = nullptr;
    }
    std::string path_;
    void* handle_ = nullptr;
    pulsim_xdev_get_fn get_ = nullptr;
};

// ---- the model (a validated descriptor) ------------------------------------
/// The version-1 prefix of the descriptor: a library's struct must be at
/// least this long. Appended fields are read only when struct_size covers them.
inline constexpr std::size_t kV1DescriptorSize = PULSIM_XDEV_DESCRIPTOR_V1_SIZE;
static_assert(sizeof(pulsim_xdev_descriptor) == kV1DescriptorSize,
              "an appended descriptor field must be read behind a struct_size test");

class Model {
public:
    Model(std::shared_ptr<const Library> lib, const pulsim_xdev_descriptor* d)
        : lib_(std::move(lib)), d_(d) {
        validate_descriptor_();
        own_();
    }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    [[nodiscard]] const pulsim_xdev_descriptor& d() const noexcept { return *d_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] std::size_t rows() const noexcept { return R_; }
    [[nodiscard]] std::string library_path() const { return lib_ ? lib_->path() : "<in-process>"; }

    /// The model called `name` in `lib`, refused by name if absent.
    static std::shared_ptr<const Model> find(const std::shared_ptr<const Library>& lib,
                                             const std::string& name) {
        std::string seen;
        for (std::uint32_t i = 0; i < 1024; ++i) {
            const auto* d = lib->get(i);
            if (!d) break;
            const std::string n = d->name ? d->name : "";
            if (n == name) return std::make_shared<const Model>(lib, d);
            seen += (seen.empty() ? "" : ", ") + n;
        }
        throw ContractError("external device library '" + lib->path()
                            + "' has no model '" + name + "' (it has: "
                            + (seen.empty() ? "none" : seen) + ")");
    }

private:
    [[noreturn]] void refuse_(const std::string& why) const {
        throw ContractError("external model '" + name_ + "' (" + library_path()
                            + "): " + why);
    }
    void validate_descriptor_() {
        if (!d_) throw ContractError("external device: null descriptor");
        name_ = d_->name ? d_->name : "<unnamed>";
        if (d_->abi_version != PULSIM_XDEV_ABI_VERSION) {
            refuse_("built for external device ABI version "
                    + std::to_string(d_->abi_version) + ", this Pulsim speaks version "
                    + std::to_string(PULSIM_XDEV_ABI_VERSION) + "; rebuild the model");
        }
        if (d_->struct_size < kV1DescriptorSize) {
            refuse_("descriptor struct_size " + std::to_string(d_->struct_size)
                    + " is smaller than the version-1 descriptor ("
                    + std::to_string(kV1DescriptorSize) + " bytes)");
        }
        if (!d_->name) refuse_("descriptor has no name");
        if (!d_->eval) refuse_("descriptor has no eval (required)");
        for (const auto r : d_->reserved)
            if (r) refuse_("a reserved descriptor slot is not NULL (built for a later ABI?)");
        R_ = std::size_t{d_->n_pins} + d_->n_internal;
        if (d_->n_pins < 1 || R_ > 64) {
            refuse_("n_pins = " + std::to_string(d_->n_pins) + ", n_internal = "
                    + std::to_string(d_->n_internal) + ": need 1 <= n_pins, rows <= 64");
        }
        if (d_->n_state > 4096) refuse_("n_state = " + std::to_string(d_->n_state) + " > 4096");
        if (d_->n_state > 0 && !d_->accept) {
            refuse_("declares n_state = " + std::to_string(d_->n_state)
                    + " but has no accept (nothing would ever write the state)");
        }
        if (!d_->pin_names) refuse_("pin_names is NULL");
        for (std::uint32_t k = 0; k < d_->n_pins; ++k)
            if (!d_->pin_names[k]) refuse_("pin_names[" + std::to_string(k) + "] is NULL");
        if (d_->n_fields > 0 && !d_->fields) refuse_("n_fields > 0 but fields is NULL");
        for (std::uint32_t k = 0; k < d_->n_fields; ++k) {
            const auto& f = d_->fields[k];
            if (!f.name || !f.unit) refuse_("field #" + std::to_string(k) + " has a NULL name or unit");
            if (!(f.lo <= f.def && f.def <= f.hi)) {
                refuse_(std::string("field '") + f.name + "': default " + num(f.def)
                        + " outside its bounds [" + num(f.lo) + ", " + num(f.hi) + "]");
            }
        }
        check_pattern_("f", d_->nnz_f, d_->f_rows, d_->f_cols);
        check_pattern_("q", d_->nnz_q, d_->q_rows, d_->q_cols);
        if (!d_->probe_lo || !d_->probe_hi) refuse_("probe_lo / probe_hi is NULL (the check box)");
        for (std::size_t k = 0; k < R_; ++k) {
            if (!(std::isfinite(d_->probe_lo[k]) && std::isfinite(d_->probe_hi[k])
                  && d_->probe_lo[k] < d_->probe_hi[k])) {
                refuse_("probe box row " + std::to_string(k) + " is not a finite lo < hi");
            }
        }
        if (d_->n_monotone > 0 && !d_->monotone) refuse_("n_monotone > 0 but monotone is NULL");
        for (std::uint32_t k = 0; k < d_->n_monotone; ++k) {
            if (d_->monotone[k].row >= R_ || d_->monotone[k].col >= R_) {
                refuse_("monotone claim #" + std::to_string(k) + " indexes past the "
                        + std::to_string(R_) + " rows");
            }
        }
        const std::uint32_t known = PULSIM_XDEV_CLAIM_SMOOTH | PULSIM_XDEV_CLAIM_PASSIVE
                                    | PULSIM_XDEV_CLAIM_RECIPROCAL;
        if (d_->claims & ~known) refuse_("unknown claim bits " + std::to_string(d_->claims & ~known));
    }
    /// Deep copy of the validated descriptor: the host never reads the
    /// library's (possibly writable) struct or arrays again.
    void own_() {
        desc_ = *d_;
        const std::size_t R = R_;
        for (std::uint32_t k = 0; k < d_->n_pins; ++k) pin_s_.emplace_back(d_->pin_names[k]);
        for (const auto& n : pin_s_) pin_p_.push_back(n.c_str());
        for (std::uint32_t k = 0; k < d_->n_fields; ++k) {
            field_s_.emplace_back(d_->fields[k].name);
            field_s_.emplace_back(d_->fields[k].unit);
            fields_.push_back(d_->fields[k]);
        }
        for (std::size_t k = 0; k < fields_.size(); ++k) {
            fields_[k].name = field_s_[2 * k].c_str();
            fields_[k].unit = field_s_[2 * k + 1].c_str();
        }
        auto cp = [](const auto* a, std::size_t n) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(a)>>;
            return n ? std::vector<T>(a, a + n) : std::vector<T>{};
        };
        fr_ = cp(d_->f_rows, d_->nnz_f);
        fc_ = cp(d_->f_cols, d_->nnz_f);
        qr_ = cp(d_->q_rows, d_->nnz_q);
        qc_ = cp(d_->q_cols, d_->nnz_q);
        lo_ = cp(d_->probe_lo, R);
        hi_ = cp(d_->probe_hi, R);
        mono_ = cp(d_->monotone, d_->n_monotone);
        desc_.struct_size = static_cast<std::uint32_t>(sizeof desc_);
        desc_.name = name_.c_str();
        desc_.pin_names = pin_p_.data();
        desc_.fields = fields_.data();
        desc_.f_rows = fr_.data();
        desc_.f_cols = fc_.data();
        desc_.q_rows = qr_.data();
        desc_.q_cols = qc_.data();
        desc_.probe_lo = lo_.data();
        desc_.probe_hi = hi_.data();
        desc_.monotone = mono_.data();
        d_ = &desc_;
    }
    void check_pattern_(const char* which, std::uint32_t nnz, const std::uint32_t* rr,
                        const std::uint32_t* cc) {
        if (nnz > R_ * R_) refuse_(std::string("nnz_") + which + " exceeds rows^2");
        if (nnz > 0 && (!rr || !cc)) refuse_(std::string(which) + "_rows / _cols is NULL");
        std::vector<char> seen(R_ * R_, 0);
        for (std::uint32_t k = 0; k < nnz; ++k) {
            if (rr[k] >= R_ || cc[k] >= R_) {
                refuse_(std::string("pattern of d") + which + " entry #" + std::to_string(k)
                        + " = (" + std::to_string(rr[k]) + ", " + std::to_string(cc[k])
                        + ") is outside the " + std::to_string(R_) + " rows");
            }
            if (seen[rr[k] * R_ + cc[k]]++) {
                refuse_(std::string("pattern of d") + which + " lists (" + std::to_string(rr[k])
                        + ", " + std::to_string(cc[k]) + ") twice");
            }
        }
    }
    static std::string num(double v) {
        std::ostringstream s;
        s.precision(17);
        s << v;
        return s.str();
    }

    std::shared_ptr<const Library> lib_;
    const pulsim_xdev_descriptor* d_;
    std::string name_;
    std::size_t R_ = 0;
    pulsim_xdev_descriptor desc_{};
    std::vector<std::string> pin_s_, field_s_;
    std::vector<const char*> pin_p_;
    std::vector<pulsim_xdev_field> fields_;
    std::vector<std::uint32_t> fr_, fc_, qr_, qc_;
    std::vector<double> lo_, hi_;
    std::vector<pulsim_xdev_monotone> mono_;
};

[[nodiscard]] inline std::shared_ptr<const Model> load_model(const std::string& path,
                                                            const std::string& name) {
    return Model::find(std::make_shared<const Library>(path), name);
}

// ---- one instance -----------------------------------------------------------
/// Scratch of one evaluation (sized once per instance).
struct EvalOut {
    std::vector<Real> f, q, df, dq;
};

class Instance {
public:
    /// `debug_jacobian`: check the Jacobian against central differences at
    /// EVERY run-time evaluation (the operating points the run visits).
    Instance(std::string device, std::shared_ptr<const Model> m, std::vector<Real> p,
             bool debug_jacobian = false)
        : device_(std::move(device)), m_(std::move(m)), p_(std::move(p)),
          debug_(debug_jacobian) {
        const auto& d = m_->d();
        if (p_.size() != d.n_fields) {
            refuse_("got " + std::to_string(p_.size()) + " parameter values, the model has "
                    + std::to_string(d.n_fields) + " fields");
        }
        for (std::uint32_t k = 0; k < d.n_fields; ++k) {
            const auto& f = d.fields[k];
            if (!(p_[k] >= f.lo && p_[k] <= f.hi)) {
                refuse_(std::string("parameter ") + f.name + " = " + num(p_[k]) + " " + f.unit
                        + " is outside its declared bounds [" + num(f.lo) + ", " + num(f.hi) + "]");
            }
        }
        if (d.validate) {
            char msg[256] = {0};
            if (d.validate(p_.data(), msg, sizeof msg) != PULSIM_XDEV_OK) {
                msg[sizeof msg - 1] = '\0';
                refuse_(std::string("the model's validate refused the parameters: ") + msg);
            }
        }
        if (d.create && d.create(p_.data(), &inst_) != PULSIM_XDEV_OK) {
            if (inst_ && d.destroy) d.destroy(inst_);
            inst_ = nullptr;
            refuse_("the model's create failed");
        }
    }
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    ~Instance() {
        if (inst_ && m_->d().destroy) m_->d().destroy(inst_);
    }

    [[nodiscard]] const Model& model() const noexcept { return *m_; }
    [[nodiscard]] const std::shared_ptr<const Model>& model_ptr() const noexcept { return m_; }
    [[nodiscard]] const std::string& device() const noexcept { return device_; }
    [[nodiscard]] const std::vector<Real>& params() const noexcept { return p_; }
    [[nodiscard]] std::size_t rows() const noexcept { return m_->rows(); }
    [[nodiscard]] std::size_t n_state() const noexcept { return m_->d().n_state; }
    [[nodiscard]] bool debug_jacobian() const noexcept { return debug_; }

    void size(EvalOut& o) const {
        const auto& d = m_->d();
        o.f.assign(rows(), 0);
        o.q.assign(rows(), 0);
        o.df.assign(d.nnz_f, 0);
        o.dq.assign(d.nnz_q, 0);
    }

    /// One evaluation at x with the committed model state. The non-finite
    /// check is ONE comparison (the sum of every output); only a failure
    /// scans to name the entry.
    void eval(const Real* x, const Real* state, EvalOut& o) const {
        const auto& d = m_->d();
        const pulsim_xdev_ctx c = ctx_(state);
        poison_(o);   // an output the model leaves unwritten reads NaN, not the last point's value
        const int32_t st = d.eval(&c, x, o.f.data(), o.q.data(), o.df.data(), o.dq.data());
        if (st != PULSIM_XDEV_OK) {
            refuse_("eval returned status " + std::to_string(st) + " at x = " + vec(x, rows()));
        }
        Real sum = 0;
        for (Real v : o.f) sum += v;
        for (Real v : o.q) sum += v;
        for (Real v : o.df) sum += v;
        for (Real v : o.dq) sum += v;
        if (!std::isfinite(sum)) name_nonfinite_(x, o);
        if (debug_) check_jacobian_at(x, state, o, "run-time debug check");
    }

    /// Jacobian of `o` (evaluated at x) against Richardson-extrapolated
    /// central differences of f and q; refuses naming the worst entry.
    void check_jacobian_at(const Real* x, const Real* state, const EvalOut& o,
                           const char* where) const {
        const std::size_t R = rows();
        const auto& d = m_->d();
        const auto Jf = dense_(o.df, d.f_rows, d.f_cols, d.nnz_f);
        const auto Jq = dense_(o.dq, d.q_rows, d.q_cols, d.nnz_q);
        std::vector<Real> xp(x, x + R);
        EvalOut a, b;
        size(a);
        size(b);
        for (std::size_t c = 0; c < R; ++c) {
            const Real h = Real{1e-4} * std::max(Real{1e-3}, std::abs(x[c]));
            auto diff = [&](Real hh, std::vector<Real>& df_, std::vector<Real>& dq_) {
                xp[c] = x[c] + hh;
                raw_eval_(xp.data(), state, a);
                xp[c] = x[c] - hh;
                raw_eval_(xp.data(), state, b);
                xp[c] = x[c];
                for (std::size_t r = 0; r < R; ++r) {
                    df_[r] = (a.f[r] - b.f[r]) / (2 * hh);
                    dq_[r] = (a.q[r] - b.q[r]) / (2 * hh);
                }
            };
            std::vector<Real> f1(R), q1(R), f2(R), q2(R);
            diff(h, f1, q1);
            diff(h / 2, f2, q2);
            for (std::size_t r = 0; r < R; ++r) {
                compare_(where, "f", r, c, Jf[r * R + c], (4 * f2[r] - f1[r]) / 3,
                         row_scale_(Jf, r), x);
                compare_(where, "q", r, c, Jq[r * R + c], (4 * q2[r] - q1[r]) / 3,
                         row_scale_(Jq, r), x);
            }
        }
    }

    /// Dense R×R of a pattern-ordered value list.
    [[nodiscard]] std::vector<Real> dense_(const std::vector<Real>& v, const std::uint32_t* rr,
                                           const std::uint32_t* cc, std::uint32_t nnz) const {
        const std::size_t R = rows();
        std::vector<Real> J(R * R, 0);
        for (std::uint32_t k = 0; k < nnz; ++k) J[rr[k] * R + cc[k]] = v[k];
        return J;
    }

    [[noreturn]] void refuse_(const std::string& why) const {
        throw ContractError("external device '" + device_ + "' (model '" + m_->name()
                            + "'): " + why);
    }
    static std::string num(double v) {
        std::ostringstream s;
        s.precision(17);
        s << v;
        return s.str();
    }
    static std::string vec(const Real* x, std::size_t n) {
        std::string s = "[";
        for (std::size_t k = 0; k < n; ++k) s += (k ? ", " : "") + num(x[k]);
        return s + "]";
    }

    [[nodiscard]] pulsim_xdev_ctx ctx_(const Real* state) const noexcept {
        pulsim_xdev_ctx c{};
        c.struct_size = static_cast<std::uint32_t>(sizeof(pulsim_xdev_ctx));
        c.t = 0;
        c.p = p_.data();
        c.state = m_->d().n_state ? state : nullptr;
        c.instance = inst_;
        return c;
    }
    static void poison_(EvalOut& o) noexcept {
        constexpr Real nan = std::numeric_limits<Real>::quiet_NaN();
        std::fill(o.f.begin(), o.f.end(), nan);
        std::fill(o.q.begin(), o.q.end(), nan);
        std::fill(o.df.begin(), o.df.end(), nan);
        std::fill(o.dq.begin(), o.dq.end(), nan);
    }
    void raw_eval_(const Real* x, const Real* state, EvalOut& o) const {
        const pulsim_xdev_ctx c = ctx_(state);
        poison_(o);
        if (m_->d().eval(&c, x, o.f.data(), o.q.data(), o.df.data(), o.dq.data())
            != PULSIM_XDEV_OK) {
            refuse_("eval returned an error status at x = " + vec(x, rows()));
        }
    }

private:
    static Real row_scale_(const std::vector<Real>& J, std::size_t r) {
        const std::size_t R = static_cast<std::size_t>(std::llround(std::sqrt(J.size())));
        Real s = 0;
        for (std::size_t k = 0; k < R; ++k) s = std::max(s, std::abs(J[r * R + k]));
        return s;
    }
    void compare_(const char* where, const char* which, std::size_t r, std::size_t c, Real declared,
                  Real fd, Real scale, const Real* x) const {
        // 1e-5 of the row scale (design.md), with a floor for an all-zero row
        const Real tol = Real{1e-5} * std::max(scale, std::abs(fd)) + Real{1e-12};
        if (!(std::abs(declared - fd) <= tol)) {
            refuse_(std::string(where) + ": declared d" + which + "[" + std::to_string(r)
                    + "]/dx[" + std::to_string(c) + "] = " + num(declared)
                    + " but central differences give " + num(fd) + " at x = " + vec(x, rows())
                    + " (tolerance " + num(tol) + ")"
                    + (declared == 0 ? "; the entry may be missing from the declared pattern" : ""));
        }
    }
    void name_nonfinite_(const Real* x, const EvalOut& o) const {
        auto scan = [&](const char* w, const std::vector<Real>& v) {
            for (std::size_t k = 0; k < v.size(); ++k) {
                if (!std::isfinite(v[k])) {
                    refuse_(std::string("eval returned a non-finite ") + w + "["
                            + std::to_string(k) + "] = " + num(v[k]) + " at x = "
                            + vec(x, rows()));
                }
            }
        };
        scan("f", o.f);
        scan("q", o.q);
        scan("df", o.df);
        scan("dq", o.dq);
        refuse_("eval outputs overflow when summed at x = " + vec(x, rows()));
    }

    std::string device_;
    std::shared_ptr<const Model> m_;
    std::vector<Real> p_;
    void* inst_ = nullptr;
    bool debug_ = false;
};

/// Parameter values in field order from (name, value, unit) triples: an
/// unknown name, a repeated name or a unit that is not the field's declared
/// unit is refused by name; an omitted unit is not checked; missing fields
/// take their defaults. Bounds are checked by `Instance`.
struct NamedParam {
    std::string name;
    Real value;
    std::string unit;   ///< empty: not given
    bool has_unit = false;
};
[[nodiscard]] inline std::vector<Real> resolve_params(const Model& m, const std::string& device,
                                                     const std::vector<NamedParam>& given) {
    const auto& d = m.d();
    std::vector<Real> p(d.n_fields);
    std::vector<char> set(d.n_fields, 0);
    for (std::uint32_t k = 0; k < d.n_fields; ++k) p[k] = d.fields[k].def;
    for (const auto& g : given) {
        std::uint32_t k = 0;
        while (k < d.n_fields && g.name != d.fields[k].name) ++k;
        if (k == d.n_fields) {
            std::string known;
            for (std::uint32_t j = 0; j < d.n_fields; ++j)
                known += (j ? ", " : "") + std::string(d.fields[j].name);
            throw ContractError("external device '" + device + "' (model '" + m.name()
                                + "'): no parameter '" + g.name + "' (the model has: " + known + ")");
        }
        if (set[k]++) {
            throw ContractError("external device '" + device + "': parameter '" + g.name
                                + "' given twice");
        }
        if (g.has_unit && g.unit != d.fields[k].unit) {
            throw ContractError("external device '" + device + "': parameter '" + g.name
                                + "' given in '" + g.unit + "', the model declares '"
                                + d.fields[k].unit + "' (no unit conversion is done)");
        }
        p[k] = g.value;
    }
    return p;
}

}  // namespace pulsim::xdev
