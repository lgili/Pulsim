#include "pulsim/api/grpc/session_manager.hpp"

#include "pulsim/types.hpp"

#include <algorithm>
#include <atomic>
#include <random>

namespace pulsim::api::grpc {
namespace {

std::string generate_id(std::size_t length = 16) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    static constexpr char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<std::size_t> dist(0, sizeof(alphabet) - 2);

    std::string out;
    out.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        out.push_back(alphabet[dist(rng)]);
    }
    return out;
}

std::vector<int> resolve_indices(const std::vector<std::string>& all_signals,
                                 const std::vector<std::string>& requested) {
    if (requested.empty()) {
        std::vector<int> indices(all_signals.size());
        for (std::size_t i = 0; i < all_signals.size(); ++i) {
            indices[i] = static_cast<int>(i);
        }
        return indices;
    }

    std::vector<int> indices;
    indices.reserve(requested.size());
    for (const auto& name : requested) {
        auto it = std::find(all_signals.begin(), all_signals.end(), name);
        if (it != all_signals.end()) {
            indices.push_back(static_cast<int>(std::distance(all_signals.begin(), it)));
        }
    }
    return indices;
}

std::vector<std::string> compute_signal_names(const pulsim::Circuit& circuit) {
    std::vector<std::string> names;
    const pulsim::Index count = circuit.total_variables();
    names.reserve(static_cast<std::size_t>(count));
    for (pulsim::Index i = 0; i < count; ++i) {
        names.push_back(circuit.signal_name(i));
    }
    return names;
}

}  // namespace

class SimulationControlImpl : public pulsim::SimulationControl {
public:
    bool should_stop() const override { return stop_.load(std::memory_order_acquire); }
    bool should_pause() const override { return pause_.load(std::memory_order_acquire); }

    void wait_until_resumed() override {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return (!pause_.load(std::memory_order_acquire)) || stop_.load(std::memory_order_acquire);
        });
    }

    void request_stop() {
        stop_.store(true, std::memory_order_release);
        resume();
    }

    void request_pause() { pause_.store(true, std::memory_order_release); }

    void resume() {
        pause_.store(false, std::memory_order_release);
        cv_.notify_all();
    }

private:
    std::atomic<bool> stop_{false};
    std::atomic<bool> pause_{false};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

WaveformSubscriber::WaveformSubscriber(WaveformHeaderInfo header,
                                       std::vector<int> indices,
                                       uint32_t decimation,
                                       std::optional<double> start_time)
    : header_(std::move(header))
    , indices_(std::move(indices))
    , decimation_(decimation == 0 ? 1 : decimation)
    , start_time_(start_time) {}

void WaveformSubscriber::enqueue_header_locked() {
    if (header_sent_) {
        return;
    }
    Event event;
    event.type = Event::Type::Header;
    event.header = header_;
    queue_.push_back(std::move(event));
    header_sent_ = true;
}

void WaveformSubscriber::push_sample(double time, const pulsim::Vector& state) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }

    if (!header_sent_) {
        enqueue_header_locked();
    }

    if (start_time_.has_value() && time < *start_time_) {
        return;
    }

    ++sample_counter_;
    if ((sample_counter_ % decimation_) != 0) {
        return;
    }

    Event event;
    event.type = Event::Type::Sample;
    event.sample.time = time;
    event.sample.values.reserve(indices_.size());
    for (int idx : indices_) {
        if (idx >= 0 && idx < state.size()) {
            event.sample.values.push_back(state(idx));
        } else {
            event.sample.values.push_back(0.0);
        }
    }

    queue_.push_back(std::move(event));
    cv_.notify_all();
}

void WaveformSubscriber::push_complete(const WaveformCompleteInfo& info) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    if (!header_sent_) {
        enqueue_header_locked();
    }

    Event event;
    event.type = Event::Type::Complete;
    event.complete = info;
    queue_.push_back(std::move(event));
    closed_ = true;
    cv_.notify_all();
}

bool WaveformSubscriber::next_event(Event& event) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return !queue_.empty(); });
    event = std::move(queue_.front());
    queue_.pop_front();
    return true;
}

struct SessionManager::SessionRuntime {
    SessionInfo info;
    pulsim::Circuit circuit;
    pulsim::SimulationOptions options;
    std::vector<std::string> signal_names;
    std::vector<std::weak_ptr<WaveformSubscriber>> subscribers;
    std::mutex subscribers_mutex;
    std::unique_ptr<std::thread> worker;
    std::shared_ptr<SimulationControlImpl> control;
    ResultSnapshot result;
    bool has_result = false;
};

SessionManager::SessionManager(const ServerConfig& config)
    : config_(config) {}

SessionManager::~SessionManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, runtime] : sessions_) {
        if (runtime->control) {
            runtime->control->request_stop();
        }
        if (runtime->worker && runtime->worker->joinable()) {
            runtime->worker->join();
        }
    }
}

std::string SessionManager::create_session(const std::string& name,
                                           const std::string& owner,
                                           const std::optional<std::string>& model_id,
                                           pulsim::Circuit circuit,
                                           pulsim::SimulationOptions options,
                                           const std::vector<std::string>& signals) {
    auto runtime = std::make_unique<SessionRuntime>();
    runtime->info.session_id = generate_id();
    runtime->info.model_id = model_id.value_or("");
    runtime->info.name = name.empty() ? ("session-" + runtime->info.session_id) : name;
    runtime->info.owner = owner;
    runtime->info.state = SessionState::Ready;
    runtime->info.created_at = std::chrono::system_clock::now();
    runtime->info.updated_at = runtime->info.created_at;
    runtime->info.expiry = runtime->info.created_at + config_.session_retention;
    runtime->info.active_signals = signals;

    runtime->circuit = std::move(circuit);
    runtime->options = options;
    runtime->signal_names = compute_signal_names(runtime->circuit);

    const auto session_id = runtime->info.session_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_.emplace(session_id, std::move(runtime));
    }
    return session_id;
}

std::vector<SessionInfo> SessionManager::list_sessions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SessionInfo> out;
    out.reserve(sessions_.size());
    for (const auto& [id, runtime] : sessions_) {
        out.push_back(to_info(*runtime));
    }
    return out;
}

std::optional<SessionInfo> SessionManager::get_session(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return std::nullopt;
    }
    return to_info(*it->second);
}

bool SessionManager::start_session(const std::string& session_id,
                                   const pulsim::SimulationOptions& overrides) {
    SessionRuntime* runtime = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = find_runtime(session_id);
        if (!runtime) {
            return false;
        }
        if (runtime->info.state == SessionState::Running) {
            return true;
        }
        if (runtime->info.state == SessionState::Completed || runtime->info.state == SessionState::Failed) {
            return false;
        }

        runtime->options = overrides;
        runtime->info.state = SessionState::Running;
        runtime->info.updated_at = std::chrono::system_clock::now();
        runtime->control = std::make_shared<SimulationControlImpl>();
        runtime->subscribers.clear();
        runtime->has_result = false;
        runtime->result = ResultSnapshot{};
    }

    runtime->worker = std::make_unique<std::thread>([this, runtime]() {
        pulsim::Simulator simulator(runtime->circuit, runtime->options);

        auto broadcast_sample = [runtime](pulsim::Real time, const pulsim::Vector& state) {
            std::vector<std::shared_ptr<WaveformSubscriber>> subs;
            {
                std::lock_guard<std::mutex> lock(runtime->subscribers_mutex);
                for (auto it = runtime->subscribers.begin(); it != runtime->subscribers.end();) {
                    if (auto sub = it->lock()) {
                        subs.push_back(sub);
                        ++it;
                    } else {
                        it = runtime->subscribers.erase(it);
                    }
                }
            }
            for (auto& sub : subs) {
                sub->push_sample(time, state);
            }
        };

        auto result = simulator.run_transient(broadcast_sample, nullptr, runtime->control.get());

        ResultSnapshot snapshot;
        snapshot.result = result;
        snapshot.final_state = (result.final_status == pulsim::SolverStatus::Success)
            ? SessionState::Completed
            : SessionState::Failed;
        snapshot.error_message = result.error_message;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            runtime->result = snapshot;
            runtime->has_result = true;
            runtime->info.state = snapshot.final_state;
            runtime->info.error_message = snapshot.error_message;
            runtime->info.updated_at = std::chrono::system_clock::now();
            runtime->info.expiry = runtime->info.updated_at + config_.session_retention;
        }

        WaveformCompleteInfo complete;
        complete.final_state = snapshot.final_state;
        complete.error_message = snapshot.error_message;

        std::vector<std::shared_ptr<WaveformSubscriber>> subs;
        {
            std::lock_guard<std::mutex> lock(runtime->subscribers_mutex);
            for (auto it = runtime->subscribers.begin(); it != runtime->subscribers.end();) {
                if (auto sub = it->lock()) {
                    subs.push_back(sub);
                    ++it;
                } else {
                    it = runtime->subscribers.erase(it);
                }
            }
            runtime->subscribers.clear();
        }

        for (auto& sub : subs) {
            sub->push_complete(complete);
        }
    });
    runtime->worker->detach();
    runtime->worker.reset();
    return true;
}

std::shared_ptr<WaveformSubscriber> SessionManager::attach_stream(const std::string& session_id,
                                                                  const std::vector<std::string>& signals,
                                                                  uint32_t decimation,
                                                                  std::optional<double> start_time) {
    SessionRuntime* runtime = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runtime = find_runtime(session_id);
        if (!runtime) {
            return nullptr;
        }
    }

    const auto indices = resolve_indices(runtime->signal_names, signals);
    if (indices.empty()) {
        return nullptr;
    }

    WaveformHeaderInfo header;
    header.session_id = session_id;
    header.signals.reserve(indices.size());
    for (int idx : indices) {
        if (idx >= 0 && idx < static_cast<int>(runtime->signal_names.size())) {
            header.signals.push_back(runtime->signal_names[static_cast<std::size_t>(idx)]);
        }
    }
    header.tstart = runtime->options.tstart;
    header.tstop = runtime->options.tstop;
    header.total_samples = runtime->has_result ? runtime->result.result.time.size() : 0;

    auto subscriber = std::make_shared<WaveformSubscriber>(header, indices, decimation, start_time);

    if (runtime->has_result) {
        const auto& res = runtime->result.result;
        for (std::size_t i = 0; i < res.time.size(); ++i) {
            subscriber->push_sample(res.time[i], res.data[i]);
        }
        WaveformCompleteInfo complete;
        complete.final_state = runtime->result.final_state;
        complete.error_message = runtime->result.error_message;
        subscriber->push_complete(complete);
    } else {
        std::lock_guard<std::mutex> lock(runtime->subscribers_mutex);
        runtime->subscribers.emplace_back(subscriber);
    }

    return subscriber;
}

bool SessionManager::pause_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* runtime = find_runtime(session_id);
    if (!runtime || !runtime->control) {
        return false;
    }
    runtime->control->request_pause();
    runtime->info.state = SessionState::Paused;
    runtime->info.updated_at = std::chrono::system_clock::now();
    return true;
}

bool SessionManager::resume_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* runtime = find_runtime(session_id);
    if (!runtime || !runtime->control) {
        return false;
    }
    runtime->control->resume();
    runtime->info.state = SessionState::Running;
    runtime->info.updated_at = std::chrono::system_clock::now();
    return true;
}

bool SessionManager::stop_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* runtime = find_runtime(session_id);
    if (!runtime || !runtime->control) {
        return false;
    }
    runtime->control->request_stop();
    runtime->info.state = SessionState::Stopped;
    runtime->info.updated_at = std::chrono::system_clock::now();
    return true;
}

std::optional<ResultSnapshot> SessionManager::get_result(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* runtime = find_runtime(session_id);
    if (!runtime || !runtime->has_result) {
        return std::nullopt;
    }
    return runtime->result;
}

std::optional<pulsim::SimulationOptions> SessionManager::get_options(const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* runtime = find_runtime(session_id);
    if (!runtime) {
        return std::nullopt;
    }
    return runtime->options;
}

void SessionManager::cleanup_expired() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = std::chrono::system_clock::now();
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        auto& runtime = it->second;
        if (runtime->info.expiry <= now && runtime->has_result) {
            if (runtime->worker && runtime->worker->joinable()) {
                runtime->worker->join();
            }
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

SessionManager::SessionRuntime* SessionManager::find_runtime(const std::string& session_id) const {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return nullptr;
    }
    return it->second.get();
}

SessionInfo SessionManager::to_info(const SessionRuntime& runtime) const {
    return runtime.info;
}

}  // namespace pulsim::api::grpc
