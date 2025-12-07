#pragma once

#include "pulsim/api/grpc/server_config.hpp"
#include "pulsim/simulation.hpp"
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pulsim::api::grpc {

enum class SessionState {
    Created,
    Validating,
    Ready,
    Running,
    Paused,
    Completed,
    Stopped,
    Failed,
};

struct SessionInfo {
    std::string session_id;
    std::string model_id;
    std::string name;
    std::string owner;
    SessionState state = SessionState::Created;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    std::chrono::system_clock::time_point expiry;
    std::vector<std::string> active_signals;
    std::string error_message;
};

struct ResultSnapshot {
    pulsim::SimulationResult result;
    SessionState final_state = SessionState::Completed;
    std::string error_message;
};

struct WaveformHeaderInfo {
    std::string session_id;
    std::vector<std::string> signals;
    double tstart = 0.0;
    double tstop = 0.0;
    std::uint64_t total_samples = 0;
};

struct WaveformSampleInfo {
    double time = 0.0;
    std::vector<double> values;
};

struct WaveformCompleteInfo {
    SessionState final_state = SessionState::Completed;
    std::string error_message;
};

class SimulationControlImpl;

class WaveformSubscriber {
public:
    struct Event {
        enum class Type { Header, Sample, Complete } type;
        WaveformHeaderInfo header;
        WaveformSampleInfo sample;
        WaveformCompleteInfo complete;
    };

    WaveformSubscriber(WaveformHeaderInfo header,
                       std::vector<int> indices,
                       uint32_t decimation,
                       std::optional<double> start_time);

    void push_sample(double time, const pulsim::Vector& state);
    void push_complete(const WaveformCompleteInfo& info);

    bool next_event(Event& event);

    const WaveformHeaderInfo& header() const { return header_; }

private:
    void enqueue_header_locked();

    WaveformHeaderInfo header_;
    std::vector<int> indices_;
    uint32_t decimation_;
    std::optional<double> start_time_;
    std::uint64_t sample_counter_ = 0;
    bool header_sent_ = false;
    bool closed_ = false;
    std::deque<Event> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class SessionManager {
public:
    explicit SessionManager(const ServerConfig& config);
    ~SessionManager();

    std::string create_session(const std::string& name,
                               const std::string& owner,
                               const std::optional<std::string>& model_id,
                               pulsim::Circuit circuit,
                               pulsim::SimulationOptions options,
                               const std::vector<std::string>& signals);

    std::vector<SessionInfo> list_sessions() const;
    std::optional<SessionInfo> get_session(const std::string& session_id) const;

    bool start_session(const std::string& session_id,
                       const pulsim::SimulationOptions& overrides);

    std::shared_ptr<WaveformSubscriber> attach_stream(const std::string& session_id,
                                                      const std::vector<std::string>& signals,
                                                      uint32_t decimation,
                                                      std::optional<double> start_time);

    bool pause_session(const std::string& session_id);
    bool resume_session(const std::string& session_id);
    bool stop_session(const std::string& session_id);

    std::optional<ResultSnapshot> get_result(const std::string& session_id) const;
    std::optional<pulsim::SimulationOptions> get_options(const std::string& session_id) const;

    void cleanup_expired();

private:
    struct SessionRuntime;

    SessionRuntime* find_runtime(const std::string& session_id) const;
    SessionInfo to_info(const SessionRuntime& runtime) const;

    const ServerConfig config_;
    std::unordered_map<std::string, std::unique_ptr<SessionRuntime>> sessions_;
    mutable std::mutex mutex_;
};

}  // namespace pulsim::api::grpc
