#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim::api::grpc {

struct ServerConfig {
    std::string listen_address = "0.0.0.0:50051";
    bool enable_reflection = true;
    bool enable_authentication = false;
    std::vector<std::string> allowed_tokens;
    std::optional<std::string> default_token;
    std::chrono::seconds session_retention = std::chrono::minutes(10);
    std::size_t max_sessions = 64;
    std::size_t max_sessions_per_user = 8;
    bool enable_metrics = true;
    std::string metrics_listen_address = "0.0.0.0";
    std::uint16_t metrics_port = 9464;
    bool enable_tracing = false;
    std::optional<std::string> otlp_endpoint;  // OTLP/HTTP endpoint
    std::string version = "0.1.0";
};

ServerConfig load_config_from_env();

}  // namespace pulsim::api::grpc
