#include "pulsim/api/grpc/server_config.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace pulsim::api::grpc {

namespace {

std::string getenv_or(const char* key, const std::string& fallback) {
    const char* value = std::getenv(key);
    if (!value) {
        return fallback;
    }
    return value;
}

bool getenv_bool(const char* key, bool fallback) {
    const char* value = std::getenv(key);
    if (!value) {
        return fallback;
    }
    std::string lower;
    lower.reserve(std::strlen(value));
    for (const char c : std::string_view(value)) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return lower == "1" || lower == "true" || lower == "yes";
}

std::optional<std::string> getenv_optional(const char* key) {
    const char* value = std::getenv(key);
    if (!value || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
}

std::size_t getenv_size(const char* key, std::size_t fallback) {
    const char* value = std::getenv(key);
    if (!value) {
        return fallback;
    }
    try {
        return static_cast<std::size_t>(std::stoll(value));
    } catch (...) {
        return fallback;
    }
}

std::chrono::seconds getenv_duration(const char* key, std::chrono::seconds fallback) {
    const char* value = std::getenv(key);
    if (!value) {
        return fallback;
    }
    try {
        return std::chrono::seconds(std::stoll(value));
    } catch (...) {
        return fallback;
    }
}

}  // namespace

ServerConfig load_config_from_env() {
    ServerConfig config;
    config.listen_address = getenv_or("PULSIM_GRPC_ADDR", config.listen_address);
    config.enable_reflection = getenv_bool("PULSIM_GRPC_REFLECTION", config.enable_reflection);
    config.enable_authentication = getenv_bool("PULSIM_GRPC_AUTH", config.enable_authentication);
    config.session_retention = getenv_duration("PULSIM_SESSION_RETENTION", config.session_retention);
    config.max_sessions = getenv_size("PULSIM_MAX_SESSIONS", config.max_sessions);
    config.max_sessions_per_user = getenv_size("PULSIM_MAX_SESSIONS_PER_USER", config.max_sessions_per_user);
    config.enable_metrics = getenv_bool("PULSIM_GRPC_METRICS", config.enable_metrics);
    config.metrics_listen_address = getenv_or("PULSIM_METRICS_ADDR", config.metrics_listen_address);
    config.metrics_port = static_cast<std::uint16_t>(getenv_size("PULSIM_METRICS_PORT", config.metrics_port));
    config.enable_tracing = getenv_bool("PULSIM_GRPC_TRACING", config.enable_tracing);
    config.otlp_endpoint = getenv_optional("PULSIM_OTLP_ENDPOINT");
    config.version = getenv_or("PULSIM_VERSION", config.version);

    if (config.enable_authentication) {
        if (auto token = getenv_optional("PULSIM_DEFAULT_TOKEN")) {
            config.default_token = token;
            config.allowed_tokens.push_back(*token);
        }
    }

    return config;
}

}  // namespace pulsim::api::grpc
