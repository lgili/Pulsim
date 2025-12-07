#include "pulsim/api/grpc/server_config.hpp"
#include "pulsim/api/grpc/session_manager.hpp"
#include "pulsim/api/grpc/simulator.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <csignal>
#include <iostream>
#include <memory>
#include <utility>

namespace pulsim::api::grpc {
std::pair<std::unique_ptr<::grpc::Server>, std::unique_ptr<::pulsim::api::v1::SimulatorService::Service>>
build_server(SessionManager& manager, const ServerConfig& config);
}

int main() {
    using namespace pulsim::api::grpc;

    ServerConfig config = load_config_from_env();
    SessionManager manager(config);

    auto built = build_server(manager, config);
    auto service = std::move(built.second);  // keep service alive
    auto& server = built.first;

    std::cout << "Pulsim gRPC server listening on " << config.listen_address << std::endl;
    server->Wait();
    return 0;
}
