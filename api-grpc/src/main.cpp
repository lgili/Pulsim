#include "spicelab/api/grpc/server_config.hpp"
#include "spicelab/api/grpc/session_manager.hpp"
#include "spicelab/api/grpc/simulator.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <csignal>
#include <iostream>
#include <memory>
#include <utility>

namespace spicelab::api::grpc {
std::pair<std::unique_ptr<::grpc::Server>, std::unique_ptr<::spicelab::api::v1::SimulatorService::Service>>
build_server(SessionManager& manager, const ServerConfig& config);
}

int main() {
    using namespace spicelab::api::grpc;

    ServerConfig config = load_config_from_env();
    SessionManager manager(config);

    auto built = build_server(manager, config);
    auto service = std::move(built.second);  // keep service alive
    auto& server = built.first;

    std::cout << "SpiceLab gRPC server listening on " << config.listen_address << std::endl;
    server->Wait();
    return 0;
}
