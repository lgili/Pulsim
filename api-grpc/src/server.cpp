#include "pulsim/api/grpc/server_config.hpp"
#include "pulsim/api/grpc/session_manager.hpp"
#include "pulsim/api/grpc/simulator.grpc.pb.h"
#include "pulsim/api/grpc/simulator.pb.h"
#include "pulsim/parser.hpp"

#include <grpcpp/grpcpp.h>
#include <google/protobuf/timestamp.pb.h>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pulsim::api::grpc {
namespace {

::google::protobuf::Timestamp to_timestamp(const std::chrono::system_clock::time_point& tp) {
    const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(tp);
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(tp - secs);
    ::google::protobuf::Timestamp ts;
    ts.set_seconds(secs.time_since_epoch().count());
    ts.set_nanos(static_cast<int>(nanos.count()));
    return ts;
}

pulsim::SimulationOptions merge_options(pulsim::SimulationOptions base,
                                          const ::pulsim::api::v1::SimulationOptions& req) {
    if (req.has_tstart()) base.tstart = req.tstart().value();
    if (req.has_tstop()) base.tstop = req.tstop().value();
    if (req.has_dt()) base.dt = req.dt().value();
    if (req.has_dtmin()) base.dtmin = req.dtmin().value();
    if (req.has_dtmax()) base.dtmax = req.dtmax().value();
    if (req.has_abstol()) base.abstol = req.abstol().value();
    if (req.has_reltol()) base.reltol = req.reltol().value();
    if (req.has_max_newton_iterations()) base.max_newton_iterations = req.max_newton_iterations().value();
    if (req.has_damping_factor()) base.damping_factor = req.damping_factor().value();
    if (req.has_use_ic()) base.use_ic = req.use_ic().value();
    base.output_signals.assign(req.output_signals().begin(), req.output_signals().end());
    return base;
}

void fill_session_descriptor(const SessionInfo& info, ::pulsim::api::v1::SessionDescriptor* out) {
    out->set_session_id(info.session_id);
    out->set_model_id(info.model_id);
    out->set_name(info.name);
    out->set_status(static_cast<::pulsim::api::v1::SessionStatus>(static_cast<int>(info.state)));
    *out->mutable_created_at() = to_timestamp(info.created_at);
    *out->mutable_updated_at() = to_timestamp(info.updated_at);
    for (const auto& s : info.active_signals) {
        out->add_active_signals(s);
    }
    out->mutable_retention()->set_seconds(std::chrono::duration_cast<std::chrono::seconds>(info.expiry - info.created_at).count());
    out->set_owner(info.owner);
}

void fill_metadata(const ResultSnapshot& snapshot, ::pulsim::api::v1::ResultMetadata* out) {
    if (!snapshot.result.time.empty()) {
        out->set_start_time(snapshot.result.time.front());
        out->set_end_time(snapshot.result.time.back());
    }
    out->set_sample_count(snapshot.result.time.size());
    for (const auto& s : snapshot.result.signal_names) {
        out->add_signals(s);
    }
    out->set_status(static_cast<::pulsim::api::v1::SessionStatus>(static_cast<int>(snapshot.final_state)));
    out->set_error_message(snapshot.error_message);
}

class SimulatorServiceImpl final : public ::pulsim::api::v1::SimulatorService::Service {
public:
    explicit SimulatorServiceImpl(SessionManager& mgr, const ServerConfig& cfg)
        : mgr_(mgr), cfg_(cfg), start_time_(std::chrono::steady_clock::now()) {}

    ::grpc::Status HealthCheck(::grpc::ServerContext*, const ::pulsim::api::v1::HealthCheckRequest*, ::pulsim::api::v1::HealthCheckResponse* resp) override {
        resp->set_status(::pulsim::api::v1::HEALTH_STATUS_OK);
        resp->set_version(cfg_.version);
        auto uptime = std::chrono::steady_clock::now() - start_time_;
        resp->mutable_uptime()->set_seconds(std::chrono::duration_cast<std::chrono::seconds>(uptime).count());
        resp->set_active_sessions(static_cast<uint32_t>(mgr_.list_sessions().size()));
        resp->set_completed_sessions(0);
        resp->set_authentication_enabled(cfg_.enable_authentication);
        return ::grpc::Status::OK;
    }

    ::grpc::Status CreateSession(::grpc::ServerContext*, const ::pulsim::api::v1::CreateSessionRequest* req, ::pulsim::api::v1::CreateSessionResponse* resp) override {
        if (!req->has_inline_model()) {
            return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, "inline_model required");
        }

        const auto& model = req->inline_model();
        auto parse = pulsim::NetlistParser::parse_string(model.model_json());
        if (!parse) {
            return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT, parse.error().to_string());
        }
        pulsim::Circuit circuit = *parse;
        pulsim::SimulationOptions options = merge_options(pulsim::SimulationOptions{}, req->options());
        std::vector<std::string> signals(options.output_signals.begin(), options.output_signals.end());

        auto session_id = mgr_.create_session(model.name(), "", std::nullopt, std::move(circuit), options, signals);
        auto info = mgr_.get_session(session_id);
        if (!info) {
            return ::grpc::Status(::grpc::StatusCode::INTERNAL, "failed to create session");
        }
        fill_session_descriptor(*info, resp->mutable_session());
        return ::grpc::Status::OK;
    }

    ::grpc::Status ListSessions(::grpc::ServerContext*, const ::pulsim::api::v1::ListSessionsRequest*, ::pulsim::api::v1::ListSessionsResponse* resp) override {
        for (const auto& info : mgr_.list_sessions()) {
            fill_session_descriptor(info, resp->add_sessions());
        }
        return ::grpc::Status::OK;
    }

    ::grpc::Status GetSession(::grpc::ServerContext*, const ::pulsim::api::v1::GetSessionRequest* req, ::pulsim::api::v1::GetSessionResponse* resp) override {
        auto info = mgr_.get_session(req->session_id());
        if (!info) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "session not found");
        }
        fill_session_descriptor(*info, resp->mutable_session());
        return ::grpc::Status::OK;
    }

    ::grpc::Status StartSimulation(::grpc::ServerContext*, const ::pulsim::api::v1::StartSimulationRequest* req, ::pulsim::api::v1::StartSimulationResponse* resp) override {
        auto base_opts = mgr_.get_options(req->session_id());
        if (!base_opts) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "session not found");
        }
        auto merged = merge_options(*base_opts, req->overrides());
        if (!mgr_.start_session(req->session_id(), merged)) {
            return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION, "cannot start session");
        }
        if (auto info = mgr_.get_session(req->session_id())) {
            fill_session_descriptor(*info, resp->mutable_session());
        }
        return ::grpc::Status::OK;
    }

    ::grpc::Status PauseSimulation(::grpc::ServerContext*, const ::pulsim::api::v1::PauseSimulationRequest* req, ::pulsim::api::v1::PauseSimulationResponse* resp) override {
        if (!mgr_.pause_session(req->session_id())) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "session not found");
        }
        if (auto info = mgr_.get_session(req->session_id())) {
            fill_session_descriptor(*info, resp->mutable_session());
        }
        return ::grpc::Status::OK;
    }

    ::grpc::Status ResumeSimulation(::grpc::ServerContext*, const ::pulsim::api::v1::ResumeSimulationRequest* req, ::pulsim::api::v1::ResumeSimulationResponse* resp) override {
        if (!mgr_.resume_session(req->session_id())) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "session not found");
        }
        if (auto info = mgr_.get_session(req->session_id())) {
            fill_session_descriptor(*info, resp->mutable_session());
        }
        return ::grpc::Status::OK;
    }

    ::grpc::Status StopSimulation(::grpc::ServerContext*, const ::pulsim::api::v1::StopSimulationRequest* req, ::pulsim::api::v1::StopSimulationResponse* resp) override {
        if (!mgr_.stop_session(req->session_id())) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "session not found");
        }
        if (auto info = mgr_.get_session(req->session_id())) {
            fill_session_descriptor(*info, resp->mutable_session());
        }
        return ::grpc::Status::OK;
    }

    ::grpc::Status StreamWaveforms(::grpc::ServerContext*, const ::pulsim::api::v1::StreamWaveformsRequest* req, ::grpc::ServerWriter<::pulsim::api::v1::WaveformStreamResponse>* writer) override {
        auto subscriber = mgr_.attach_stream(req->session_id(), {req->signals().begin(), req->signals().end()}, req->decimation(), req->has_start_time() ? std::make_optional(req->start_time().value()) : std::nullopt);
        if (!subscriber) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "session not found or signals invalid");
        }

        WaveformSubscriber::Event event;
        while (subscriber->next_event(event)) {
            ::pulsim::api::v1::WaveformStreamResponse resp;
            switch (event.type) {
                case WaveformSubscriber::Event::Type::Header: {
                    auto* h = resp.mutable_header();
                    h->set_session_id(event.header.session_id);
                    for (const auto& s : event.header.signals) h->add_signals(s);
                    h->set_tstart(event.header.tstart);
                    h->set_tstop(event.header.tstop);
                    h->set_total_samples(event.header.total_samples);
                    break;
                }
                case WaveformSubscriber::Event::Type::Sample: {
                    auto* s = resp.mutable_sample();
                    s->set_time(event.sample.time);
                    for (double v : event.sample.values) s->add_values(v);
                    break;
                }
                case WaveformSubscriber::Event::Type::Complete: {
                    auto* c = resp.mutable_complete();
                    c->set_final_status(static_cast<::pulsim::api::v1::SessionStatus>(static_cast<int>(event.complete.final_state)));
                    c->set_error_message(event.complete.error_message);
                    writer->Write(resp);
                    return ::grpc::Status::OK;
                }
            }
            if (!writer->Write(resp)) {
                return ::grpc::Status::OK;  // client closed
            }
        }
        return ::grpc::Status::OK;
    }

    ::grpc::Status GetResult(::grpc::ServerContext*, const ::pulsim::api::v1::GetResultRequest* req, ::pulsim::api::v1::GetResultResponse* resp) override {
        auto snapshot = mgr_.get_result(req->session_id());
        if (!snapshot) {
            return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "result not ready");
        }
        fill_metadata(*snapshot, resp->mutable_metadata());
        return ::grpc::Status::OK;
    }

    ::grpc::Status DownloadResult(::grpc::ServerContext*, const ::pulsim::api::v1::DownloadResultRequest*, ::grpc::ServerWriter<::pulsim::api::v1::DownloadResultResponse>*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "DownloadResult not implemented");
    }

    ::grpc::Status UploadModel(::grpc::ServerContext*, const ::pulsim::api::v1::UploadModelRequest*, ::pulsim::api::v1::UploadModelResponse*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Model storage not implemented");
    }

    ::grpc::Status ListModels(::grpc::ServerContext*, const ::pulsim::api::v1::ListModelsRequest*, ::pulsim::api::v1::ListModelsResponse*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Model storage not implemented");
    }

    ::grpc::Status GetModel(::grpc::ServerContext*, const ::pulsim::api::v1::GetModelRequest*, ::pulsim::api::v1::GetModelResponse*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Model storage not implemented");
    }

    ::grpc::Status DeleteModel(::grpc::ServerContext*, const ::pulsim::api::v1::DeleteModelRequest*, ::pulsim::api::v1::DeleteModelResponse*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Model storage not implemented");
    }

    ::grpc::Status CreateSweep(::grpc::ServerContext*, const ::pulsim::api::v1::CreateSweepRequest*, ::pulsim::api::v1::CreateSweepResponse*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Sweeps not implemented");
    }

    ::grpc::Status RunSweep(::grpc::ServerContext*, const ::pulsim::api::v1::RunSweepRequest*, ::pulsim::api::v1::RunSweepResponse*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Sweeps not implemented");
    }

    ::grpc::Status GetSweepResults(::grpc::ServerContext*, const ::pulsim::api::v1::GetSweepResultsRequest*, ::pulsim::api::v1::GetSweepResultsResponse*) override {
        return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "Sweeps not implemented");
    }

private:
    SessionManager& mgr_;
    const ServerConfig& cfg_;
    std::chrono::steady_clock::time_point start_time_;
};

}  // namespace

std::pair<std::unique_ptr<::grpc::Server>, std::unique_ptr<::pulsim::api::v1::SimulatorService::Service>>
build_server(SessionManager& manager, const ServerConfig& config) {
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(config.listen_address, ::grpc::InsecureServerCredentials());
    auto service = std::make_unique<SimulatorServiceImpl>(manager, config);
    builder.RegisterService(service.get());

    auto server = builder.BuildAndStart();
    return {std::unique_ptr<::grpc::Server>(server.release()), std::move(service)};
}

}  // namespace pulsim::api::grpc
