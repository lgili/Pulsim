// CLI integration test: run `pulsim validate` and `pulsim run` and verify outputs
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <cstdlib>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#endif

using namespace std::filesystem;

static std::string find_cli_binary() {
    const char* candidates[] = {
        "./cli/pulsim",
        "../cli/pulsim",
        "../../cli/pulsim",
        "build/cli/pulsim",
        "../build/cli/pulsim",
        "./pulsim",
        "../pulsim"
    };

    for (const char* c : candidates) {
        std::error_code ec;
        path p = canonical(path(c), ec);
        if (!ec && exists(p) && is_regular_file(p)) {
            return p.string();
        }
    }

    // Fallback: try to find in PATH
    const char* path_env = std::getenv("PATH");
    if (path_env) {
        std::string pathstr(path_env);
        size_t start = 0;
        while (true) {
            size_t pos = pathstr.find(':', start);
            std::string dir = (pos == std::string::npos) ? pathstr.substr(start) : pathstr.substr(start, pos - start);
            path p = path(dir) / "pulsim";
            if (exists(p) && is_regular_file(p)) return p.string();
            if (pos == std::string::npos) break;
            start = pos + 1;
        }
    }

    return {};
}

TEST_CASE("CLI integration: validate and run", "[cli][integration]") {
    path tmp = temp_directory_path() / path("pulsim-cli-test");
    for (int i = 0; i < 1000 && exists(tmp); ++i) tmp += std::to_string(i);
    create_directories(tmp);

    path netlist = tmp / "rc_test.json";
    path outcsv = tmp / "out.csv";
    {
        std::ofstream f(netlist);
        f << R"({
  "name": "RC Test",
  "components": [
    { "type": "voltage_source", "name": "V1", "npos": "in", "nneg": "0", "waveform": { "type": "dc", "value": 5.0 } },
    { "type": "resistor", "name": "R1", "n1": "in", "n2": "out", "value": "1k" },
    { "type": "capacitor", "name": "C1", "n1": "out", "n2": "0", "value": "1u" }
  ],
  "simulation": { "type": "transient", "tstop": 1e-4, "dt": 1e-6 }
})";
    }

    auto cli = find_cli_binary();
    REQUIRE(!cli.empty());

    auto run_cli = [&](const std::vector<std::string>& args, const path& capture_file) -> int {
#if defined(__unix__) || defined(__APPLE__)
        pid_t pid = fork();
        if (pid == -1) return -1;
        if (pid == 0) {
            int fd = open(capture_file.string().c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            if (fd >= 0) {
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                close(fd);
            }

            std::vector<char*> argv;
            argv.push_back(const_cast<char*>(cli.c_str()));
            for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);

            execv(cli.c_str(), argv.data());
            _exit(127);
        } else {
            int status = 0;
            waitpid(pid, &status, 0);
            if (WIFEXITED(status)) return WEXITSTATUS(status);
            if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
            return status;
        }
#else
        std::string cmd = '"' + cli + '"';
        for (const auto& a : args) { cmd += " "; cmd += a; }
        cmd += " > "; cmd += capture_file.string();
        cmd += " 2>&1";
        return std::system(cmd.c_str());
#endif
    };

    // 1) Validate
    {
        path out = tmp / "validate.txt";
        std::vector<std::string> args = {"validate", netlist.string()};
        int rc = run_cli(args, out);
        if (rc != 0) {
            std::ifstream in(out);
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            WARN("Validate command failed, output:\n" + content);
        }
        REQUIRE(rc == 0);
        std::ifstream in(out);
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(content.find("OK") != std::string::npos);
    }

    // 2) Run
    {
        path out = tmp / "run.txt";
        std::vector<std::string> args = {"run", netlist.string(), "-o", outcsv.string()};
        int rc = run_cli(args, out);
        if (rc != 0) {
            std::ifstream in(out);
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            WARN("Run command failed, output:\n" + content);
        }
        REQUIRE(rc == 0);
        REQUIRE(exists(outcsv));
        std::ifstream in(outcsv);
        std::string header;
        std::getline(in, header);
        CHECK(header.rfind("time", 0) == 0);
    }

    std::error_code ec;
    remove_all(tmp, ec);
}
