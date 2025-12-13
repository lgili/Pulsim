#pragma once

/**
 * @file simulation_control.hpp
 * @brief Simulation control, progress monitoring, and streaming configuration.
 *
 * This header provides the infrastructure for GUI integration with long-running
 * simulations, including:
 * - SimulationController: Thread-safe pause/resume/stop control
 * - SimulationProgress: Real-time progress information for progress bars
 * - ProgressCallbackConfig: Configuration for progress callbacks
 * - StreamingConfig: Memory-efficient result storage options
 *
 * @section usage_example Basic Usage Example
 * @code
 * #include "pulsim/simulation_control.hpp"
 * #include "pulsim/simulation.hpp"
 *
 * // Create controller (shared between GUI and simulation threads)
 * SimulationController controller;
 *
 * // In simulation thread:
 * Simulator sim(circuit, opts);
 * ProgressCallbackConfig progress_config;
 * progress_config.callback = [](const SimulationProgress& p) {
 *     std::cout << "Progress: " << p.progress_percent << "%" << std::endl;
 * };
 * progress_config.min_interval_ms = 100;  // Update every 100ms max
 *
 * auto result = sim.run_transient_with_progress(nullptr, nullptr, &controller, progress_config);
 *
 * // In GUI thread (e.g., button click handlers):
 * controller.request_pause();   // Pause simulation
 * controller.request_resume();  // Resume simulation
 * controller.request_stop();    // Stop simulation
 * @endcode
 */

#include "pulsim/types.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>

namespace pulsim {

// Forward declaration
struct SimulationProgress;

/**
 * @brief Abstract interface for cooperative control of long-running simulations.
 *
 * @deprecated Use SimulationController instead for new code. This interface is
 * kept for backward compatibility only.
 */
class SimulationControl {
public:
    virtual ~SimulationControl() = default;
    virtual bool should_stop() const = 0;
    virtual bool should_pause() const = 0;
    virtual void wait_until_resumed() = 0;
};

/**
 * @brief Callback function type for progress updates.
 *
 * Called periodically during simulation to report progress. The callback
 * should be fast and non-blocking to avoid slowing down the simulation.
 *
 * @param progress Current simulation progress information
 */
using ProgressCallback = std::function<void(const SimulationProgress& progress)>;

/**
 * @brief Real-time progress information during simulation.
 *
 * This structure is passed to progress callbacks to provide current
 * simulation status. All fields are updated at each callback invocation.
 *
 * @section gui_usage GUI Integration
 * Use this information to:
 * - Update progress bars (progress_percent)
 * - Show elapsed/remaining time estimates
 * - Display convergence warnings
 * - Monitor memory usage (if enabled)
 */
struct SimulationProgress {
    // Time progress
    Real current_time = 0.0;          ///< Current simulation time (seconds)
    Real total_time = 0.0;            ///< Target end time (seconds)
    double progress_percent = 0.0;    ///< Progress from 0.0 to 100.0

    // Step statistics
    int64_t steps_completed = 0;          ///< Number of timesteps completed
    int64_t total_steps_estimate = -1;    ///< Estimated total steps (-1 if unknown)

    // Current step info
    int newton_iterations = 0;        ///< Newton iterations for current step
    bool convergence_warning = false; ///< True if current step needed >10 iterations

    // Wall-clock timing
    double elapsed_seconds = 0.0;              ///< Wall-clock time elapsed
    double estimated_remaining_seconds = -1.0; ///< Estimated time remaining (-1 if unknown)

    // Memory usage (optional, may be expensive to compute)
    int64_t memory_bytes = -1;        ///< Current memory usage (-1 if not tracked)
};

/**
 * @brief Configuration for progress callback behavior.
 *
 * Controls how frequently progress callbacks are invoked during simulation.
 * Callbacks are throttled to avoid excessive overhead while still providing
 * responsive GUI updates.
 *
 * @section throttling Throttling Behavior
 * A callback is invoked when BOTH conditions are met:
 * - At least min_interval_ms milliseconds have elapsed since last callback
 * - At least min_steps simulation steps have completed since last callback
 *
 * @code
 * ProgressCallbackConfig config;
 * config.callback = [](const SimulationProgress& p) { update_ui(p); };
 * config.min_interval_ms = 50;   // Max 20 updates per second
 * config.min_steps = 10;         // But at least 10 steps between updates
 * config.include_memory = false; // Don't track memory (faster)
 * @endcode
 */
struct ProgressCallbackConfig {
    ProgressCallback callback;        ///< Callback function (required)
    double min_interval_ms = 100.0;   ///< Minimum milliseconds between callbacks
    int min_steps = 100;              ///< Minimum simulation steps between callbacks
    bool include_memory = false;      ///< Track memory usage (slower if true)
};

/**
 * @brief Thread-safe simulation controller for pause/resume/stop operations.
 *
 * SimulationController provides a thread-safe mechanism for GUI applications
 * to control long-running simulations. It implements a state machine with
 * the following states:
 *
 * - **Idle**: Initial state, simulation not started
 * - **Running**: Simulation actively executing
 * - **Paused**: Simulation paused, can be resumed
 * - **Stopping**: Stop requested, finishing current step
 * - **Completed**: Simulation finished successfully
 * - **Error**: Simulation terminated with error
 *
 * @section state_transitions State Transitions
 * ```
 * Idle --[start]--> Running
 * Running --[request_pause]--> Paused
 * Paused --[request_resume]--> Running
 * Running/Paused --[request_stop]--> Stopping --> Completed
 * Running --> Error (on simulation error)
 * Any --[reset]--> Idle
 * ```
 *
 * @section thread_safety Thread Safety
 * All public methods are thread-safe and can be called from any thread.
 * Typical usage has the simulation running in a background thread while
 * the GUI thread calls control methods like request_pause().
 *
 * @code
 * // GUI application example
 * SimulationController controller;
 *
 * // Start simulation in background thread
 * std::thread sim_thread([&]() {
 *     Simulator sim(circuit, opts);
 *     ProgressCallbackConfig config;
 *     config.callback = [](const SimulationProgress& p) {
 *         // Update progress bar (ensure thread-safe UI update)
 *     };
 *     sim.run_transient_with_progress(nullptr, nullptr, &controller, config);
 * });
 *
 * // GUI button handlers
 * void on_pause_clicked() { controller.request_pause(); }
 * void on_resume_clicked() { controller.request_resume(); }
 * void on_stop_clicked() { controller.request_stop(); }
 *
 * // Wait for completion
 * controller.wait_for_state(SimulationState::Completed, 30000);
 * sim_thread.join();
 * @endcode
 */
class SimulationController : public SimulationControl {
public:
    SimulationController() = default;
    ~SimulationController() = default;

    // Non-copyable, non-movable (contains mutex)
    SimulationController(const SimulationController&) = delete;
    SimulationController& operator=(const SimulationController&) = delete;
    SimulationController(SimulationController&&) = delete;
    SimulationController& operator=(SimulationController&&) = delete;

    /// @name State Queries
    /// Thread-safe methods to query current simulation state.
    /// @{

    /** @brief Get the current simulation state. */
    SimulationState state() const { return state_.load(std::memory_order_acquire); }

    /** @brief Check if simulation is in Idle state (not started). */
    bool is_idle() const { return state() == SimulationState::Idle; }

    /** @brief Check if simulation is actively running. */
    bool is_running() const { return state() == SimulationState::Running; }

    /** @brief Check if simulation is paused. */
    bool is_paused() const { return state() == SimulationState::Paused; }

    /** @brief Check if simulation is stopping (finishing current step). */
    bool is_stopping() const { return state() == SimulationState::Stopping; }

    /** @brief Check if simulation completed successfully. */
    bool is_completed() const { return state() == SimulationState::Completed; }

    /** @brief Check if simulation terminated with an error. */
    bool is_error() const { return state() == SimulationState::Error; }

    /// @}

    /// @name Control Commands
    /// Thread-safe methods to control simulation execution.
    /// These are typically called from the GUI thread.
    /// @{

    /**
     * @brief Request simulation to pause at next checkpoint.
     *
     * The simulation will pause after completing the current timestep.
     * Call request_resume() to continue.
     */
    void request_pause();

    /**
     * @brief Request simulation to resume after pause.
     *
     * Only effective when simulation is in Paused state.
     */
    void request_resume();

    /**
     * @brief Request simulation to stop.
     *
     * The simulation will stop after completing the current timestep.
     * Results up to the stop point will be available.
     */
    void request_stop();

    /**
     * @brief Reset controller to Idle state.
     *
     * Call this before starting a new simulation to clear any previous state.
     */
    void reset();

    /**
     * @brief Wait for simulation to reach a specific state.
     *
     * Blocks the calling thread until the target state is reached or timeout.
     *
     * @param target The state to wait for
     * @param timeout_ms Maximum time to wait in milliseconds (-1 for infinite)
     * @return true if target state was reached, false if timeout occurred
     */
    bool wait_for_state(SimulationState target, int timeout_ms = -1);

    /// @}

    /// @name Internal Methods
    /// Used by Simulator class - not intended for direct use.
    /// @{

    /** @brief Set simulation state (called by Simulator). */
    void set_state(SimulationState new_state);

    /**
     * @brief Check for pause/stop requests (called by Simulator).
     * @return false if should stop, true to continue
     */
    bool check_and_handle_pause();

    /// @}

    // SimulationControl interface (for backward compatibility)
    bool should_stop() const override;
    bool should_pause() const override;
    void wait_until_resumed() override;

private:
    std::atomic<SimulationState> state_{SimulationState::Idle};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

/**
 * @brief Configuration for memory-efficient result streaming.
 *
 * For long simulations, storing every timestep can consume excessive memory.
 * StreamingConfig provides options to reduce memory usage:
 *
 * - **Decimation**: Store only every Nth point (e.g., decimation_factor=10
 *   stores 1/10th of the data)
 * - **Rolling buffer**: Keep only the most recent max_points, discarding older data
 *
 * @section streaming_example Example
 * @code
 * SimulationOptions opts;
 * opts.streaming_decimation = 10;       // Store every 10th point
 * opts.streaming_rolling_buffer = true; // Enable rolling buffer
 * opts.streaming_max_points = 10000;    // Keep last 10000 points
 *
 * // For a 1-second simulation at 1us timestep (1M steps):
 * // Without streaming: 1,000,000 points stored
 * // With decimation=10: 100,000 points stored
 * // With rolling buffer: always <= 10,000 points
 * @endcode
 */
struct StreamingConfig {
    int decimation_factor = 1;      ///< Store every Nth point (1 = all points)
    bool use_rolling_buffer = false; ///< If true, keep only last max_points
    int64_t max_points = 100000;    ///< Maximum points when rolling buffer enabled
    double callback_interval_ms = 0; ///< Callback throttle (0 = every stored point)
};

}  // namespace pulsim
