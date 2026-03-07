/**
 * @file simulation_control.cpp
 * @brief Thread-safe pause/resume/stop coordination used by long-running simulations.
 */

#include "pulsim/simulation_control.hpp"

namespace pulsim {

/// Requests transition from Running to Paused state.
void SimulationController::request_pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    SimulationState current = state_.load(std::memory_order_acquire);
    if (current == SimulationState::Running) {
        state_.store(SimulationState::Paused, std::memory_order_release);
        cv_.notify_all();
    }
}

/// Requests transition from Paused to Running state.
void SimulationController::request_resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    SimulationState current = state_.load(std::memory_order_acquire);
    if (current == SimulationState::Paused) {
        state_.store(SimulationState::Running, std::memory_order_release);
        cv_.notify_all();
    }
}

/// Requests cooperative stop when simulation is Running or Paused.
void SimulationController::request_stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    SimulationState current = state_.load(std::memory_order_acquire);
    if (current == SimulationState::Running || current == SimulationState::Paused) {
        state_.store(SimulationState::Stopping, std::memory_order_release);
        cv_.notify_all();
    }
}

/// Resets controller to Idle state and wakes all waiters.
void SimulationController::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.store(SimulationState::Idle, std::memory_order_release);
    cv_.notify_all();
}

/// Waits until the internal state reaches the requested target state.
bool SimulationController::wait_for_state(SimulationState target, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (timeout_ms < 0) {
        // Wait indefinitely
        cv_.wait(lock, [this, target]() {
            return state_.load(std::memory_order_acquire) == target;
        });
        return true;
    } else {
        // Wait with timeout
        return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, target]() {
            return state_.load(std::memory_order_acquire) == target;
        });
    }
}

/// Forces state transition and notifies all waiters.
void SimulationController::set_state(SimulationState new_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.store(new_state, std::memory_order_release);
    cv_.notify_all();
}

/**
 * @brief Handles pause semantics in cooperative simulation loops.
 * @return `false` when loop should terminate due to stop request; `true` otherwise.
 */
bool SimulationController::check_and_handle_pause() {
    SimulationState current = state_.load(std::memory_order_acquire);

    // If stopping, return false to signal simulation should end
    if (current == SimulationState::Stopping) {
        return false;
    }

    // If paused, wait until resumed or stopped
    if (current == SimulationState::Paused) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {
            SimulationState s = state_.load(std::memory_order_acquire);
            return s != SimulationState::Paused;
        });

        // Check if we're stopping after being unpaused
        current = state_.load(std::memory_order_acquire);
        if (current == SimulationState::Stopping) {
            return false;
        }
    }

    return true;
}

// Legacy compatibility
/// Returns whether simulation should stop according to legacy state semantics.
bool SimulationController::should_stop() const {
    SimulationState current = state_.load(std::memory_order_acquire);
    return current == SimulationState::Stopping ||
           current == SimulationState::Completed ||
           current == SimulationState::Error;
}

/// Returns whether simulation is currently paused.
bool SimulationController::should_pause() const {
    return state_.load(std::memory_order_acquire) == SimulationState::Paused;
}

/// Blocks while state remains Paused.
void SimulationController::wait_until_resumed() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() {
        SimulationState s = state_.load(std::memory_order_acquire);
        return s != SimulationState::Paused;
    });
}

}  // namespace pulsim
