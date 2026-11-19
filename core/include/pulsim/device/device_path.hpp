#pragma once

// add-unified-device-interface, dual path (DECISIONS #23): which code
// stamps the Newton kinds of a run. `Legacy` is the hand stamps, the
// default and bit-identical to 2.0; `Interface` evaluates every Newton
// kind through its law on the `device::Device` contract (device/laws.hpp,
// pwl/interface_stamps.hpp). A run is one or the other, never mixed; a
// circuit with a kind not yet on the interface refuses `Interface` by
// name (pwl::StatefulHistories::set_path). The release flips the default
// and deletes the hand stamps.

#include <cstdint>

namespace pulsim::device {

enum class DevicePath : std::uint8_t { Legacy = 0, Interface = 1 };

}  // namespace pulsim::device
