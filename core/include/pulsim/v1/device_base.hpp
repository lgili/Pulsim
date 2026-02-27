#pragma once

// =============================================================================
// PulsimCore v2 - Device Model Aggregator
// =============================================================================
// This header preserves the legacy include path while exposing a modular
// component library with one model per file.
// =============================================================================

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/components/resistor.hpp"
#include "pulsim/v1/components/capacitor.hpp"
#include "pulsim/v1/components/inductor.hpp"
#include "pulsim/v1/components/voltage_source.hpp"
#include "pulsim/v1/components/current_source.hpp"
#include "pulsim/v1/components/ideal_diode.hpp"
#include "pulsim/v1/components/ideal_switch.hpp"
#include "pulsim/v1/components/voltage_controlled_switch.hpp"
#include "pulsim/v1/components/mosfet.hpp"
#include "pulsim/v1/components/igbt.hpp"
#include "pulsim/v1/components/transformer.hpp"

namespace pulsim::v1 {

// =============================================================================
// Static Assertions to Verify Concepts
// =============================================================================

static_assert(StampableDevice<Resistor>, "Resistor must satisfy StampableDevice concept");
static_assert(is_linear_device_v<Resistor>, "Resistor must be linear");
static_assert(!is_dynamic_device_v<Resistor>, "Resistor must not be dynamic");

static_assert(StampableDevice<Capacitor>, "Capacitor must satisfy StampableDevice concept");
static_assert(is_linear_device_v<Capacitor>, "Capacitor must be linear");
static_assert(is_dynamic_device_v<Capacitor>, "Capacitor must be dynamic");

static_assert(StampableDevice<MOSFET>, "MOSFET must satisfy StampableDevice concept");
static_assert(!is_linear_device_v<MOSFET>, "MOSFET must be nonlinear");

static_assert(StampableDevice<IGBT>, "IGBT must satisfy StampableDevice concept");
static_assert(!is_linear_device_v<IGBT>, "IGBT must be nonlinear");

static_assert(StampableDevice<Transformer>, "Transformer must satisfy StampableDevice concept");
static_assert(is_linear_device_v<Transformer>, "Transformer must be linear");

// =============================================================================
// Device Registration for Runtime Introspection (C++26 Reflection Prep)
// =============================================================================
// These macros register device metadata for runtime introspection.
// In C++26, this will be replaced by static reflection.

PULSIM_REGISTER_DEVICE(Resistor, "Resistor", "passive", 2, true, false, false);
PULSIM_REGISTER_DEVICE(Capacitor, "Capacitor", "passive", 2, true, true, false);
PULSIM_REGISTER_DEVICE(Inductor, "Inductor", "passive", 2, true, true, false);
PULSIM_REGISTER_DEVICE(VoltageSource, "VoltageSource", "source", 2, true, false, false);
PULSIM_REGISTER_DEVICE(CurrentSource, "CurrentSource", "source", 2, true, false, false);
PULSIM_REGISTER_DEVICE(IdealDiode, "IdealDiode", "active", 2, false, false, false);
PULSIM_REGISTER_DEVICE(IdealSwitch, "IdealSwitch", "switch", 2, true, false, false);
PULSIM_REGISTER_DEVICE(MOSFET, "MOSFET", "active", 3, false, false, true);
PULSIM_REGISTER_DEVICE(IGBT, "IGBT", "active", 3, false, false, true);
PULSIM_REGISTER_DEVICE(Transformer, "Transformer", "passive", 4, true, false, false);

// Register device parameters for introspection
PULSIM_REGISTER_PARAMS(Resistor,
    PULSIM_PARAM("resistance", "Ohm", 1000.0, 0.0, 1e12)
);

PULSIM_REGISTER_PARAMS(Capacitor,
    PULSIM_PARAM("capacitance", "F", 1e-6, 0.0, 1e3),
    PULSIM_PARAM("initial_voltage", "V", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(Inductor,
    PULSIM_PARAM("inductance", "H", 1e-3, 0.0, 1e3),
    PULSIM_PARAM("initial_current", "A", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(VoltageSource,
    PULSIM_PARAM("voltage", "V", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(CurrentSource,
    PULSIM_PARAM("current", "A", 0.0, -1e6, 1e6)
);

PULSIM_REGISTER_PARAMS(MOSFET,
    PULSIM_PARAM("vth", "V", 2.0, -10.0, 10.0),
    PULSIM_PARAM("kp", "A/V^2", 0.1, 0.0, 100.0),
    PULSIM_PARAM("lambda", "1/V", 0.01, 0.0, 1.0)
);

PULSIM_REGISTER_PARAMS(IGBT,
    PULSIM_PARAM("vth", "V", 5.0, 0.0, 20.0),
    PULSIM_PARAM("g_on", "S", 1e4, 0.0, 1e6),
    PULSIM_PARAM("v_ce_sat", "V", 1.5, 0.0, 10.0)
);

PULSIM_REGISTER_PARAMS(Transformer,
    PULSIM_PARAM("turns_ratio", "", 1.0, 0.001, 1000.0)
);

}  // namespace pulsim::v1
