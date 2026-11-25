"""The release profile's simulate() keywords, passed EXPLICITLY (no pytest plugin): the
release findings F-Y-* reproduced on this line with these and form='branch' devices."""

RELEASE = dict(engine="pwl", discontinuity_restart=True, enable_substep_state_correction=True,
               mode_selection="auto", control_tick="state", device_path="interface")
