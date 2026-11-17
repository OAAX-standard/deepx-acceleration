// Platform hooks of runtime_core.cpp.
//
// Windows: unload without runtime_cleanup.
//
// Static destructors run at DLL_PROCESS_DETACH, under the loader lock, and at
// process exit only after Windows has ended every other thread.  Destroying a
// loaded model there waits for DX-RT's threads, which then never finish: the
// host hangs in FreeLibrary or at exit.  So:
//   - While models are loaded the DLL holds a reference to itself, dropped by
//     runtime_cleanup.  FreeLibrary without cleanup then only drops the host's
//     reference and returns; the DLL stays loaded until the process exits.
//   - At process exit DllMain moves the loaded models out of g_models into
//     storage that is never destroyed, so no engine destructor runs.  Windows
//     reclaims the memory and the device handles.
//
// The Windows side is in platform_windows.cpp, which includes <windows.h> and
// never the DX-RT header: <windows.h> defines macros such as ERROR that break
// the DX-RT C++ header, so the two must not meet in one translation unit.
// Elsewhere the hooks do nothing.

#ifndef RUNTIME_PLATFORM_H
#define RUNTIME_PLATFORM_H

namespace platform {

#ifdef _WIN32
// Called by runtime_core.cpp with its load lock held.
void hold_self_reference() noexcept;  // after models are loaded
void drop_self_reference() noexcept;  // at the end of runtime_cleanup

// Defined by runtime_core.cpp; called by DllMain at process exit, when no
// other thread is left.  Keeps the loaded models from being destroyed.
void abandon_models() noexcept;
#else
inline void hold_self_reference() noexcept {}
inline void drop_self_reference() noexcept {}
#endif

}  // namespace platform

#endif  // RUNTIME_PLATFORM_H
