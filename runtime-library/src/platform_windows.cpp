// Windows platform hooks of runtime_core.cpp; see platform.h.  Must not include
// the DX-RT header (see platform.h).

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "platform.h"

namespace {

HMODULE g_self_ref = nullptr;  // guarded by runtime_core.cpp's load lock

}  // namespace

void platform::hold_self_reference() noexcept {
    if (g_self_ref) return;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                            reinterpret_cast<LPCWSTR>(&platform::hold_self_reference), &g_self_ref)) {
        g_self_ref = nullptr;
    }
}

void platform::drop_self_reference() noexcept {
    if (g_self_ref) {
        FreeLibrary(g_self_ref);  // the caller still holds its own reference
        g_self_ref = nullptr;
    }
}

extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID reserved) {
    // reserved != NULL: the process is exiting and every other thread has
    // ended.  DllMain runs before this DLL's static destructors.
    if (reason == DLL_PROCESS_DETACH && reserved != nullptr) platform::abandon_models();
    return TRUE;
}
