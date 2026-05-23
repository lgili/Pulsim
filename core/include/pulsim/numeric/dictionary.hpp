#pragma once

// =============================================================================
// Pulsim — Layer 0: portable ordered-map alias
// =============================================================================
//
// ``pulsim::numeric::Dictionary<K, V>`` aliases either ``std::flat_map``
// (C++23, libc++ 18+, libstdc++ 15+) or ``std::map`` (C++17, always
// available). Both are ordered maps, so callers can rely on iteration
// order without changing semantics across compilers.
//
// Why we prefer ``flat_map`` when available:
//
//   * Storage is two parallel contiguous vectors (keys, values) — one
//     cache line per ~8 entries on a 64-bit machine. ``std::map`` is a
//     red-black tree with one heap allocation per node.
//   * For the typical Pulsim cache sizes (2^N switch combinations, N≤8 →
//     ≤256 entries), the contiguous lookup is 2–5× faster than
//     ``std::unordered_map`` and 1.5–3× faster than ``std::map``.
//
// Iteration / insertion / lookup semantics match ``std::map``. The only
// callers that need to care about the underlying storage are those that
// hold long-lived pointers/iterators across mutations (``flat_map``
// invalidates everything on insertion, ``std::map`` only invalidates the
// erased iterator). None of the in-tree Pulsim consumers do that.

#if defined(__has_include)
#  if __has_include(<flat_map>)
#    include <flat_map>
#  endif
#endif

#include <functional>
#include <map>

namespace pulsim::numeric {

#if defined(__cpp_lib_flat_map) && __cpp_lib_flat_map >= 202207L
template <class Key, class Value, class Compare = std::less<Key>>
using Dictionary = std::flat_map<Key, Value, Compare>;

inline constexpr bool kDictionaryIsFlat = true;
#else
template <class Key, class Value, class Compare = std::less<Key>>
using Dictionary = std::map<Key, Value, Compare>;

inline constexpr bool kDictionaryIsFlat = false;
#endif

}  // namespace pulsim::numeric
