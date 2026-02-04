# Fix for Sub-byte Type Conversion Build Error

## Problem Statement

The codebase had a build error when attempting to convert from `uint8_t` to `uint4_t` (and `int8_t` to `int4_t`). The error was:

```
error: SYCL kernel cannot call a recursive function
  251 |   reorder(dst_c, dst, dlayout, dlayout);
```

## Root Cause: Recursive Dispatch Paths

The `choose_xe_reorder_impl` function used recursive dispatch for type conversions:

**Before (problematic):**
```cpp
if constexpr (has_xe_optimized_reorder<rclass, SType, DType>())
  return Xe_Reorder<rclass, SType, DType>{};
else if constexpr (is_subbyte_v<SType>)
  return ReorderDispatchConvertRelayout{};      // RECURSIVE!
else if constexpr (is_subbyte_v<DType>)  
  return ReorderDispatchRelayoutConvert{};      // RECURSIVE!
else if constexpr (!is_same_v<SType, DType>)
  return ReorderDispatchConvertRelayout{};      // RECURSIVE!
```

These recursive implementations caused infinite loops in SYCL kernels.

## Solution: Two-Part Fix

### Part 1: Add Specialized Implementations (Tactical)
Added `Xe_Reorder<ReorderKind::UU, uint8_t, uint4_t>` and `int8_t→int4_t` for optimized conversions.

### Part 2: Replace Recursive Dispatch (Strategic)

**After (fixed):**
```cpp
if constexpr (has_xe_optimized_reorder<rclass, SType, DType>())
  return Xe_Reorder<rclass, SType, DType>{};
else if constexpr (rclass == ReorderKind::UU_Universal)
  return Universal_Reorder_UU<SType, DType>{};
else if constexpr (!is_same_v<SType, DType>)
  // Type conversion: use Universal_Reorder_UU (no recursion)
  return Universal_Reorder_UU<SType, DType>{};
else
  return ReorderDispatchXeGeneric{};
```

## Why This Works

`Universal_Reorder_UU` performs element-wise type conversion:
```cpp
dst0 = DstType(src0);
```

Layout transformation is handled by the `reorder_impl` wrapper, which:
1. Computes layout mapping
2. Maps source → destination indices
3. Calls atom's reorder on properly mapped elements

Therefore: No recursion, works for any type conversion!

## Impact

✅ **Eliminates all recursive dispatch** - prevents future recursion errors  
✅ Safe fallback for ANY type conversion  
✅ Optimized paths still available via specialized Xe_Reorder  
⚠️ Unoptimized conversions slower (acceptable for correctness)

## Files Modified

1. `include/cute/arch/reorder_xe.hpp` - Added uint8→uint4, int8→int4 specializations
2. `include/cute/atom/reorder_atom_xe.hpp` - **Replaced recursive dispatch with Universal_Reorder_UU**
3. `test/unit/cute/intel_xe/reorder.cpp` - Enabled tests

## Verification

**Before:** Build fails with recursion error  
**After:** All conversions work (optimized if specialized, safe fallback otherwise)
