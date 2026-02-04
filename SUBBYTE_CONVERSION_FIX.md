# Fix for Sub-byte Type Conversion Build Error

## Problem Statement

The codebase had a build error when attempting to convert from `uint8_t` to `uint4_t` (and `int8_t` to `int4_t`). The error was:

```
error: SYCL kernel cannot call a recursive function
  251 |   reorder(dst_c, dst, dlayout, dlayout);
```

## Root Cause: Infinite Recursion

The `ReorderDispatchRelayoutConvert` and `ReorderDispatchConvertRelayout` implementations made **two reorder() calls**:

```cpp
reorder(src, dst_c, slayout, dlayout);   // Step 1
reorder(dst_c, dst, dlayout, dlayout);   // Step 2 <- Problem!
```

The second call with `dlayout, dlayout` (same layout) was intended to be just a type conversion. However, calling `reorder()` re-entered the dispatch logic, which could return the same recursive dispatcher again, causing infinite recursion.

## Solution: Two-Part Fix

### Part 1: Tactical (Specialized Implementations)

Added optimized `Xe_Reorder` implementations for common conversions:
- `Xe_Reorder<ReorderKind::UU, uint8_t, uint4_t>` - Optimized packing assembly
- `Xe_Reorder<ReorderKind::UU, int8_t, int4_t>` - Reuses uint8→uint4

**File:** `include/cute/arch/reorder_xe.hpp`

### Part 2: Strategic (Fix Recursive Implementations)

**Fixed the recursive implementations** to avoid re-entering dispatch for type conversions:

**Before (Problematic):**
```cpp
void reorder_impl(ReorderDispatchRelayoutConvert const&, ...) {
  auto dst_c = make_fragment_like<NewDstType>(dst);
  
  reorder(src, dst_c, slayout, dlayout);   // Layout change
  reorder(dst_c, dst, dlayout, dlayout);   // Type conversion <- Recursion!
}
```

**After (Fixed):**
```cpp
void reorder_impl(ReorderDispatchRelayoutConvert const&, ...) {
  auto dst_c = make_fragment_like<NewDstType>(dst);
  
  reorder(src, dst_c, slayout, dlayout);   // Layout change (may dispatch)
  reorder_impl(Universal_Reorder_UU<NewDstType, DstType>{}, 
               dst_c, dst, dlayout, dlayout);  // Direct call - no recursion!
}
```

**Key Change:** Instead of calling `reorder()` which re-enters dispatch, we directly call `reorder_impl()` with `Universal_Reorder_UU` for the type conversion step.

**Files Modified:**
- `include/cute/algorithm/reorder.hpp` - Fixed both recursive implementations
- `include/cute/atom/reorder_atom_xe.hpp` - Restored original dispatch logic

## Why This Works

### The Problem with Calling reorder()

When `reorder()` is called, it:
1. Calls `choose_xe_reorder_impl()` to select implementation
2. Returns a dispatcher (e.g., `ReorderDispatchRelayoutConvert`)
3. Calls `reorder_impl()` with that dispatcher

For the second call with same layout (`dlayout, dlayout`):
- Should just do type conversion
- But `choose_xe_reorder_impl()` sees `is_subbyte_v<DType>` or type mismatch
- Returns **same recursive dispatcher again**
- Infinite loop!

### The Solution: Direct Call

By calling `reorder_impl(Universal_Reorder_UU{}, ...)` directly:
- Skips dispatch logic entirely
- Goes straight to type conversion atom
- `Universal_Reorder_UU` does: `dst = DstType(src)` (element-wise)
- Layout transformation handled by `reorder_impl` wrapper
- No recursion possible

## Dispatch Logic (Restored)

The dispatch logic is back to its original form:

```cpp
if constexpr (has_xe_optimized_reorder<rclass, SType, DType>())
  return Xe_Reorder<rclass, SType, DType>{};           // Optimized
else if constexpr (rclass == ReorderKind::UU_Universal)
  return Universal_Reorder_UU<SType, DType>{};         // Universal
else if constexpr (is_subbyte_v<SType>)
  return ReorderDispatchConvertRelayout{};             // Convert then relayout (now safe)
else if constexpr (is_subbyte_v<DType>)
  return ReorderDispatchRelayoutConvert{};             // Relayout then convert (now safe)
else if constexpr (!is_same_v<SType, DType>)
  return ReorderDispatchConvertRelayout{};             // Type conversion (now safe)
else
  return ReorderDispatchXeGeneric{};                   // Same-type reorder
```

All paths are now safe because the recursive implementations have been fixed.

## Example: uint8_t → uint4_t

**Before Fix:**
1. `reorder(uint8 → uint4)` calls dispatch
2. Dispatch returns `ReorderDispatchRelayoutConvert` (because `is_subbyte_v<uint4_t>`)
3. First call: `reorder(uint8 → uint8, layout change)` - OK
4. Second call: `reorder(uint8 → uint4, same layout)` - calls dispatch again!
5. Dispatch returns `ReorderDispatchRelayoutConvert` again - **infinite recursion**

**After Fix:**
1. `reorder(uint8 → uint4)` calls dispatch
2. Dispatch sees specialized `Xe_Reorder<UU, uint8_t, uint4_t>` exists → use it (optimized path)
3. OR if no specialization: Dispatch returns `ReorderDispatchRelayoutConvert`
4. First call: `reorder(uint8 → uint8, layout change)` - dispatches normally
5. Second call: `reorder_impl(Universal_Reorder_UU{}, uint8 → uint4, same layout)` - **direct call, no recursion!**

## Impact

### Part 1 (Specialized Implementations)
✅ Optimized performance for uint8→uint4, int8→int4  
✅ Enabled 4 sub-byte conversion tests

### Part 2 (Fixed Recursive Implementations)
✅ **Recursive implementations now safe** - no infinite loops  
✅ Original dispatch logic restored and working  
✅ Two-step approach (convert then relayout, or vice versa) works correctly  
✅ Maintains intended separation of concerns

## Files Modified

1. **`include/cute/arch/reorder_xe.hpp`**
   - Added `Xe_Reorder<UU, uint8_t, uint4_t>` specialization
   - Added `Xe_Reorder<UU, int8_t, int4_t>` alias

2. **`include/cute/algorithm/reorder.hpp`**
   - Fixed `reorder_impl(ReorderDispatchConvertRelayout)` to use direct call
   - Fixed `reorder_impl(ReorderDispatchRelayoutConvert)` to use direct call

3. **`include/cute/atom/reorder_atom_xe.hpp`**
   - Restored original dispatch logic with all conditions

4. **`test/unit/cute/intel_xe/reorder.cpp`**
   - Enabled 4 sub-byte conversion tests
   - Updated test data initialization

## Comparison: Bypassing vs Fixing

| Approach | Previous "Strategic" Fix | Current Fix |
|----------|-------------------------|-------------|
| Method | Bypass recursive dispatchers | Fix recursive implementations |
| Dispatch | Removed recursive conditions | Restored all conditions |
| Implementation | Use Universal_Reorder_UU only | Fixed to call Universal_Reorder_UU directly |
| Semantics | Lost two-step separation | Maintains two-step approach |
| Safety | Safe (no recursion) | Safe (no recursion) |
| Performance | Universal fallback for all | Optimized when available, safe fallback when needed |

## Conclusion

The fix properly addresses the recursion issue by:
1. **Keeping** the two-step conversion approach (convert-then-relayout or relayout-then-convert)
2. **Fixing** the implementations to avoid re-entering dispatch for the type conversion step
3. **Adding** optimized specializations for critical conversions
4. **Maintaining** the original design intent while ensuring SYCL compatibility

Result: **Safe, efficient, and maintainable** type conversion system.
