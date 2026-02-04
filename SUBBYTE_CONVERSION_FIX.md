# Fix for Sub-byte Type Conversion Build Error

## Problem Statement

The codebase had a build error when attempting to convert from `uint8_t` to `uint4_t` (and `int8_t` to `int4_t`). The error was:

```
error: SYCL kernel cannot call a recursive function
  251 |   reorder(dst_c, dst, dlayout, dlayout);
```

This occurred in the `ReorderDispatchRelayoutConvert` implementation in `include/cute/algorithm/reorder.hpp`.

## Root Cause Analysis

### 1. Dispatch Logic Flow

The `choose_xe_reorder_impl` function (in `include/cute/atom/reorder_atom_xe.hpp`) determines which reorder implementation to use:

```cpp
if constexpr (has_xe_optimized_reorder<rclass, SType, DType>())
  return Xe_Reorder<rclass, SType, DType>{};
else if constexpr (rclass == ReorderKind::UU_Universal)
  return Universal_Reorder_UU<SType, DType>{};
else if constexpr (is_subbyte_v<SType>)
  return ReorderDispatchConvertRelayout{};
else if constexpr (is_subbyte_v<DType>)  // <- THIS PATH TRIGGERED
  return ReorderDispatchRelayoutConvert{};
else if constexpr (!is_same_v<remove_cv_t<SType>, remove_cv_t<DType>>)
  return ReorderDispatchConvertRelayout{};
else
  return ReorderDispatchXeGeneric{};
```

For `uint8_t` → `uint4_t` conversion:
- `is_subbyte_v<uint8_t>` = false (8 bits, not < 8)
- `is_subbyte_v<uint4_t>` = **true** (4 bits < 8)
- Therefore, it returns `ReorderDispatchRelayoutConvert`

### 2. The Recursion Problem

The `ReorderDispatchRelayoutConvert` implementation:

```cpp
template <...>
void reorder_impl(ReorderDispatchRelayoutConvert const&, ...) {
  using NewDstType = conditional_t<is_same_v<SrcType, DstType>, 
                                   upcast_subbyte_t<DstType>, 
                                   SrcType>;
  auto dst_c = make_fragment_like<NewDstType>(dst);
  
  reorder(src, dst_c, slayout, dlayout);   // Layout change
  reorder(dst_c, dst, dlayout, dlayout);   // Type conversion <-- RECURSION!
}
```

For `uint8_t` → `uint4_t`:
- `NewDstType` = `uint8_t` (since `SrcType != DstType`)
- First call: `reorder(uint8_t → uint8_t)` - OK
- Second call: `reorder(uint8_t → uint4_t)` - **INFINITE RECURSION**

The second call re-enters the same dispatch logic, which again returns `ReorderDispatchRelayoutConvert`, causing infinite recursion that SYCL kernels cannot handle.

### 3. Why No Optimized Path?

There was **no specialized `Xe_Reorder` implementation** for `uint8_t` → `uint4_t`. The code had specializations for:
- `uint4_t` → `half_t`, `uint4_t` → `bfloat16_t`
- `uint8_t` → `half_t`, `uint8_t` → `bfloat16_t`
- `uint4_t` → `uint4_t` (identity, UV kind only)

But **no `uint8_t` → `uint4_t`** narrowing conversion.

## Solution

Added specialized `Xe_Reorder` implementations for narrowing conversions:

### 1. `Xe_Reorder<ReorderKind::UU, uint8_t, uint4_t>`

Located in `include/cute/arch/reorder_xe.hpp`:

```cpp
template <>
struct Xe_Reorder<ReorderKind::UU, uint8_t, uint4_t>
{
  using SRegisters = intel::uchar4[1];  // 64 uint8_t elements (64 bytes in 4 GRFs)
  using DRegisters = intel::uchar2[1];  // 64 uint4_t elements packed into 32 bytes (2 GRFs)

  CUTE_HOST_DEVICE static void
  reorder(intel::uchar4 const& src0, intel::uchar2& dst0)
  {
    // Assembly implementation:
    // - Takes 64 input bytes (uint8_t values)
    // - Packs pairs into 32 output bytes (each byte contains 2 uint4_t values)
    // - OUT[i] = (IN[2i+1] & 0xF) << 4 | (IN[2i] & 0xF)
  }
};
```

### 2. `Xe_Reorder<ReorderKind::UU, int8_t, int4_t>`

Reuses the `uint8_t` → `uint4_t` implementation since the bit-level representation is identical.

### 3. Updated Test Cases

Enabled the previously disabled test cases in `test/unit/cute/intel_xe/reorder.cpp`:
- `conversion_uint8_to_uint4_subgroup`
- `conversion_int8_to_int4_subgroup`
- `conversion_uint8_to_uint4_tensor`
- `conversion_int8_to_int4_tensor`

### 4. Test Data Initialization

Updated `initialize_conversion_source` to handle `uint8_t` source type, ensuring test values stay within the 4-bit range [0, 15] for proper validation.

## Technical Details

### Register Layout
- Intel Xe GPUs use GRF (General Register Files) of 16 bytes each
- `intel::uchar4` = 64 bytes = 4 GRF registers
- `intel::uchar2` = 32 bytes = 2 GRF registers

### Packing Strategy
Each pair of input `uint8_t` values is packed into one output byte:
- Lower nibble: `input[2i] & 0x0F`
- Upper nibble: `(input[2i+1] & 0x0F) << 4`

This matches the standard sub-byte packing convention used throughout the codebase.

### Reorder Classification
For `uint8_t` → `uint4_t` with identity layout:
- SV = 32/8 = 4 (source elements per 32-bit channel)
- DV = 32/4 = 8 (destination elements per 32-bit channel)
- Classified as `ReorderKind::UU` (unit-to-unit) for simple identity reorders

## Impact

- **Fixes build error**: Eliminates SYCL kernel recursion error
- **Enables sub-byte conversions**: Allows `uint8_t`/`int8_t` → `uint4_t`/`int4_t` conversions
- **No breaking changes**: Only adds new specializations, doesn't modify existing behavior
- **Test coverage**: Four test cases now validate the conversion correctness

## Files Modified

1. `include/cute/arch/reorder_xe.hpp` - Added specialized reorder implementations
2. `test/unit/cute/intel_xe/reorder.cpp` - Enabled tests and updated initialization

## Future Work

Additional narrowing conversions could be added if needed:
- `uint16_t` → `uint8_t`
- `uint16_t` → `uint4_t`
- Other signed/unsigned narrowing conversions

These would follow the same pattern: add specialized `Xe_Reorder` implementations to avoid the recursive dispatch path.
