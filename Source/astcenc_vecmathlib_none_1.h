// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2019-2020 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/*
 * This module implements flexible N-wide float and integer vectors, where the
 * width can be selected at compile time depending on the underlying ISA. It
 * is not possible to mix different ISAs (or vector widths) in a single file -
 * the ISA is statically selected when the header is first included.
 *
 * ISA support is provided for:
 *
 *     * 1-wide for scalar reference.
 *     * 4-wide for SSE2.
 *     * 4-wide for SSE4.2.
 *     * 8-wide for AVX2.
 *
 */

#ifndef ASTC_VECMATHLIB_NONE_1_H_INCLUDED
#define ASTC_VECMATHLIB_NONE_1_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <algorithm>
#include <math.h>
#include <string.h>

struct vfloat1
{
	ASTCENC_SIMD_INLINE vfloat1() {}
	ASTCENC_SIMD_INLINE explicit vfloat1(const float *p) { m = *p; }
	ASTCENC_SIMD_INLINE explicit vfloat1(float v) { m = v; }
	ASTCENC_SIMD_INLINE float lane(int i) const { (void)i; return m; }
	static ASTCENC_SIMD_INLINE vfloat1 zero() { return vfloat1(0.0f); }
	static ASTCENC_SIMD_INLINE vfloat1 lane_id() { return vfloat1(0.0f); }
	float m;
};

struct vint1
{
	ASTCENC_SIMD_INLINE vint1() {}
	ASTCENC_SIMD_INLINE explicit vint1(const int *p) { m = *p; }
	ASTCENC_SIMD_INLINE explicit vint1(int v) { m = v; }
	ASTCENC_SIMD_INLINE int lane(int i) const { (void)i; return m; }
	static ASTCENC_SIMD_INLINE vint1 lane_id() { return vint1(0); }
	int m;
};

struct vmask1
{
	ASTCENC_SIMD_INLINE explicit vmask1(bool v) { m = v; }
	bool m;
};


ASTCENC_SIMD_INLINE vfloat1 load1a_1f(const float* p) { return vfloat1(*p); }
ASTCENC_SIMD_INLINE vfloat1 loada_1f(const float* p) { return vfloat1(*p); }

ASTCENC_SIMD_INLINE vfloat1 operator+ (vfloat1 a, vfloat1 b) { a.m = a.m + b.m; return a; }
ASTCENC_SIMD_INLINE vfloat1 operator- (vfloat1 a, vfloat1 b) { a.m = a.m - b.m; return a; }
ASTCENC_SIMD_INLINE vfloat1 operator* (vfloat1 a, vfloat1 b) { a.m = a.m * b.m; return a; }
ASTCENC_SIMD_INLINE vfloat1 operator/ (vfloat1 a, vfloat1 b) { a.m = a.m / b.m; return a; }
ASTCENC_SIMD_INLINE vmask1 operator==(vfloat1 a, vfloat1 b) { return vmask1(a.m = a.m == b.m); }
ASTCENC_SIMD_INLINE vmask1 operator!=(vfloat1 a, vfloat1 b) { return vmask1(a.m = a.m != b.m); }
ASTCENC_SIMD_INLINE vmask1 operator< (vfloat1 a, vfloat1 b) { return vmask1(a.m = a.m < b.m); }
ASTCENC_SIMD_INLINE vmask1 operator> (vfloat1 a, vfloat1 b) { return vmask1(a.m = a.m > b.m); }
ASTCENC_SIMD_INLINE vmask1 operator<=(vfloat1 a, vfloat1 b) { return vmask1(a.m = a.m <= b.m); }
ASTCENC_SIMD_INLINE vmask1 operator>=(vfloat1 a, vfloat1 b) { return vmask1(a.m = a.m >= b.m); }
ASTCENC_SIMD_INLINE vmask1 operator| (vmask1 a, vmask1 b) { return vmask1(a.m || b.m); }
ASTCENC_SIMD_INLINE vmask1 operator& (vmask1 a, vmask1 b) { return vmask1(a.m && b.m); }
ASTCENC_SIMD_INLINE vmask1 operator^ (vmask1 a, vmask1 b) { return vmask1(a.m ^ b.m); }
ASTCENC_SIMD_INLINE unsigned mask(vmask1 v) { return v.m; }
ASTCENC_SIMD_INLINE bool any(vmask1 v) { return mask(v) != 0; }
ASTCENC_SIMD_INLINE bool all(vmask1 v) { return mask(v) != 0; }

ASTCENC_SIMD_INLINE vfloat1 min(vfloat1 a, vfloat1 b) { a.m = a.m < b.m ? a.m : b.m; return a; }
ASTCENC_SIMD_INLINE vfloat1 max(vfloat1 a, vfloat1 b) { a.m = a.m > b.m ? a.m : b.m; return a; }
ASTCENC_SIMD_INLINE vfloat1 saturate(vfloat1 a) { return vfloat1(std::min(std::max(a.m,0.0f), 1.0f)); }

ASTCENC_SIMD_INLINE vfloat1 abs(vfloat1 x) { return vfloat1(std::abs(x.m)); }

ASTCENC_SIMD_INLINE vfloat1 round(vfloat1 v)
{
	return vfloat1(std::floor(v.m + 0.5f));
}

ASTCENC_SIMD_INLINE vint1 floatToInt(vfloat1 v) { return vint1(v.m); }

ASTCENC_SIMD_INLINE vfloat1 intAsFloat(vint1 v) { vfloat1 r; memcpy(&r.m, &v.m, 4); return r; }
ASTCENC_SIMD_INLINE vint1 floatAsInt(vfloat1 v) { vint1 r; memcpy(&r.m, &v.m, 4); return r; }

ASTCENC_SIMD_INLINE vint1 operator~ (vint1 a) { a.m = ~a.m; return a; }
ASTCENC_SIMD_INLINE vint1 operator+ (vint1 a, vint1 b) { a.m = a.m + b.m; return a; }
ASTCENC_SIMD_INLINE vint1 operator- (vint1 a, vint1 b) { a.m = a.m - b.m; return a; }
ASTCENC_SIMD_INLINE vint1 operator| (vint1 a, vint1 b) { return vint1(a.m | b.m); }
ASTCENC_SIMD_INLINE vint1 operator& (vint1 a, vint1 b) { return vint1(a.m & b.m); }
ASTCENC_SIMD_INLINE vint1 operator^ (vint1 a, vint1 b) { return vint1(a.m ^ b.m); }
ASTCENC_SIMD_INLINE vmask1 operator< (vint1 a, vint1 b) { return vmask1(a.m = a.m < b.m); }
ASTCENC_SIMD_INLINE vmask1 operator> (vint1 a, vint1 b) { return vmask1(a.m = a.m > b.m); }
ASTCENC_SIMD_INLINE vmask1 operator==(vint1 a, vint1 b) { return vmask1(a.m = a.m == b.m); }
ASTCENC_SIMD_INLINE vmask1 operator!=(vint1 a, vint1 b) { return vmask1(a.m = a.m != b.m); }
ASTCENC_SIMD_INLINE vint1 min(vint1 a, vint1 b) { a.m = a.m < b.m ? a.m : b.m; return a; }
ASTCENC_SIMD_INLINE vint1 max(vint1 a, vint1 b) { a.m = a.m > b.m ? a.m : b.m; return a; }

ASTCENC_SIMD_INLINE vfloat1 hmin(vfloat1 v) { return v; }
ASTCENC_SIMD_INLINE vint1 hmin(vint1 v) { return v; }

ASTCENC_SIMD_INLINE void store(vfloat1 v, float* ptr) { *ptr = v.m; }
ASTCENC_SIMD_INLINE void store(vint1 v, int* ptr) { *ptr = v.m; }

ASTCENC_SIMD_INLINE void store_nbytes(vint1 v, uint8_t* ptr) { *ptr = (uint8_t)v.m; }

ASTCENC_SIMD_INLINE vfloat1 gatherf(const float* base, vint1 indices)
{
	return vfloat1(base[indices.m]);
}
ASTCENC_SIMD_INLINE vint1 gatheri(const int* base, vint1 indices)
{
	return vint1(base[indices.m]);
}

// packs low 8 bits of each lane into low 8 bits of result (a no-op in scalar code path)
ASTCENC_SIMD_INLINE vint1 pack_low_bytes(vint1 v)
{
	return v;
}


// "select", i.e. highbit(cond) ? b : a
ASTCENC_SIMD_INLINE vfloat1 select(vfloat1 a, vfloat1 b, vmask1 cond)
{
	return cond.m ? b : a;
}
ASTCENC_SIMD_INLINE vint1 select(vint1 a, vint1 b, vmask1 cond)
{
	return cond.m ? b : a;
}

#endif // #ifndef ASTC_VECMATHLIB_NONE_1_H_INCLUDED
