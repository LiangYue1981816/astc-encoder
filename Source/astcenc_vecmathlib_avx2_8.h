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

#ifndef ASTC_VECMATHLIB_AVX2_8_H_INCLUDED
#define ASTC_VECMATHLIB_AVX2_8_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>

// N-wide float
struct vfloat8
{
	ASTCENC_SIMD_INLINE vfloat8() {}
	// Initialize with N floats from an unaligned memory address.
	// Using loada() when address is aligned might be more optimal.
	ASTCENC_SIMD_INLINE explicit vfloat8(const float *p) { m = _mm256_loadu_ps(p); }
	// Initialize with the same given float value in all lanes.
	ASTCENC_SIMD_INLINE explicit vfloat8(float v) { m = _mm256_set1_ps(v); }

	ASTCENC_SIMD_INLINE explicit vfloat8(__m256 v) { m = v; }

	// Get SIMD lane #i value.
	template <int l> ASTCENC_SIMD_INLINE float lane() const
	{
		#ifdef _MSC_VER
		return m.m256_f32[i];
		#else
		union { __m256 m; float f[8]; } cvt;
		cvt.m = m;
		return cvt.f[l];
		#endif
	}

	// Float vector with all zero values
	static ASTCENC_SIMD_INLINE vfloat8 zero() { return vfloat8(_mm256_setzero_ps()); }

	// Initialize with one float in all SIMD lanes, from an aligned memory address.
	static ASTCENC_SIMD_INLINE vfloat8 load1(const float* p) { return vfloat8(_mm256_broadcast_ss(p)); }

	// Initialize with N floats from an aligned memory address.
	static ASTCENC_SIMD_INLINE vfloat8 loada(const float* p) { return vfloat8(_mm256_load_ps(p)); }

	// Float vector with each lane having the lane index (0, 1, 2, ...)
	static ASTCENC_SIMD_INLINE vfloat8 lane_id() { return vfloat8(_mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)); }

	__m256 m;
};

// N-wide integer (32 bit in each lane)
struct vint8
{
	ASTCENC_SIMD_INLINE vint8() {}
	// Initialize with N ints from an unaligned memory address.
	ASTCENC_SIMD_INLINE explicit vint8(const int *p) { m = _mm256_loadu_si256((const __m256i*)p); }
	// Initialize with the same given integer value in all lanes.
	ASTCENC_SIMD_INLINE explicit vint8(int v) { m = _mm256_set1_epi32(v); }

	ASTCENC_SIMD_INLINE explicit vint8(__m256i v) { m = v; }

	// Get SIMD lane #i value
	template <int l> ASTCENC_SIMD_INLINE int lane() const
	{
		#ifdef _MSC_VER
		return m.m256i_i32[i];
		#else
		union { __m256i m; int f[8]; } cvt;
		cvt.m = m;
		return cvt.f[l];
		#endif
	}

	// Integer vector with each lane having the lane index (0, 1, 2, ...)
	static ASTCENC_SIMD_INLINE vint8 lane_id() { return vint8(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

	__m256i m;
};

// N-wide comparison mask. vmask is a result of comparison operators,
// and an argument for select() function below.
struct vmask8
{
	ASTCENC_SIMD_INLINE explicit vmask8(__m256 v) { m = v; }
	ASTCENC_SIMD_INLINE explicit vmask8(__m256i v) { m = _mm256_castsi256_ps(v); }
	__m256 m;
};

// Per-lane float arithmetic operations
ASTCENC_SIMD_INLINE vfloat8 operator+ (vfloat8 a, vfloat8 b) { a.m = _mm256_add_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat8 operator- (vfloat8 a, vfloat8 b) { a.m = _mm256_sub_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat8 operator* (vfloat8 a, vfloat8 b) { a.m = _mm256_mul_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat8 operator/ (vfloat8 a, vfloat8 b) { a.m = _mm256_div_ps(a.m, b.m); return a; }

// Per-lane float comparison operations
ASTCENC_SIMD_INLINE vmask8 operator==(vfloat8 a, vfloat8 b) { return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_EQ_OQ)); }
ASTCENC_SIMD_INLINE vmask8 operator!=(vfloat8 a, vfloat8 b) { return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_NEQ_OQ)); }
ASTCENC_SIMD_INLINE vmask8 operator< (vfloat8 a, vfloat8 b) { return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_LT_OQ)); }
ASTCENC_SIMD_INLINE vmask8 operator> (vfloat8 a, vfloat8 b) { return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_GT_OQ)); }
ASTCENC_SIMD_INLINE vmask8 operator<=(vfloat8 a, vfloat8 b) { return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_LE_OQ)); }
ASTCENC_SIMD_INLINE vmask8 operator>=(vfloat8 a, vfloat8 b) { return vmask8(_mm256_cmp_ps(a.m, b.m, _CMP_GE_OQ)); }

// Logical operations on comparison mask values
ASTCENC_SIMD_INLINE vmask8 operator| (vmask8 a, vmask8 b) { return vmask8(_mm256_or_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask8 operator& (vmask8 a, vmask8 b) { return vmask8(_mm256_and_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask8 operator^ (vmask8 a, vmask8 b) { return vmask8(_mm256_xor_ps(a.m, b.m)); }

// Returns a 8-bit code where bit0..bit7 map to lanes
ASTCENC_SIMD_INLINE unsigned mask(vmask8 v) { return _mm256_movemask_ps(v.m); }
// Whether any lane in the comparison mask is set
ASTCENC_SIMD_INLINE bool any(vmask8 v) { return mask(v) != 0; }
// Whether all lanes in the comparison mask are set
ASTCENC_SIMD_INLINE bool all(vmask8 v) { return mask(v) == 0xFF; }

// Per-lane float min & max
ASTCENC_SIMD_INLINE vfloat8 min(vfloat8 a, vfloat8 b) { a.m = _mm256_min_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat8 max(vfloat8 a, vfloat8 b) { a.m = _mm256_max_ps(a.m, b.m); return a; }

// Per-lane clamp to 0..1 range
ASTCENC_SIMD_INLINE vfloat8 clampzo(vfloat8 a)
{
	__m256 zero = _mm256_setzero_ps();
	__m256 one = _mm256_set1_ps(1.0f);
	return vfloat8(_mm256_min_ps(_mm256_max_ps(a.m, zero), one));
}

ASTCENC_SIMD_INLINE vfloat8 abs(vfloat8 x)
{
	__m256 msk = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	return vfloat8(_mm256_and_ps(x.m, msk));
}

// Round to nearest integer (nearest even for .5 cases)
ASTCENC_SIMD_INLINE vfloat8 round(vfloat8 v)
{
	return vfloat8(_mm256_round_ps(v.m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Per-lane convert to integer (truncate)
ASTCENC_SIMD_INLINE vint8 float_to_int(vfloat8 v) { return vint8(_mm256_cvttps_epi32(v.m)); }

// Reinterpret-bitcast integer vector as a float vector (this is basically a no-op on the CPU)
ASTCENC_SIMD_INLINE vfloat8 int_as_float(vint8 v) { return vfloat8(_mm256_castsi256_ps(v.m)); }
// Reinterpret-bitcast float vector as an integer vector (this is basically a no-op on the CPU)
ASTCENC_SIMD_INLINE vint8 float_as_int(vfloat8 v) { return vint8(_mm256_castps_si256(v.m)); }

ASTCENC_SIMD_INLINE vint8 operator~ (vint8 a) { return vint8(_mm256_xor_si256(a.m, _mm256_set1_epi32(-1))); }
ASTCENC_SIMD_INLINE vmask8 operator~ (vmask8 a) { return vmask8(_mm256_xor_si256(_mm256_castps_si256(a.m), _mm256_set1_epi32(-1))); }

// Per-lane arithmetic integer operations
ASTCENC_SIMD_INLINE vint8 operator+ (vint8 a, vint8 b) { a.m = _mm256_add_epi32(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vint8 operator- (vint8 a, vint8 b) { a.m = _mm256_sub_epi32(a.m, b.m); return a; }

// Per-lane logical bit operations
ASTCENC_SIMD_INLINE vint8 operator| (vint8 a, vint8 b) { return vint8(_mm256_or_si256(a.m, b.m)); }
ASTCENC_SIMD_INLINE vint8 operator& (vint8 a, vint8 b) { return vint8(_mm256_and_si256(a.m, b.m)); }
ASTCENC_SIMD_INLINE vint8 operator^ (vint8 a, vint8 b) { return vint8(_mm256_xor_si256(a.m, b.m)); }

// Per-lane integer comparison operations
ASTCENC_SIMD_INLINE vmask8 operator< (vint8 a, vint8 b) { return vmask8(_mm256_cmpgt_epi32(b.m, a.m)); }
ASTCENC_SIMD_INLINE vmask8 operator> (vint8 a, vint8 b) { return vmask8(_mm256_cmpgt_epi32(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask8 operator==(vint8 a, vint8 b) { return vmask8(_mm256_cmpeq_epi32(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask8 operator!=(vint8 a, vint8 b) { return ~vmask8(_mm256_cmpeq_epi32(a.m, b.m)); }

// Per-lane integer min & max
ASTCENC_SIMD_INLINE vint8 min(vint8 a, vint8 b) { a.m = _mm256_min_epi32(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vint8 max(vint8 a, vint8 b) { a.m = _mm256_max_epi32(a.m, b.m); return a; }

// Horizontal minimum - returns vector with all lanes
// set to the minimum value of the input vector.
ASTCENC_SIMD_INLINE vfloat8 hmin(vfloat8 v)
{
	__m128 vlow = _mm256_castps256_ps128(v.m);
	__m128 vhigh = _mm256_extractf128_ps(v.m, 1);
	vlow  = _mm_min_ps(vlow, vhigh);

	// First do an horizontal reduction.                                // v = [ D C | B A ]
	__m128 shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));  //     [ C D | A B ]
	__m128 mins = _mm_min_ps(vlow, shuf);                            // mins = [ D+C C+D | B+A A+B ]
	shuf        = _mm_movehl_ps(shuf, mins);                         //        [   C   D | D+C C+D ]
	mins        = _mm_min_ss(mins, shuf);


	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	//__m256i r = _mm256_set_m128(m, m)
	__m256 r = _mm256_insertf128_ps(_mm256_castps128_ps256(mins), mins, 1);

	vfloat8 vmin(_mm256_permute_ps(r, 0));
	return vmin;
}

ASTCENC_SIMD_INLINE vint8 hmin(vint8 v)
{
	__m128i m = _mm_min_epi32(_mm256_extracti128_si256(v.m, 0), _mm256_extracti128_si256(v.m, 1));
	m = _mm_min_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,3,2)));
	m = _mm_min_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,0,1)));
	m = _mm_shuffle_epi32(m, _MM_SHUFFLE(0,0,0,0));

	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	//__m256i r = _mm256_set_m128i(m, m)
	__m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(m), m, 1);
	vint8 vmin(r);
	return vmin;
}

// Store float vector into an aligned address.
ASTCENC_SIMD_INLINE void storea(vfloat8 v, float* ptr) { _mm256_store_ps(ptr, v.m); }

// Store integer vector into an aligned address.
ASTCENC_SIMD_INLINE void storea(vint8 v, int* ptr) { _mm256_store_si256((__m256i*)ptr, v.m); }

// Store lowest N (simd width) bytes of integer vector into an unaligned address.
ASTCENC_SIMD_INLINE void store_nbytes(vint8 v, uint8_t* ptr)
{
	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	// _mm_storeu_si64(ptr, _mm256_extracti128_si256(v.m, 0))
	_mm_storel_epi64((__m128i*)ptr, _mm256_extracti128_si256(v.m, 0));
}

// SIMD "gather" - load each lane with base[indices[i]]
ASTCENC_SIMD_INLINE vfloat8 gatherf(const float* base, vint8 indices)
{
	return vfloat8(_mm256_i32gather_ps(base, indices.m, 4));
}
ASTCENC_SIMD_INLINE vint8 gatheri(const int* base, vint8 indices)
{
	return vint8(_mm256_i32gather_epi32(base, indices.m, 4));
}

// Pack low 8 bits of each lane into low 64 bits of result.
ASTCENC_SIMD_INLINE vint8 pack_low_bytes(vint8 v)
{
	__m256i shuf = _mm256_set_epi8(0, 0, 0, 0,  0,  0,  0,  0,
	                               0, 0, 0, 0, 28, 24, 20, 16,
	                               0, 0, 0, 0,  0,  0,  0,  0,
	                               0, 0, 0, 0, 12,  8,  4,  0);
	__m256i a = _mm256_shuffle_epi8(v.m, shuf);
	__m128i a0 = _mm256_extracti128_si256(a, 0);
	__m128i a1 = _mm256_extracti128_si256(a, 1);
	__m128i b = _mm_unpacklo_epi32(a0, a1);

	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	//__m256i r = _mm256_set_m128i(b, b)
	__m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(b), b, 1);
	return vint8(r);
}

// "select", i.e. highbit(cond) ? b : a
ASTCENC_SIMD_INLINE vfloat8 select(vfloat8 a, vfloat8 b, vmask8 cond)
{
	return vfloat8(_mm256_blendv_ps(a.m, b.m, cond.m));
}
ASTCENC_SIMD_INLINE vint8 select(vint8 a, vint8 b, vmask8 cond)
{
	return vint8(_mm256_blendv_epi8(a.m, b.m, _mm256_castps_si256(cond.m)));
}

ASTCENC_SIMD_INLINE void print(vfloat8 a)
{
	alignas(ASTCENC_VECALIGN) float v[8];
	storea(a, v);
	printf("v8_f32:\n  %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f\n",
	       (double)v[0], (double)v[1], (double)v[2], (double)v[3],
	       (double)v[4], (double)v[5], (double)v[6], (double)v[7]);
}

ASTCENC_SIMD_INLINE void print(vint8 a)
{
	alignas(ASTCENC_VECALIGN) int v[8];
	storea(a, v);
	printf("v8_i32:\n  %8u %8u %8u %8u %8u %8u %8u %8u\n",
	       v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

#endif // #ifndef ASTC_VECMATHLIB_AVX2_8_H_INCLUDED
