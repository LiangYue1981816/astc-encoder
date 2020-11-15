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

#ifndef ASTC_VECMATHLIB_SSE_4_H_INCLUDED
#define ASTC_VECMATHLIB_SSE_4_H_INCLUDED

#ifndef ASTCENC_SIMD_INLINE
	#error "Include astcenc_vecmathlib.h, do not include directly"
#endif

#include <cstdio>

struct vfloat4
{
	ASTCENC_SIMD_INLINE vfloat4() {}

	ASTCENC_SIMD_INLINE explicit vfloat4(const float *p) { m = _mm_loadu_ps(p); }

	ASTCENC_SIMD_INLINE explicit vfloat4(float v) { m = _mm_set_ps1(v); }

	ASTCENC_SIMD_INLINE explicit vfloat4(float p, float q, float s, float t) { m = _mm_set_ps(t, s, q, p); }

	ASTCENC_SIMD_INLINE explicit vfloat4(__m128 v) { m = v; }

	ASTCENC_SIMD_INLINE float lane(int i) const
	{
		#ifdef _MSC_VER
		return m.m128_f32[i];
		#else
		union { __m128 m; float f[4]; } cvt;
		cvt.m = m;
		return cvt.f[i];
		#endif
	}

	template <int i> ASTCENC_SIMD_INLINE void set_lane(float v)
	{
		assert(i < 4);
		__m128 nv = _mm_set_ps1(v);
		m = _mm_insert_ps(m, nv, i << 6 | i << 4);
	}

	ASTCENC_SIMD_INLINE float r() const
	{
		return lane(0);
	}

	ASTCENC_SIMD_INLINE float g() const
	{
		return lane(1);
	}

	ASTCENC_SIMD_INLINE float b() const
	{
		return lane(2);
	}

	ASTCENC_SIMD_INLINE float a() const
	{
		return lane(3);
	}

	static ASTCENC_SIMD_INLINE vfloat4 zero() { return vfloat4(_mm_setzero_ps()); }

	static ASTCENC_SIMD_INLINE vfloat4 lane_id() { return vfloat4(_mm_set_ps(3, 2, 1, 0)); }

	__m128 m;
};

struct vint4
{
	ASTCENC_SIMD_INLINE vint4() {}
	ASTCENC_SIMD_INLINE explicit vint4(const int *p) { m = _mm_load_si128((const __m128i*)p); }
	ASTCENC_SIMD_INLINE explicit vint4(int v) { m = _mm_set1_epi32(v); }
	ASTCENC_SIMD_INLINE explicit vint4(__m128i v) { m = v; }
	ASTCENC_SIMD_INLINE int lane(int i) const
	{
		#ifdef _MSC_VER
		return m.m128i_i32[i];
		#else
		union { __m128i m; int f[4]; } cvt;
		cvt.m = m;
		return cvt.f[i];
		#endif
	}
	static ASTCENC_SIMD_INLINE vint4 lane_id() { return vint4(_mm_set_epi32(3, 2, 1, 0)); }
	__m128i m;
};

struct vmask4
{
	ASTCENC_SIMD_INLINE explicit vmask4(__m128 v) { m = v; }
	ASTCENC_SIMD_INLINE explicit vmask4(__m128i v) { m = _mm_castsi128_ps(v); }
	__m128 m;
};


ASTCENC_SIMD_INLINE vfloat4 load1a_4f(const float* p) { return vfloat4(_mm_load_ps1(p)); }
ASTCENC_SIMD_INLINE vfloat4 loada_4f(const float* p) { return vfloat4(_mm_load_ps(p)); }

ASTCENC_SIMD_INLINE vfloat4 operator+ (vfloat4 a, vfloat4 b) { a.m = _mm_add_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat4 operator- (vfloat4 a, vfloat4 b) { a.m = _mm_sub_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat4 operator* (vfloat4 a, vfloat4 b) { a.m = _mm_mul_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat4 operator* (vfloat4 a, float b) { a.m = _mm_mul_ps(a.m, _mm_set_ps1(b)); return a; }
ASTCENC_SIMD_INLINE vfloat4 operator* (float a, vfloat4 b) { b.m = _mm_mul_ps(_mm_set_ps1(a), b.m); return b; }
ASTCENC_SIMD_INLINE vfloat4 operator/ (vfloat4 a, vfloat4 b) { a.m = _mm_div_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vmask4 operator==(vfloat4 a, vfloat4 b) { return vmask4(_mm_cmpeq_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator!=(vfloat4 a, vfloat4 b) { return vmask4(_mm_cmpneq_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator< (vfloat4 a, vfloat4 b) { return vmask4(_mm_cmplt_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator> (vfloat4 a, vfloat4 b) { return vmask4(_mm_cmpgt_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator<=(vfloat4 a, vfloat4 b) { return vmask4(_mm_cmple_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator>=(vfloat4 a, vfloat4 b) { return vmask4(_mm_cmpge_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator| (vmask4 a, vmask4 b) { return vmask4(_mm_or_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator& (vmask4 a, vmask4 b) { return vmask4(_mm_and_ps(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator^ (vmask4 a, vmask4 b) { return vmask4(_mm_xor_ps(a.m, b.m)); }

// Returns a 4-bit code where bit0..bit3 is X..W
ASTCENC_SIMD_INLINE unsigned mask(vmask4 v) { return _mm_movemask_ps(v.m); }
ASTCENC_SIMD_INLINE bool any(vmask4 v) { return mask(v) != 0; }
ASTCENC_SIMD_INLINE bool all(vmask4 v) { return mask(v) == 0xF; }

ASTCENC_SIMD_INLINE vfloat4 min(vfloat4 a, vfloat4 b) { a.m = _mm_min_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat4 max(vfloat4 a, vfloat4 b) { a.m = _mm_max_ps(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vfloat4 saturate(vfloat4 a)
{
	__m128 zero = _mm_setzero_ps();
	__m128 one = _mm_set1_ps(1.0f);
	return vfloat4(_mm_min_ps(_mm_max_ps(a.m, zero), one));
}

ASTCENC_SIMD_INLINE vfloat4 abs(vfloat4 x)
{
	__m128 msk = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
	return vfloat4(_mm_and_ps(x.m, msk));
}

ASTCENC_SIMD_INLINE vfloat4 round(vfloat4 v)
{
#if ASTCENC_SSE >= 41
	return vfloat4(_mm_round_ps(v.m, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
	__m128 V = v.m;
	__m128 negZero = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	__m128 noFraction = _mm_set_ps1(8388608.0f);
	__m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
	__m128 sign = _mm_and_ps(V, negZero);
	__m128 sMagic = _mm_or_ps(noFraction, sign);
	__m128 R1 = _mm_add_ps(V, sMagic);
	R1 = _mm_sub_ps(R1, sMagic);
	__m128 R2 = _mm_and_ps(V, absMask);
	__m128 mask = _mm_cmple_ps(R2, noFraction);
	R2 = _mm_andnot_ps(mask, V);
	R1 = _mm_and_ps(R1, mask);
	return vfloat4(_mm_xor_ps(R1, R2));
#endif
}

ASTCENC_SIMD_INLINE vint4 float_to_int(vfloat4 v) { return vint4(_mm_cvttps_epi32(v.m)); }

ASTCENC_SIMD_INLINE vfloat4 int_as_float(vint4 v) { return vfloat4(_mm_castsi128_ps(v.m)); }
ASTCENC_SIMD_INLINE vint4 float_as_int(vfloat4 v) { return vint4(_mm_castps_si128(v.m)); }

ASTCENC_SIMD_INLINE vint4 operator~ (vint4 a) { return vint4(_mm_xor_si128(a.m, _mm_set1_epi32(-1))); }
ASTCENC_SIMD_INLINE vmask4 operator~ (vmask4 a) { return vmask4(_mm_xor_si128(_mm_castps_si128(a.m), _mm_set1_epi32(-1))); }

ASTCENC_SIMD_INLINE vint4 operator+ (vint4 a, vint4 b) { a.m = _mm_add_epi32(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vint4 operator- (vint4 a, vint4 b) { a.m = _mm_sub_epi32(a.m, b.m); return a; }
ASTCENC_SIMD_INLINE vint4 operator| (vint4 a, vint4 b) { return vint4(_mm_or_si128(a.m, b.m)); }
ASTCENC_SIMD_INLINE vint4 operator& (vint4 a, vint4 b) { return vint4(_mm_and_si128(a.m, b.m)); }
ASTCENC_SIMD_INLINE vint4 operator^ (vint4 a, vint4 b) { return vint4(_mm_xor_si128(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator< (vint4 a, vint4 b) { return vmask4(_mm_cmplt_epi32(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator> (vint4 a, vint4 b) { return vmask4(_mm_cmpgt_epi32(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator==(vint4 a, vint4 b) { return vmask4(_mm_cmpeq_epi32(a.m, b.m)); }
ASTCENC_SIMD_INLINE vmask4 operator!=(vint4 a, vint4 b) { return ~vmask4(_mm_cmpeq_epi32(a.m, b.m)); }

ASTCENC_SIMD_INLINE vint4 min(vint4 a, vint4 b) {
#if ASTCENC_SSE >= 41
	a.m = _mm_min_epi32(a.m, b.m);
#else
	vmask4 d = a < b;
	a.m = _mm_or_si128(_mm_and_si128(_mm_castps_si128(d.m), a.m), _mm_andnot_si128(_mm_castps_si128(d.m), b.m));
#endif
	return a;
}

ASTCENC_SIMD_INLINE vint4 max(vint4 a, vint4 b) {
#if ASTCENC_SSE >= 41
	a.m = _mm_max_epi32(a.m, b.m);
#else
	vmask4 d = a > b;
	a.m = _mm_or_si128(_mm_and_si128(_mm_castps_si128(d.m), a.m), _mm_andnot_si128(_mm_castps_si128(d.m), b.m));
#endif
	return a;
}

#define ASTCENC_SHUFFLE4F(V, X, Y, Z, W) vfloat4(_mm_shuffle_ps((V).m, (V).m, _MM_SHUFFLE(W,Z,Y,X)))
#define ASTCENC_SHUFFLE4I(V, X, Y, Z, W) vint4(_mm_shuffle_epi32((V).m, _MM_SHUFFLE(W,Z,Y,X)))

ASTCENC_SIMD_INLINE vfloat4 hmin(vfloat4 v)
{
	v = min(v, ASTCENC_SHUFFLE4F(v, 2, 3, 0, 0));
	v = min(v, ASTCENC_SHUFFLE4F(v, 1, 0, 0, 0));
	return ASTCENC_SHUFFLE4F(v, 0, 0, 0, 0);
}

ASTCENC_SIMD_INLINE vint4 hmin(vint4 v)
{
	v = min(v, ASTCENC_SHUFFLE4I(v, 2, 3, 0, 0));
	v = min(v, ASTCENC_SHUFFLE4I(v, 1, 0, 0, 0));
	return ASTCENC_SHUFFLE4I(v, 0, 0, 0, 0);
}

ASTCENC_SIMD_INLINE float dot(vfloat4 p, vfloat4 q)  {
#if (ASTCENC_SSE >= 42) && (ASTCENC_ISA_INVARIANCE == 0)
	return _mm_cvtss_f32(_mm_dp_ps(p.m, q.m, 0xFF));
#else
	vfloat4 res = p * q;
	return res.r() + res.g() + res.b() + res.a();
#endif
}


ASTCENC_SIMD_INLINE vfloat4 normalize(vfloat4 p) {
	// TODO: Provide a fallback for this one
	float len = astc::rsqrt(dot(p, p));
	p.set_lane<0>(p.r() * len);
	p.set_lane<1>(p.g() * len);
	p.set_lane<2>(p.b() * len);
	p.set_lane<3>(p.a() * len);
	return p;
	// TODO: This introduces rounding differences
	//__m128 len = _mm_rsqrt_ps(_mm_dp_ps(p.m, p.m, 0xFF));
	//return vfloat4(_mm_mul_ps(p.m, len));
}

ASTCENC_SIMD_INLINE vfloat4 sqrt(vfloat4 p) {
#if ASTCENC_SSE >= 20
	return vfloat4(_mm_sqrt_ps(p.m));
#else
	return vfloat4(std::sqrt(p.r), std::sqrt(p.g), std::sqrt(p.b), std::sqrt(p.a));
#endif
}

ASTCENC_SIMD_INLINE void store(vfloat4 v, float* ptr) { _mm_store_ps(ptr, v.m); }
ASTCENC_SIMD_INLINE void store(vint4 v, int* ptr) { _mm_store_si128((__m128i*)ptr, v.m); }

ASTCENC_SIMD_INLINE void store_nbytes(vint4 v, uint8_t* ptr)
{
	// This is the most logical implementation, but the convenience intrinsic
	// is missing on older compilers (supported in g++ 9 and clang++ 9).
	// _mm_storeu_si32(ptr, v.m);
	_mm_store_ss((float*)ptr, _mm_castsi128_ps(v.m));
}

ASTCENC_SIMD_INLINE vfloat4 gatherf(const float* base, vint4 indices)
{
	int idx[4];
	store(indices, idx);
	return vfloat4(_mm_set_ps(base[idx[3]], base[idx[2]], base[idx[1]], base[idx[0]]));
}

ASTCENC_SIMD_INLINE vint4 gatheri(const int* base, vint4 indices)
{
	int idx[4];
	store(indices, idx);
	return vint4(_mm_set_epi32(base[idx[3]], base[idx[2]], base[idx[1]], base[idx[0]]));
}

// packs low 8 bits of each lane into low 32 bits of result
ASTCENC_SIMD_INLINE vint4 pack_low_bytes(vint4 v)
{
	#if ASTCENC_SSE >= 41
	__m128i shuf = _mm_set_epi8(0,0,0,0, 0,0,0,0, 0,0,0,0, 12,8,4,0);
	return vint4(_mm_shuffle_epi8(v.m, shuf));
	#else
	__m128i va = _mm_unpacklo_epi8(v.m, _mm_shuffle_epi32(v.m, _MM_SHUFFLE(1,1,1,1)));
	__m128i vb = _mm_unpackhi_epi8(v.m, _mm_shuffle_epi32(v.m, _MM_SHUFFLE(3,3,3,3)));
	return vint4(_mm_unpacklo_epi16(va, vb));
	#endif
}

// "select", i.e. highbit(cond) ? b : a
// on SSE4.1 and up this can be done easily via "blend" instruction;
// on older SSEs we have to do some hoops, see
// https://fgiesen.wordpress.com/2016/04/03/sse-mind-the-gap/
ASTCENC_SIMD_INLINE vfloat4 select(vfloat4 a, vfloat4 b, vmask4 cond)
{
#if ASTCENC_SSE >= 41
	a.m = _mm_blendv_ps(a.m, b.m, cond.m);
#else
	__m128 d = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(cond.m), 31));
	a.m = _mm_or_ps(_mm_and_ps(d, b.m), _mm_andnot_ps(d, a.m));
#endif
	return a;
}

ASTCENC_SIMD_INLINE vint4 select(vint4 a, vint4 b, vmask4 cond)
{
#if ASTCENC_SSE >= 41
	return vint4(_mm_blendv_epi8(a.m, b.m, _mm_castps_si128(cond.m)));
#else
	__m128i d = _mm_srai_epi32(_mm_castps_si128(cond.m), 31);
	return vint4(_mm_or_si128(_mm_and_si128(d, b.m), _mm_andnot_si128(d, a.m)));
#endif
}

ASTCENC_SIMD_INLINE void print(vfloat4 a)
{
	alignas(ASTCENC_VECALIGN) float v[4];
	store(a, v);
	printf("v4_f32:\n  %0.4f %0.4f %0.4f %0.4f\n",
	       (double)v[0], (double)v[1], (double)v[2], (double)v[3]);
}

ASTCENC_SIMD_INLINE void print(vint4 a)
{
	alignas(ASTCENC_VECALIGN) int v[4];
	store(a, v);
	printf("v4_i32:\n  %8u %8u %8u %8u\n",
	       v[0], v[1], v[2], v[3]);
}

#endif // #ifndef ASTC_VECMATHLIB_SSE_4_H_INCLUDED
