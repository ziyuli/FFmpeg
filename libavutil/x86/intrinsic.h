/*
 * This file is part of FFmpeg.
 *
 * from https://ffmpeg.org/pipermail/ffmpeg-devel/2016-March/191167.html
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVUTIL_X86_INTRINSIC_H
#define AVUTIL_X86_INTRINSIC_H

#ifdef __GNUC__
#    define av_intrinsic_mmx __attribute__((__target__("mmx")))
#    define av_intrinsic_mmxext __attribute__((__target__("sse")))
#    define av_intrinsic_amd3dnow __attribute__((__target__("3dnow")))
#    define av_intrinsic_amd3dnowext __attribute__((__target__("3dnow")))
#    define av_intrinsic_sse __attribute__((__target__("sse")))
#    define av_intrinsic_sse2 __attribute__((__target__("sse2")))
#    define av_intrinsic_sse3 __attribute__((__target__("sse3")))
#    define av_intrinsic_ssse3 __attribute__((__target__("ssse3")))
#    define av_intrinsic_sse4 __attribute__((__target__("sse4.1")))
#    define av_intrinsic_sse42 __attribute__((__target__("sse4.2")))
#    define av_intrinsic_avx __attribute__((__target__("avx")))
#    define av_intrinsic_avx2 __attribute__((__target__("avx2")))
#    define av_intrinsic_fma3 __attribute__((__target__("fma")))
#    define av_intrinsic_fma4 __attribute__((__target__("fma4")))
#    define av_intrinsic_xop __attribute__((__target__("xop")))
#    define av_intrinsic_aesni __attribute__((__target__("aes")))
#else
#    define av_intrinsic_mmx
#    define av_intrinsic_mmxext
#    define av_intrinsic_amd3dnow
#    define av_intrinsic_amd3dnowext
#    define av_intrinsic_sse
#    define av_intrinsic_sse2
#    define av_intrinsic_sse3
#    define av_intrinsic_ssse3
#    define av_intrinsic_sse4
#    define av_intrinsic_sse42
#    define av_intrinsic_avx
#    define av_intrinsic_avx2
#    define av_intrinsic_fma
#    define av_intrinsic_fma4
#    define av_intrinsic_xop
#    define av_intrinsic_aesni
#endif

#endif /* AVUTIL_X86_INTRINSIC_H */