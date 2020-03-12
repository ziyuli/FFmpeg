#include "libavutil/x86/intrinsic.h"
#include "libavutil/common.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"

#if HAVE_MMX_INTRINSIC
#include <mmintrin.h>
#endif

#if HAVE_SSE_INTRINSIC
#include <xmmintrin.h>
#endif

#if HAVE_SSE2_INTRINSIC
#include <emmintrin.h>
#endif

#if HAVE_SSE3_INTRINSIC
#include <pmmintrin.h>
#endif

#if HAVE_SSSE3_INTRINSIC
#include <tmmintrin.h>
#endif

#if HAVE_SSE41_INTRINSIC
#include <smmintrin.h>
#endif

#if HAVE_SSE42_INTRINSIC
#include <nmmintrin.h>
#endif

#if HAVE_AVX_INTRINSIC
#include <immintrin.h>
#endif

typedef struct ScrollBiContext {
    const AVClass *class;

    float h_speed, v_speed;
    float h_pos, v_pos;
    float h_ipos, v_ipos;

    int pos_h[4], pos_v[4];

    const AVPixFmtDescriptor *desc;
    int nb_planes;
    int bytes;
    int chroma;

    int planewidth[4];
    int planeheight[4];
} ScrollBiContext;

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV440P,
        AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P,
        AV_PIX_FMT_YUVA422P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVA420P, AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P,
        AV_PIX_FMT_YUV420P9, AV_PIX_FMT_YUV422P9, AV_PIX_FMT_YUV444P9,
        AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV444P10,
        AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV440P12,
        AV_PIX_FMT_YUV420P14, AV_PIX_FMT_YUV422P14, AV_PIX_FMT_YUV444P14,
        AV_PIX_FMT_YUV420P16, AV_PIX_FMT_YUV422P16, AV_PIX_FMT_YUV444P16,
        AV_PIX_FMT_YUVA420P9, AV_PIX_FMT_YUVA422P9, AV_PIX_FMT_YUVA444P9,
        AV_PIX_FMT_YUVA422P12, AV_PIX_FMT_YUVA444P12,
        AV_PIX_FMT_YUVA420P10, AV_PIX_FMT_YUVA422P10, AV_PIX_FMT_YUVA444P10,
        AV_PIX_FMT_YUVA420P16, AV_PIX_FMT_YUVA422P16, AV_PIX_FMT_YUVA444P16,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRP9, AV_PIX_FMT_GBRP10,
        AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP14, AV_PIX_FMT_GBRP16,
        AV_PIX_FMT_GBRAP, AV_PIX_FMT_GBRAP10, AV_PIX_FMT_GBRAP12, AV_PIX_FMT_GBRAP16,
        AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9, AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12, AV_PIX_FMT_GRAY14, AV_PIX_FMT_GRAY16,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

typedef struct ThreadData {
    AVFrame *in, *out;
    uint8_t h_interp, v_interp;
} ThreadData;

static av_always_inline uint8_t lerp_u8(const uint8_t a, const uint8_t b, const uint8_t w) 
{
    const uint16_t w16 = (uint16_t)w;
    const uint16_t w16_ = 255 - w16;
    return (a * w16_ + b * w16) >> 8;
}

av_always_inline void bilinear_block8x8_kernel_sse(const uint8_t w_h, const uint8_t w_v,
    const uint8_t* restrict in_block, uint8_t* restrict out_block)
{
    __m128i i0, i1;
    __m128i pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7, pa8;
    __m128i pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7, pb8;
    __m128i pc0, pc1, pc2, pc3, pc4, pc5, pc6, pc7;

    __m128i w_h_i   = _mm_set1_epi16((uint16_t)w_h);
    __m128i w_v_i   = _mm_set1_epi16((uint16_t)w_v);
    __m128i w_h_c_i = _mm_set1_epi16(255 - (uint16_t)w_h);
    __m128i w_v_c_i = _mm_set1_epi16(255 - (uint16_t)w_v);

    _mm_prefetch(in_block, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 1, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 2, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 3, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 4, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 5, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 6, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 7, _MM_HINT_T0);
    _mm_prefetch(in_block + 9 * 8, _MM_HINT_T0);

    pa0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block)));
    pa1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 1)));
    pa2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 2)));
    pa3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 3)));
    pa4 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 4)));
    pa5 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 5)));
    pa6 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 6)));
    pa7 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 7)));
    pa8 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 9 * 8))); 

    pb0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1)));
    pb1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 1)));
    pb2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 2)));
    pb3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 3)));
    pb4 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 4)));
    pb5 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 5)));
    pb6 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 6)));
    pb7 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 7)));
    pb8 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(in_block + 1 + 9 * 8)));

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa0, w_h_c_i), _mm_mullo_epi16(pb0, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa1, w_h_c_i), _mm_mullo_epi16(pb1, w_h_i)), 8);
    pc0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa1, w_h_c_i), _mm_mullo_epi16(pb1, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa2, w_h_c_i), _mm_mullo_epi16(pb2, w_h_i)), 8);
    pc1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa2, w_h_c_i), _mm_mullo_epi16(pb2, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa3, w_h_c_i), _mm_mullo_epi16(pb3, w_h_i)), 8);
    pc2 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa3, w_h_c_i), _mm_mullo_epi16(pb3, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa4, w_h_c_i), _mm_mullo_epi16(pb4, w_h_i)), 8);
    pc3 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa4, w_h_c_i), _mm_mullo_epi16(pb4, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa5, w_h_c_i), _mm_mullo_epi16(pb5, w_h_i)), 8);
    pc4 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa5, w_h_c_i), _mm_mullo_epi16(pb5, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa6, w_h_c_i), _mm_mullo_epi16(pb6, w_h_i)), 8);
    pc5 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa6, w_h_c_i), _mm_mullo_epi16(pb6, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa7, w_h_c_i), _mm_mullo_epi16(pb7, w_h_i)), 8);
    pc6 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    i0  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa7, w_h_c_i), _mm_mullo_epi16(pb7, w_h_i)), 8);
    i1  = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(pa8, w_h_c_i), _mm_mullo_epi16(pb8, w_h_i)), 8);
    pc7 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(i0, w_v_c_i), _mm_mullo_epi16(i1, w_v_i)), 8);

    pc0 = _mm_packus_epi16(pc0, pc1);
    pc1 = _mm_packus_epi16(pc2, pc3);
    pc2 = _mm_packus_epi16(pc4, pc5);
    pc3 = _mm_packus_epi16(pc6, pc7);

    _mm_store_si128((void*)(out_block + 0),     pc0);
    _mm_store_si128((void*)(out_block + 8 * 2), pc1);
    _mm_store_si128((void*)(out_block + 8 * 4), pc2);
    _mm_store_si128((void*)(out_block + 8 * 6), pc3);
}

static int scroll_bilinear_slice(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    ScrollBiContext *s = ctx->priv;
    ThreadData *td = arg;   
    AVFrame *in = td->in;
    AVFrame *out = td->out;
    uint8_t w_h = td->h_interp;
    uint8_t w_v = td->v_interp;

    int yy, xx;

    uint8_t interp_h0, interp_h1, interp_v;
    uint8_t pixels[4];

    for (int p = 0; p < s->nb_planes; p++) {

        const uint8_t *src = in->data[p];
        const int h = s->planeheight[p];
        const int w = s->planewidth[p] * s->bytes;
        const int slice_start = (h * jobnr) / nb_jobs;
        const int slice_end = (h * (jobnr + 1)) / nb_jobs;
        uint8_t *dst = out->data[p] + slice_start * out->linesize[p];

        if (s->chroma || p == 0 || p == 3) {

            const int32_t slices = slice_end - slice_start;
            const int32_t y_nb = slices >> 3;
            const int32_t y_leftover = slices - (y_nb << 3);

            uint8_t in_block[81] __attribute__((aligned(16))) = { 0 };
            uint8_t out_block[64] __attribute__((aligned(16))) = { 0 };

            for (int i = 0; i < y_nb; ++i) {

                const int32_t x_nb = w >> 3;
                const int32_t x_leftover = w - (x_nb << 3);

                for (int j = 0; j < x_nb; ++j) {

                    int y = (i << 3) + s->pos_v[p] + slice_start;
                    int x = (j << 3) + s->pos_h[p];

                    for (int k = 0; k < 9; ++k) {
                        uint8_t* src_offset = src + FFMIN(y + k, h - 1) * in->linesize[p];
                        uint8_t* in_offset = in_block + k * 9;
                        memcpy(in_offset, src_offset + x, 8);
                        in_offset[8] = src_offset[FFMIN(x + 8, w - 1)];
                    }

                    bilinear_block8x8_kernel_sse(w_h, w_v, in_block, out_block); 

                    for (int k = 0; k < 8; ++k) {
                        memcpy(dst + (j << 3) + (k + (i << 3)) * out->linesize[p], out_block + 8 * k, 8);
                    }
                }

    			// x leftover
    			for (int ii = 0; ii < 9; ++ii) {

                    int y = FFMIN((i << 3) + ii + s->pos_v[p] + slice_start, h - 1);
                    int x = FFMIN((x_nb << 3) + s->pos_h[p], w - 1);

                    uint8_t* src_offset = src + y * in->linesize[p] + x;
                    uint8_t* in_offset = in_block + ii * 9;

                    memcpy(in_offset, src_offset, x_leftover);
                    memset(in_offset + x_leftover, 0, 8 - x_leftover);
                }

                bilinear_block8x8_kernel_sse(w_h, w_v, in_block, out_block); 

                for (int k = 0; k < 8; ++k) {
                    memcpy(dst + (x_nb << 3) + (k + (i << 3)) * out->linesize[p], out_block + 8 * k, x_leftover);
                }
            } 

            // y leftover
            for (int i = 0; i < y_leftover; ++i) {
                for (int j = 0; j < w; ++j) {

                    int y = slice_start + (y_nb << 3) + i;
                    int x = j;

                    for (int by = 0; by < 2; by++) {
                        for (int bx = 0; bx < 2; bx++) {
                            yy = FFMIN(y + by + s->pos_v[p], h - 1);
                            xx = FFMIN(x + bx + s->pos_h[p], w - 1);
                            pixels[bx + by * 2] = src[yy * in->linesize[p] + xx];
                        }
                    }

                    interp_h0 = lerp_u8(pixels[0], pixels[1], w_h);
                    interp_h1 = lerp_u8(pixels[2], pixels[3], w_h);
                    interp_v  = lerp_u8(interp_h0, interp_h1, w_v);
                    
                    dst[x + ((y_nb << 3) + i) * out->linesize[p]] = interp_v;
                }
            } 

        } else {
            for (int y = slice_start; y < slice_end; y++) {
                yy = FFMIN(y + s->pos_v[p], h - 1);
                const uint8_t *ssrc = src + yy * in->linesize[p];

                if (s->pos_h[p] < w)
                    memcpy(dst, ssrc + s->pos_h[p], w - s->pos_h[p]);
                if (s->pos_h[p] > 0)
                    memcpy(dst + w - s->pos_h[p], ssrc, s->pos_h[p]);

                dst += out->linesize[p];
            }
        }
    }

    return 0;
}

static void scroll_bilinear(AVFilterContext *ctx, AVFrame *in, AVFrame *out)
{
    ScrollBiContext *s = ctx->priv;
    ThreadData td;
    int h_pos, v_pos;
    float h_interp, v_interp;

    s->h_pos = fmodf(s->h_pos, in->width);
    s->v_pos = fmodf(s->v_pos, in->height);

    h_interp = fmodf(s->h_pos, 1);
    v_interp = fmodf(s->v_pos, 1);

    h_pos = s->h_pos;
    v_pos = s->v_pos;

    if (h_pos < 0)
        h_pos += in->width;
    if (v_pos < 0)
        v_pos += in->height;

    s->pos_v[1] = s->pos_v[2] = AV_CEIL_RSHIFT(v_pos, s->desc->log2_chroma_h);
    s->pos_v[0] = s->pos_v[3] = v_pos;
    s->pos_h[1] = s->pos_h[2] = AV_CEIL_RSHIFT(h_pos, s->desc->log2_chroma_w) * s->bytes;
    s->pos_h[0] = s->pos_h[3] = h_pos * s->bytes;

    td.in = in; 
    td.out = out;
    td.h_interp = (uint8_t)(h_interp * 255); 
    td.v_interp = (uint8_t)(v_interp * 255);

    int max_threads = ceil(out->height * 0.125f);

    ctx->internal->execute(ctx, scroll_bilinear_slice, &td, NULL, 
        FFMIN(max_threads, ff_filter_get_nb_threads(ctx)));

    s->h_pos += s->h_speed * in->width;
    s->v_pos += s->v_speed * in->height;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    scroll_bilinear(ctx, in, out);

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    ScrollBiContext *s = ctx->priv;

    s->desc = av_pix_fmt_desc_get(inlink->format);
    s->nb_planes = s->desc->nb_components;
    s->bytes = (s->desc->comp[0].depth + 7) >> 3;

    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, s->desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;
    s->planewidth[1]  = s->planewidth[2]  = AV_CEIL_RSHIFT(inlink->w, s->desc->log2_chroma_w);
    s->planewidth[0]  = s->planewidth[3]  = inlink->w;

    if (s->chroma == -1) {
        // Force Interp. Only On Luma
        s->chroma = 0;
    } else if (s->chroma == 0) {
        // Auto 
        if (s->planeheight[0] == s->planeheight[1] && 
            s->planeheight[0] == s->planeheight[2]) {
            s->chroma = 1;
        }
    }


    s->h_pos = (1.f - s->h_ipos) * inlink->w;
    s->v_pos = (1.f - s->v_ipos) * inlink->h;

    return 0;
}

#define OFFSET(x) offsetof(ScrollBiContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define VFT AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_RUNTIME_PARAM

static const AVOption scroll_bi_options[] = {
    { "horizontal", "set the horizontal scrolling speed", OFFSET(h_speed), AV_OPT_TYPE_FLOAT, {.dbl=0.}, -1., 1., VFT },
    { "h",          "set the horizontal scrolling speed", OFFSET(h_speed), AV_OPT_TYPE_FLOAT, {.dbl=0.}, -1., 1., VFT },
    { "vertical",   "set the vertical scrolling speed",   OFFSET(v_speed), AV_OPT_TYPE_FLOAT, {.dbl=0.}, -1., 1., VFT },
    { "v",          "set the vertical scrolling speed",   OFFSET(v_speed), AV_OPT_TYPE_FLOAT, {.dbl=0.}, -1., 1., VFT },
    { "hpos",       "set initial horizontal position",    OFFSET(h_ipos),  AV_OPT_TYPE_FLOAT, {.dbl=0.},   0, 1., FLAGS },
    { "vpos",       "set initial vertical position",      OFFSET(v_ipos),  AV_OPT_TYPE_FLOAT, {.dbl=0.},   0, 1., FLAGS },
    { "chroma",     "set full chroma interpolation",      OFFSET(chroma),  AV_OPT_TYPE_INT,   {.dbl=0.},  -1, 1 , FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(scroll_bi);

static const AVFilterPad scroll_bi_inputs[] = {
    {
        .name           = "default",
        .type           = AVMEDIA_TYPE_VIDEO,
        .config_props   = config_input,
        .filter_frame   = filter_frame,
    },
    { NULL }
};

static const AVFilterPad scroll_bi_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_scroll_bi = {
    .name          = "scrollbi",
    .description   = NULL_IF_CONFIG_SMALL("Scroll (Bilinear) input video."),
    .priv_size     = sizeof(ScrollBiContext),
    .priv_class    = &scroll_bi_class,
    .query_formats = query_formats,
    .inputs        = scroll_bi_inputs,
    .outputs       = scroll_bi_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC | AVFILTER_FLAG_SLICE_THREADS,
    .process_command = ff_filter_process_command,
};
