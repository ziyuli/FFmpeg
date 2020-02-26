#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include "libavutil/avassert.h"
#include "libavutil/eval.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

static const char *const var_names[] = {
    "in_w",   "iw",
    "in_h",   "ih",
    "out_w",  "ow",
    "out_h",  "oh",
    "in",
    "on",
    "duration",
    "pduration",
    "time",
    "frame",
    "zoom",
    "pzoom",
    "x", "px",
    "y", "py",
    "a",
    "sar",
    "dar",
    "hsub",
    "vsub",
    NULL
};

enum var_name {
    VAR_IN_W,   VAR_IW,
    VAR_IN_H,   VAR_IH,
    VAR_OUT_W,  VAR_OW,
    VAR_OUT_H,  VAR_OH,
    VAR_IN,
    VAR_ON,
    VAR_DURATION,
    VAR_PDURATION,
    VAR_TIME,
    VAR_FRAME,
    VAR_ZOOM,
    VAR_PZOOM,
    VAR_X, VAR_PX,
    VAR_Y, VAR_PY,
    VAR_A,
    VAR_SAR,
    VAR_DAR,
    VAR_HSUB,
    VAR_VSUB,
    VARS_NB
};

typedef struct ZPBicontext {
    const AVClass *class;
    char *zoom_expr_str;
    char *x_expr_str;
    char *y_expr_str;
    char *duration_expr_str;

    AVExpr *zoom_expr, *x_expr, *y_expr;
    int planeheight[4], planewidth[4];
    int bytes;
    int nb_planes;
    float dsw, dsh;

    int w, h;
    int chroma;
    double x, y;
    double prev_zoom;
    int prev_nb_frames;
    int64_t frame_count;
    const AVPixFmtDescriptor *desc;
    AVFrame *in;
    double var_values[VARS_NB];
    int nb_frames;
    int current_frame;
    int finished;
    AVRational framerate;
} ZPBicontext;

#define OFFSET(x) offsetof(ZPBicontext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
static const AVOption zoompan_bi_options[] = {
    { "zoom", "set the zoom expression", OFFSET(zoom_expr_str), AV_OPT_TYPE_STRING, {.str = "1" }, .flags = FLAGS },
    { "z", "set the zoom expression", OFFSET(zoom_expr_str), AV_OPT_TYPE_STRING, {.str = "1" }, .flags = FLAGS },
    { "x", "set the x expression", OFFSET(x_expr_str), AV_OPT_TYPE_STRING, {.str="0"}, .flags = FLAGS },
    { "y", "set the y expression", OFFSET(y_expr_str), AV_OPT_TYPE_STRING, {.str="0"}, .flags = FLAGS },
    { "d", "set the duration expression", OFFSET(duration_expr_str), AV_OPT_TYPE_STRING, {.str="90"}, .flags = FLAGS },
    { "s", "set the output image size", OFFSET(w), AV_OPT_TYPE_IMAGE_SIZE, {.str="hd720"}, .flags = FLAGS },
    { "chroma", "set full chroma interpolation", OFFSET(chroma), AV_OPT_TYPE_INT, {.dbl=0}, 0, 1, .flags = FLAGS },
    { "fps", "set the output framerate", OFFSET(framerate), AV_OPT_TYPE_VIDEO_RATE, { .str = "30" }, 0, INT_MAX, .flags = FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(zoompan_bi);

static av_cold int init(AVFilterContext *ctx)
{
    ZPBicontext *s = ctx->priv;

    s->prev_zoom = 1;
    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    ZPBicontext *s = ctx->priv;
    int ret;

    outlink->w = s->w;
    outlink->h = s->h;
    outlink->time_base = av_inv_q(s->framerate);
    outlink->frame_rate = s->framerate;
    s->desc = av_pix_fmt_desc_get(outlink->format);
    s->nb_planes = s->desc->nb_components;
    s->bytes = (s->desc->comp[0].depth + 7) >> 3;
    s->finished = 1;

    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(outlink->h, s->desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = outlink->h;
    s->planewidth[1]  = s->planewidth[2]  = AV_CEIL_RSHIFT(outlink->w, s->desc->log2_chroma_w);
    s->planewidth[0]  = s->planewidth[3]  = outlink->w;

    s->dsh = 1.0f / (1 << s->desc->log2_chroma_h);
    s->dsw = 1.0f / (1 << s->desc->log2_chroma_w);

    if (s->planeheight[0] == s->planeheight[1] && s->planeheight[0] == s->planeheight[2]) {
        s->chroma = 1;
    }

    ret = av_expr_parse(&s->zoom_expr, s->zoom_expr_str, var_names, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;

    ret = av_expr_parse(&s->x_expr, s->x_expr_str, var_names, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;

    ret = av_expr_parse(&s->y_expr, s->y_expr_str, var_names, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;

    return 0;
}

typedef struct ThreadData {
    AVFrame *in, *out;
    float* px, *py, *ww, *wh;
    int full_chroma;
} ThreadData;

static inline uint8_t _lerp_u8(const uint8_t a, const uint8_t b, const float w) 
{
    return a * (1 - w) + b * w;
}

static inline __m128i _mm_lerp_ep16(const __m128i a, const __m128i b, const __m128i w, const __m128i w_c) 
{
    const __m128i lp  = _mm_add_epi16(_mm_mullo_epi16(a, w_c), _mm_mullo_epi16(b, w));
    return _mm_srli_epi16(lp, 8);
}

static int zoom_slice(AVFilterContext *ctx, void *arg, int jobnr,
                      int nb_jobs) 
{
    ZPBicontext *s = ctx->priv;
    ThreadData *td = arg;
    AVFrame *in = td->in;
    AVFrame *out = td->out;
    float *px = td->px;
    float *py = td->py;
    float *ww = td->ww;
    float *wh = td->wh;
    int chroma = td->full_chroma;

    float xx, yy;
    int ixx, iyy;
    int xx_int, yy_int;

    float h_interp_f, v_interp_f;
    uint8_t h_i0_u8, h_i1_u8, v_i_u8;
    uint8_t p0_u8, p1_u8, p2_u8, p3_u8;

    __m128i h_interp, v_interp;
    __m128i h_i0, h_i1, v_i;
    __m128i p0, p1, p2, p3;

    uint16_t h_interp_arr[8], v_interp_arr[8];
    uint16_t p0_arr[8], p1_arr[8], p2_arr[8], p3_arr[8];

    for (int p = 0; p < s->nb_planes; p++) {
        const uint8_t *src = in->data[p];
        const int h = s->planeheight[p];
        const int w = s->planewidth[p] * s->bytes;
        const int slice_start = (h * jobnr) / nb_jobs;
        const int slice_end = (h * (jobnr+1)) / nb_jobs;
        const float v_step = wh[p] / h;
        const float h_step = ww[p] / w;
        uint8_t *dst = out->data[p];

        for (int y = slice_start; y < slice_end; y++) {

            int x = 0;
            for (x = 0; x < w; x += 8) {

                if (chroma || p == 0 || p == 3) {

                    for (int i = 0; i < 8; i++) {
                        yy = py[p] + y * v_step;
                        xx = px[p] + (x + i) * h_step;

                        xx_int = xx;
                        yy_int = yy;

                        ixx = FFMIN(xx_int + 1, w);
                        iyy = FFMIN(yy_int + 1, h);

                        h_interp_arr[i] = (xx - (int)xx) * 256;
                        v_interp_arr[i] = (yy - (int)yy) * 256;

                        p0_arr[i] = (uint16_t)src[xx_int + yy_int * in->linesize[p]];
                        p1_arr[i] = (uint16_t)src[ixx + yy_int * in->linesize[p]];
                        p2_arr[i] = (uint16_t)src[xx_int + iyy * in->linesize[p]];
                        p3_arr[i] = (uint16_t)src[ixx + iyy * in->linesize[p]];
                    }

                    h_interp = _mm_load_si128((__m128i*)h_interp_arr);
                    v_interp = _mm_load_si128((__m128i*)v_interp_arr);

                    __m128i h_interp_c = _mm_sub_epi16(_mm_set1_epi16(256), h_interp);
                    __m128i v_interp_c = _mm_sub_epi16(_mm_set1_epi16(256), v_interp);

                    p0 = _mm_load_si128((__m128i*)p0_arr);
                    p1 = _mm_load_si128((__m128i*)p1_arr);
                    p2 = _mm_load_si128((__m128i*)p2_arr);
                    p3 = _mm_load_si128((__m128i*)p3_arr);

                    h_i0 = _mm_lerp_ep16(p0, p1, h_interp, h_interp_c);
                    h_i1 = _mm_lerp_ep16(p2, p3, h_interp, h_interp_c);
                    v_i  = _mm_lerp_ep16(h_i0, h_i1, v_interp, v_interp_c);

                    v_i = _mm_packus_epi16(v_i, _mm_setzero_si128());
                    _mm_storeu_si128((__m128i*)&dst[x + y * out->linesize[p]], v_i);

                } else {
                    for (int i = 0; i < 8; i++) {
                        xx = px[p] + (x + i) * h_step;
                        yy = py[p] + y * v_step;
                        dst[i + x + y * out->linesize[p]] = src[(int)xx + (int)yy * in->linesize[p]];
                    }
                }
            }

            int leftover = w - x;

            for (int i = 0; i < leftover; ++i) {
                if (chroma || p == 0 || p == 3) { 
                    yy = py[p] + y * v_step;
                    xx = px[p] + (x + i) * h_step;

                    xx_int = xx;
                    yy_int = yy;

                    ixx = FFMIN(xx_int + 1, w);
                    iyy = FFMIN(yy_int + 1, h);

                    h_interp_f = fmodf(xx, 1);
                    v_interp_f = fmodf(yy, 1);

                    p0_u8 = src[xx_int + yy_int * in->linesize[p]];
                    p1_u8 = src[ixx + yy_int * in->linesize[p]];
                    p2_u8 = src[xx_int + iyy * in->linesize[p]];
                    p3_u8 = src[ixx + iyy * in->linesize[p]];

                    h_i0_u8 = _lerp_u8(p0_u8, p1_u8, h_interp_f);
                    h_i1_u8 = _lerp_u8(p2_u8, p3_u8, h_interp_f);
                    v_i_u8  = _lerp_u8(h_i0_u8, h_i1_u8, v_interp_f);

                    dst[i + x + y * out->linesize[p]] = v_i_u8;
                } else {
                    xx = px[p] + (x + i) * h_step;
                    yy = py[p] + y * v_step;

                    dst[i + x + y * out->linesize[p]] = src[(int)xx + (int)yy * in->linesize[p]];
                }
            }
        } 
    } 

    return 0;
}

static int output_single_frame(AVFilterContext *ctx, AVFrame *in, double *var_values, int i,
                               double *zoom, double *dx, double *dy)
{
    ZPBicontext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int64_t pts = s->frame_count;
    int x, y, w, h, ret = 0;
    float fx, fy, fw, fh;
    float px[4], py[4], ww[4], wh[4];
    AVFrame *out;
    ThreadData td;

    var_values[VAR_PX]    = s->x;
    var_values[VAR_PY]    = s->y;
    var_values[VAR_PZOOM] = s->prev_zoom;
    var_values[VAR_PDURATION] = s->prev_nb_frames;
    var_values[VAR_TIME] = pts * av_q2d(outlink->time_base);
    var_values[VAR_FRAME] = i;
    var_values[VAR_ON] = outlink->frame_count_in;

    *zoom = av_expr_eval(s->zoom_expr, var_values, NULL);

    *zoom = av_clipd(*zoom, 1, 10);
    var_values[VAR_ZOOM] = *zoom;
    fw = in->width * (1.0 / *zoom);
    fh = in->height * (1.0 / *zoom);
    w = fw;
    h = fh;

    *dx = av_expr_eval(s->x_expr, var_values, NULL);

    fx = *dx = av_clipd(*dx, 0, FFMAX(in->width - fw, 0));
    var_values[VAR_X] = *dx;
    x = ((int)fx) & ~((1 << s->desc->log2_chroma_w) - 1);

    *dy = av_expr_eval(s->y_expr, var_values, NULL);

    fy = *dy = av_clipd(*dy, 0, FFMAX(in->height - fh, 0));
    var_values[VAR_Y] = *dy;
    y = ((int)fy) & ~((1 << s->desc->log2_chroma_h) - 1);

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        ret = AVERROR(ENOMEM);
        return ret;
    }

    ww[1] = ww[2] = fw * s->dsw;
    ww[0] = ww[3] = fw;

    wh[1] = wh[2] = fh * s->dsh;
    wh[0] = wh[3] = fh;

    px[1] = px[2] = fx * s->dsw;
    px[0] = px[3] = fx;

    py[1] = py[2] = fy * s->dsh;
    py[0] = py[3] = fy;

    td.in = in; 
    td.out = out;
    td.px = px;
    td.py = py;
    td.ww = ww;
    td.wh = wh;
    td.full_chroma = s->chroma;
    ctx->internal->execute(ctx, zoom_slice, &td, NULL, FFMIN(4, ff_filter_get_nb_threads(ctx)));

    out->pts = pts;
    s->frame_count++;

    ret = ff_filter_frame(outlink, out);
    s->current_frame++;

    if (s->current_frame >= s->nb_frames) {
        if (*dx != -1)
            s->x = *dx;
        if (*dy != -1)
            s->y = *dy;
        if (*zoom != -1)
            s->prev_zoom = *zoom;
        s->prev_nb_frames = s->nb_frames;
        s->nb_frames = 0;
        s->current_frame = 0;
        av_frame_free(&s->in);
        s->finished = 1;
    }
    return ret;
}

static int activate(AVFilterContext *ctx)
{
    ZPBicontext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    int status, ret = 0;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (s->in && ff_outlink_frame_wanted(outlink)) {
        double zoom = -1, dx = -1, dy = -1;

        ret = output_single_frame(ctx, s->in, s->var_values, s->current_frame,
                                  &zoom, &dx, &dy);

        if (ret < 0)
            return ret;
    }

    if (!s->in && (ret = ff_inlink_consume_frame(inlink, &s->in)) > 0) {
        double zoom = -1, dx = -1, dy = -1, nb_frames;

        s->finished = 0;
        s->var_values[VAR_IN_W]  = s->var_values[VAR_IW] = s->in->width;
        s->var_values[VAR_IN_H]  = s->var_values[VAR_IH] = s->in->height;
        s->var_values[VAR_OUT_W] = s->var_values[VAR_OW] = s->w;
        s->var_values[VAR_OUT_H] = s->var_values[VAR_OH] = s->h;
        s->var_values[VAR_IN]    = inlink->frame_count_out - 1;
        s->var_values[VAR_ON]    = outlink->frame_count_in;
        s->var_values[VAR_PX]    = s->x;
        s->var_values[VAR_PY]    = s->y;
        s->var_values[VAR_X]     = 0;
        s->var_values[VAR_Y]     = 0;
        s->var_values[VAR_PZOOM] = s->prev_zoom;
        s->var_values[VAR_ZOOM]  = 1;
        s->var_values[VAR_PDURATION] = s->prev_nb_frames;
        s->var_values[VAR_A]     = (double) s->in->width / s->in->height;
        s->var_values[VAR_SAR]   = inlink->sample_aspect_ratio.num ?
            (double) inlink->sample_aspect_ratio.num / inlink->sample_aspect_ratio.den : 1;
        s->var_values[VAR_DAR]   = s->var_values[VAR_A] * s->var_values[VAR_SAR];
        s->var_values[VAR_HSUB]  = 1 << s->desc->log2_chroma_w;
        s->var_values[VAR_VSUB]  = 1 << s->desc->log2_chroma_h;

        if ((ret = av_expr_parse_and_eval(&nb_frames, s->duration_expr_str,
                                          var_names, s->var_values,
                                          NULL, NULL, NULL, NULL, NULL, 0, ctx)) < 0) {
            av_frame_free(&s->in);
            return ret;
        }

        s->var_values[VAR_DURATION] = s->nb_frames = nb_frames;

        ret = output_single_frame(ctx, s->in, s->var_values, s->current_frame,
                                  &zoom, &dx, &dy);
        if (ret < 0)
            return ret;
    }
    if (ret < 0) {
        return ret;
    } else if (s->finished && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        ff_outlink_set_status(outlink, status, pts);
        return 0;
    } else {
        if (ff_outlink_frame_wanted(outlink) && s->finished)
            ff_inlink_request_frame(inlink);
        return 0;
    }
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV444P,  AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_YUV410P,  AV_PIX_FMT_YUV440P,
        AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUVA422P,
        AV_PIX_FMT_YUVA420P,
        AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P,
        AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUVJ411P,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRAP,
        AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ZPBicontext *s = ctx->priv;
    av_expr_free(s->x_expr);
    av_expr_free(s->y_expr);
    av_expr_free(s->zoom_expr);
    av_frame_free(&s->in);
}

static const AVFilterPad inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

static const AVFilterPad outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_zoompan_bi = {
    .name          = "zoompanbi",
    .description   = NULL_IF_CONFIG_SMALL("Apply Zoom & Pan (Bilinear) effect."),
    .priv_size     = sizeof(ZPBicontext),
    .priv_class    = &zoompan_bi_class,
    .query_formats = query_formats,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    .inputs        = inputs,
    .outputs       = outputs,
};
