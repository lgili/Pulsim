/**
 * efficiency_map.c — MOSFET switching-loss lookup table CBlock.
 *
 * Loads a CSV table (Vds × Id → loss_W) from a file whose path is
 * passed via the PULSIM_CBLOCK_CSV_PATH environment variable at init time.
 * At each step it performs bilinear interpolation.
 *
 * Inputs:  in[0] = Vds (V), in[1] = Id (A)
 * Outputs: out[0] = switching loss (W), out[1] = efficiency (0-1)
 *
 * The CSV format expected is:
 *   vds_V,id_A,loss_W
 *   10,1,0.05
 *   ...
 *
 * Demonstrates:
 *   - File I/O in pulsim_cblock_init
 *   - Complex state allocation (heap-allocated 2-D table)
 *   - Multi-output block (loss + efficiency)
 */

#include "pulsim/v1/cblock_abi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VDS 16
#define MAX_ID  16
#define MAX_PTS 256

/* ------------------------------------------------------------------ */
/* Internal state                                                       */
/* ------------------------------------------------------------------ */

struct PulsimCBlockCtx {
    double vds[MAX_PTS];    /* sorted unique Vds axis */
    double id [MAX_PTS];    /* sorted unique Id axis  */
    double loss[MAX_PTS][MAX_PTS]; /* loss_W at (vds[i], id[j]) */
    int    nvds;
    int    nid;
};

/* Binary search: find last index k such that arr[k] <= val */
static int lower_bound(const double* arr, int n, double val) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (arr[mid] <= val) lo = mid; else hi = mid - 1;
    }
    return lo;
}

static double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

static double bilinear(const struct PulsimCBlockCtx* s,
                        double vds, double id_a) {
    /* Clamp to table bounds */
    if (vds < s->vds[0]) vds = s->vds[0];
    if (vds > s->vds[s->nvds - 1]) vds = s->vds[s->nvds - 1];
    if (id_a < s->id[0])  id_a = s->id[0];
    if (id_a > s->id[s->nid - 1]) id_a = s->id[s->nid - 1];

    int i0 = lower_bound(s->vds, s->nvds, vds);
    int j0 = lower_bound(s->id,  s->nid,  id_a);
    int i1 = (i0 + 1 < s->nvds) ? i0 + 1 : i0;
    int j1 = (j0 + 1 < s->nid)  ? j0 + 1 : j0;

    double dv = (i0 == i1) ? 0.0 :
        (vds - s->vds[i0]) / (s->vds[i1] - s->vds[i0]);
    double di = (j0 == j1) ? 0.0 :
        (id_a - s->id[j0]) / (s->id[j1] - s->id[j0]);

    double v00 = s->loss[i0][j0], v10 = s->loss[i1][j0];
    double v01 = s->loss[i0][j1], v11 = s->loss[i1][j1];

    return lerp(lerp(v00, v10, dv), lerp(v01, v11, dv), di);
}

/* ------------------------------------------------------------------ */

PULSIM_CBLOCK_EXPORT int pulsim_cblock_abi_version = PULSIM_CBLOCK_ABI_VERSION;


PULSIM_CBLOCK_EXPORT int pulsim_cblock_init(
    PulsimCBlockCtx** ctx_out, const PulsimCBlockInfo* info)
{
    (void)info;

    const char* path = getenv("PULSIM_CBLOCK_CSV_PATH");
    if (!path) {
        fprintf(stderr, "efficiency_map.c: PULSIM_CBLOCK_CSV_PATH not set\n");
        return -1;
    }

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "efficiency_map.c: cannot open %s\n", path);
        return -1;
    }

    struct PulsimCBlockCtx* s =
        (struct PulsimCBlockCtx*)calloc(1, sizeof(struct PulsimCBlockCtx));
    if (!s) { fclose(f); return -1; }

    /* Parse header (skip) */
    char line[256];
    fgets(line, sizeof(line), f);

    /* Temporary storage for raw points */
    double raw_vds[MAX_PTS], raw_id[MAX_PTS], raw_loss[MAX_PTS];
    int npts = 0;

    while (fgets(line, sizeof(line), f) && npts < MAX_PTS) {
        double v, id_a, l;
        if (sscanf(line, "%lf,%lf,%lf", &v, &id_a, &l) == 3) {
            raw_vds[npts]  = v;
            raw_id[npts]   = id_a;
            raw_loss[npts] = l;
            npts++;
        }
    }
    fclose(f);

    if (npts == 0) { free(s); return -1; }

    /* Build sorted unique axes */
    for (int k = 0; k < npts; k++) {
        /* Vds axis */
        int found = 0;
        for (int i = 0; i < s->nvds; i++) {
            if (fabs(s->vds[i] - raw_vds[k]) < 1e-9) { found = 1; break; }
        }
        if (!found && s->nvds < MAX_PTS) s->vds[s->nvds++] = raw_vds[k];

        /* Id axis */
        found = 0;
        for (int j = 0; j < s->nid; j++) {
            if (fabs(s->id[j] - raw_id[k]) < 1e-9) { found = 1; break; }
        }
        if (!found && s->nid < MAX_PTS) s->id[s->nid++] = raw_id[k];
    }

    /* Simple insertion sort for axes */
    for (int i = 1; i < s->nvds; i++) {
        double t = s->vds[i]; int j = i - 1;
        while (j >= 0 && s->vds[j] > t) { s->vds[j+1]=s->vds[j]; j--; }
        s->vds[j+1] = t;
    }
    for (int i = 1; i < s->nid; i++) {
        double t = s->id[i]; int j = i - 1;
        while (j >= 0 && s->id[j] > t) { s->id[j+1]=s->id[j]; j--; }
        s->id[j+1] = t;
    }

    /* Fill loss table */
    for (int k = 0; k < npts; k++) {
        int i = lower_bound(s->vds, s->nvds, raw_vds[k]);
        int j = lower_bound(s->id,  s->nid,  raw_id[k]);
        s->loss[i][j] = raw_loss[k];
    }

    *ctx_out = (PulsimCBlockCtx*)s;
    return 0;
}


PULSIM_CBLOCK_EXPORT int pulsim_cblock_step(
    PulsimCBlockCtx* ctx, double t, double dt,
    const double* in, double* out)
{
    (void)t; (void)dt;

    struct PulsimCBlockCtx* s = (struct PulsimCBlockCtx*)ctx;

    double vds  = in[0];  /* drain-source voltage (V) */
    double id_a = in[1];  /* drain current (A)       */

    double loss_w = bilinear(s, vds, id_a);

    /* Power delivered = vds * id */
    double p_delivered = vds * id_a;
    double efficiency = (p_delivered > 1e-9)
        ? (p_delivered - loss_w) / p_delivered
        : 1.0;
    if (efficiency < 0.0) efficiency = 0.0;
    if (efficiency > 1.0) efficiency = 1.0;

    out[0] = loss_w;
    out[1] = efficiency;
    return 0;
}


PULSIM_CBLOCK_EXPORT void pulsim_cblock_destroy(PulsimCBlockCtx* ctx)
{
    free(ctx);
}
