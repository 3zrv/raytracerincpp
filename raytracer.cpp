#include <stdio.h>
#include <math.h>
#include <stdlib.h>
//#include <dos.h> // for outportb,inportb

#include <gd.h>

#include <assert.h>

#include "helper_cuda.h"

//#define DO_DITHER

// Raytracing is mathematics-heavy. Declare mathematical datatypes
inline double __host__ __device__ dmin(double a,double b) { return a<b ? a : b; }
struct XYZ
{
    double d[3];
    // Declare operators pertinent to vectors in general:
    void __host__ __device__ Set(double a,double b,double c)
        { d[0]=a; d[1]=b; d[2]=c;}
    #define do_op(o) \
        inline void __device__ __host__ operator o##= (const XYZ& b) { for(unsigned n=0; n<3; ++n) d[n] o##= b.d[n]; } \
        inline void __device__ __host__ operator o##= (double b)     { for(unsigned n=0; n<3; ++n) d[n] o##= b; } \
        XYZ __device__ __host__ operator o (const XYZ& b) const      { XYZ tmp(*this); tmp o##= b; return tmp; } \
        XYZ __device__ __host__ operator o (double b)     const      { XYZ tmp(*this); tmp o##= b; return tmp; }
    do_op(*)
    do_op(+)
    do_op(-)
    #undef do_op
    XYZ __device__ __host__ operator- () const { XYZ tmp = { { -d[0], -d[1], -d[2] } }; return tmp; }
    XYZ __device__ __host__ Pow(double b) const
        { XYZ tmp = {{ pow(d[0],b), pow(d[1],b), pow(d[2],b) }}; return tmp; }
    // Operators pertinent to geometrical vectors:
    inline __device__ __host__ double Dot(const XYZ& b) const
        { return d[0]*b.d[0] + d[1]*b.d[1] + d[2]*b.d[2]; }
    inline __device__ __host__ double Squared() const     { return Dot(*this); }
    inline __device__ __host__ double Len()     const     { return sqrt(Squared()); }
    inline __device__ __host__ void Normalize()           { *this *= 1.0 / Len(); }
    inline __device__ __host__ void MirrorAround(const XYZ& axis)
    {
        XYZ N = axis; N.Normalize();
        double v = Dot(N);
        *this = N * (v+v) - *this;
    }
    // Operators pertinent to colour vectors (RGB):
    inline __device__ __host__ double Luma() const
        { return d[0]*0.2126 + d[1]*0.7152 + d[2]*0.0722; }
    void __device__ __host__ Clamp()
    {
        for(unsigned n=0; n<3; ++n)
        {
            if(d[n] < 0.0) d[n] = 0.0;
            else if(d[n] > 1.0) d[n] = 1.0;
        }
    }
    void __device__ __host__ ClampWithDesaturation()
    {
        //¢£õö¶·ü ×‡Óƒ ˜™Øˆ Ð€ õöùúÐ€üõö ¢© ®¯
        // If the color represented by this triplet
        // is too bright or too dim, decrease the
        // saturation as much as required, while keeping
        // the luma unmodified.
        double l = Luma(), sat = 1.0;
        if(l > 1.0) { d[0] = d[1] = d[2] = 1.0; return; }
        if(l < 0.0) { d[0] = d[1] = d[2] = 0.0; return; }
        // If any component is over the bounds, calculate how
        // much the saturation must be reduced to achieve an
        // in-bounds value. Since the luma was verified to be
        // in 0..1 range, a maximum reduction of saturationto
        // 0% will always produce an in-bounds value, but
        // usually such a drastic reduction is not necessary.
        // Because we're only doing relative modifications,
        // we don't need to determine the original saturation
        // level of the pixel.
        for(int n=0; n<3; ++n)
            if(d[n] > 1.0) sat = dmin(sat, (l-1.0) / (l-d[n]));
            else if(d[n] < 0.0) sat = dmin(sat, l  / (l-d[n]));
        if(sat != 1.0)
             { *this = (*this - l) * sat + l; Clamp(); }
    }
};
struct Matrix /* Only needed in main() */
{
    XYZ m[3];
    void __device__ __host__ InitRotate(const XYZ& angle)
    {
        double Cx = cos(angle.d[0]), Cy = cos(angle.d[1]), Cz = cos(angle.d[2]);
        double Sx = sin(angle.d[0]), Sy = sin(angle.d[1]), Sz = sin(angle.d[2]);
        double sxsz = Sx*Sz, cxsz = Cx*Sz;
        double cxcz = Cx*Cz, sxcz = Sx*Cz;
        Matrix result = {{ {{ Cy*Cz, Cy*Sz, -Sy }},
                           {{ sxcz*Sy - cxsz, sxsz*Sy + cxcz, Sx*Cy }},
                           {{ cxcz*Sy + sxsz, cxsz*Sy - sxcz, Cx*Cy }} }};
        *this = result;
    }
    void __device__ __host__ Transform(XYZ& vec)
    {
        vec.Set( m[0].Dot(vec), m[1].Dot(vec), m[2].Dot(vec) );
    }
};

/* There. The previous part was just basic mathematics. Vector and matrix mathematics. */

extern "C" {
// Declarations for scene description
// Walls are planes. Planes have a
// normal vector and a distance.
typedef struct Plane
{ XYZ normal; double offset; } Pl;
const Pl PlanesPreinit[] =
{
// Declare six planes, each looks 
// towards origo and is 30 units away.
    { {{0,0,-1}}, -30 },
    { {{0, 1,0}}, -30 },
    { {{0,-1,0}}, -30 },
    { {{ 1,0,0}}, -30 },
    { {{0,0, 1}}, -30 },
    { {{-1,0,0}}, -30 }
};
__device__ __constant__ Pl Planes[sizeof(PlanesPreinit)/sizeof(*PlanesPreinit)];

typedef struct Sphere
{ XYZ center; double radius; } Sp;
const Sp SpheresPreinit[] =
{
// Declare a few spheres.
    { {{0,0,0}}, 7 },
    { {{19.4, -19.4, 0}}, 2.1 },
    { {{-19.4, 19.4, 0}}, 2.1 },
    { {{13.1, 5.1, 0}}, 1.1 },
    { {{ -5.1, -13.1, 0}}, 1.1 },
    { {{-30,30,15}}, 11},
    { {{15,-30,30}}, 6},
    { {{30,15,-30}}, 6}
};
__device__ __constant__ Sp Spheres[sizeof(SpheresPreinit)/sizeof(*SpheresPreinit)];

// Declare lightsources. A lightsource
// has a location and a RGB color.
typedef struct LightSource
{ XYZ where, colour; } Ls;
const Ls LightsPreinit[] =
{
    { {{-28,-14, 3}}, {{.4, .51, .9}} },
    { {{-29,-29,-29}}, {{.95, .1, .1}} },
    { {{ 14, 29,-14}}, {{.8, .8, .8}} },
    { {{ 29, 29, 29}}, {{1,1,1}} },
    { {{ 28,  0, 29}}, {{.5, .6,  .1}} }
};
__device__ __constant__ Ls Lights[sizeof(LightsPreinit)/sizeof(*LightsPreinit)];

#define NElems(x) sizeof(x)/sizeof(*x)
const unsigned
    NumPlanes = NElems(Planes),
    NumSpheres = NElems(Spheres),
    NumLights = NElems(Lights),
    MAXTRACE = 6; // Maximum trace level
} // extern "C"

/* Actual raytracing! */

/**************/
// Determine whether an object is
// in direct eyesight on the given
// line, and determine exactly which
// point of the object is seen.
int __device__ RayFindObstacle
    (const XYZ& eye, const XYZ& dir,
     double& HitDist, int& HitIndex,
     XYZ& HitLoc, XYZ& HitNormal)
{
    // Try intersecting the ray with
    // each object and see which one
    // produces the closest hit.
    int HitType = -1;
   {for(unsigned i=0; i<NumSpheres; ++i)
    {
        XYZ V (eye - Spheres[i].center);
        double r = Spheres[i].radius,
            DV = dir.Dot(V),
            D2 = dir.Squared(),
            SQ = DV*DV
               - D2*(V.Squared() - r*r);
        // Does the ray coincide
        // with the sphere?
        if(SQ < 1e-6) continue;
        // Determine where exactly
        double SQt = sqrt(SQ),
            Dist = dmin(-DV-SQt,
                        -DV+SQt) / D2;
        if(Dist<1e-6 || Dist >= HitDist)
            continue;
        HitType = 1; HitIndex = i;
        HitDist = Dist;
        HitLoc = eye + (dir * HitDist);
        HitNormal =
            (HitLoc - Spheres[i].center)
                * (1/r);
    }}
   {for(unsigned i=0; i<NumPlanes; ++i)
    {
        double DV = -Planes[i].normal.Dot(dir);
        if(DV > -1e-6) continue;
        double D2 =
            Planes[i].normal.Dot(eye),
            Dist = (D2+Planes[i].offset) / DV;
        if(Dist<1e-6 || Dist>=HitDist)
            continue;
        HitType = 0; HitIndex = i;
        HitDist = Dist;
        HitLoc = eye + (dir * HitDist);
        HitNormal = -Planes[i].normal;
    }}
    return HitType;
}

bool __device__ RayFindObstacle(const XYZ& eye, const XYZ& dir, const double HitDist)
{
    // Try intersecting the ray with
    // each object and see which one
    // produces the closest hit.
    int result = 0;
   {for(unsigned i=0; i<NumSpheres; ++i)
    {
        XYZ V (eye - Spheres[i].center);
        double r = Spheres[i].radius,
            DV = dir.Dot(V),
            D2 = dir.Squared(),
            SQ = DV*DV
               - D2*(V.Squared() - r*r);
        // Does the ray coincide
        // with the sphere?
        // Determine where exactly
        double Dist = SQ >= 1e-6 ? dmin(-DV-sqrt(SQ), -DV+sqrt(SQ)) / D2 : 0;
        result |= !(Dist<1e-6 || Dist >= HitDist);
    }}
   {for(unsigned i=0; i<NumPlanes; ++i)
    {
        double DV = -Planes[i].normal.Dot(dir);
        double D2 =
            Planes[i].normal.Dot(eye),
            Dist = (D2+Planes[i].offset) / DV;
        result |= !(Dist<1e-6 || Dist>=HitDist);
    }}
    return result;
}

//extern "C" {
const unsigned NumArealightVectors = 1;
__device__ __constant__ XYZ ArealightVectors[NumArealightVectors];
XYZ ArealightVectorsPreinit[NumArealightVectors];
//} // extern "C"
void InitArealightVectors()
{
    // To smooth out shadows cast by lightsources,
    // I plan to approximate the lightsources with
    // a _cloud_ of lightsources around the point
    for(unsigned i=0; i<NumArealightVectors; ++i)
        for(unsigned n=0; n<3; ++n)
            ArealightVectorsPreinit[i].d[n] =
                0;//2.0 * (rand() / double(RAND_MAX) - 0.5) * 0.1;
}

// Shoot a camera-ray from the specified location
// to specified location, and determine the RGB
// color of the perception corresponding to that
// location.
void __device__ RayTrace(XYZ& resultcolor, const XYZ& eye, const XYZ& dir, int k)
{
    double HitDist = 1e6;
    XYZ HitLoc, HitNormal;
    int HitIndex, HitType;
    HitType = RayFindObstacle(eye,dir, HitDist,HitIndex, HitLoc,HitNormal);
    if(HitType != -1)
    {
        // Found an obstacle. Next, find out how it is illuminated.
        // Shoot a ray to each lightsource, and determine if there
        // is an obstacle behind it. This is called "diffuse light".
        // To smooth out the infinitely sharp shadows caused by
        // infinitely small point-lightsources, assume the lightsource
        // is actually a cloud of small lightsources around its center.
        XYZ DiffuseLight = {{0,0,0}}, SpecularLight = {{0,0,0}};
        XYZ Pigment = {{1, 0.98, 0.94}}; // default pigment
        for(unsigned i=0; i<NumLights; ++i)
            for(unsigned j=0; j<NumArealightVectors; ++j)
            {
                XYZ V((Lights[i].where + ArealightVectors[j]) - HitLoc);
                double LightDist = V.Len();
                V.Normalize();
                double DiffuseEffect = HitNormal.Dot(V) / (double)NumArealightVectors;
                double Attenuation = (1 + pow(LightDist / 34.0, 2.0));
                DiffuseEffect /= Attenuation;
                if(DiffuseEffect > 1e-3)
                {
                    double ShadowDist = LightDist - 1e-4;
                    if(!RayFindObstacle(HitLoc + V*1e-4, V, ShadowDist))
                        DiffuseLight += Lights[i].colour * DiffuseEffect;
            }   }
        if(k > 1)
        {
            // Add specular light/reflection, unless recursion depth is at max
            XYZ V(-dir); V.MirrorAround(HitNormal);
            RayTrace(SpecularLight, HitLoc + V*1e-4, V, k-1);
        }
        switch(HitType)
        {
            case 0: // plane
                DiffuseLight *= 0.9;
                SpecularLight *= 0.5;
                // Color the different walls differently
                switch(HitIndex % 3)
                {
                    case 0: Pigment.Set(0.9, 0.7, 0.6); break;
                    case 1: Pigment.Set(0.6, 0.7, 0.7); break;
                    case 2: Pigment.Set(0.5, 0.8, 0.3); break;
                }
                break;
            case 1: // sphere
                DiffuseLight  *= 1.0;
                SpecularLight  *= 0.34;
        }
        resultcolor = (DiffuseLight + SpecularLight) * Pigment;
    }
}

const double Gamma = 2.0, Ungamma = 1.0 / Gamma;
#ifdef DO_DITHER
extern "C" {
// Declarations for the 8x8 Knoll-Yliluoma dithering
const unsigned CandCount = 64;
unsigned char Dither8x8_init[8][8];
XYZ Pal[16], PalG_init[16];
double lumainit[16];
__device__ __constant__ unsigned char Dither8x8[8][8];
__device__ __constant__ XYZ PalG[16];
__device__ __constant__ double luma[16];
} // extern "C"

void InitDither()
{
    // We will use the default 16-color EGA/VGA palette.
    //outportb(0x3C7, 0); // Read palette from VGA.
    for(unsigned i=0; i<16; ++i)
    {
        static const char s[16*3] =
            {0,0,0, 0,0,42, 0,42,0, 0,42,42, 42,0,0, 42,0,42, 42,21,0, 21,21,21,
             42,42,42, 21,21,63, 21,63,21, 21,63,63, 63,21,21, 63,21,63, 63,63,21, 63,63,63};
        Pal[i].Set(s[i*3+0],s[i*3+1],s[i*3+2]);
        /*for(i==8) outportb(0x3C7, 64-8);
        for(unsigned n=0; n<3; ++n)
            Pal[i].d[n] = inportb(0x3C9);*/
        Pal[i] *= 1/63.0;
        PalG_init[i] = Pal[i].Pow(Gamma);
        lumainit[i] = PalG_init[i].Luma();
    }
    // Create bayer dithering matrix, adjusted for candidate count
    for(unsigned y=0; y<8; ++y)
        for(unsigned x=0; x<8; ++x)
        {
            unsigned i = x ^ y, j;
            j = (x & 4)/4u + (x & 2)*2u + (x & 1)*16u;
            i = (i & 4)/2u + (i & 2)*4u + (i & 1)*32u;
            Dither8x8_init[y][x] = (j+i)*CandCount/64u;
        }
}
#endif

//const unsigned W = 640, H = 480;
const unsigned W = 1920, H = 1080;
const unsigned Threads = 256;
const unsigned Blocks  = (W*H + (Threads-1)) / Threads;

void __global__ RenderScreen(
                             #ifdef DO_DITHER
                             unsigned char* results,
                             #else
                             unsigned*      results,
                             #endif
                             double*        resluma,
                             double camanglex,double camangley,double camanglez,
                             double camlookx,double camlooky,double camlookz,
                             double zoom,
                             double contrast,double contrast_offset)
{
    unsigned pixno = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixno >= W*H) return;

    // Put camera between the central sephere and the walls
    XYZ camangle = { { camanglex,camangley,camanglez } };
    XYZ camlook = { { camlookx,camlooky,camlookz } };
    XYZ campos = { { 0.0, 0.0, 16.0} };
    // Rotate it around the center
    Matrix camrotatematrix, camlookmatrix;
    camrotatematrix.InitRotate(camangle);
    camrotatematrix.Transform(campos);
    camlookmatrix.InitRotate(camlook);

    const unsigned x = pixno % W;
    const unsigned y = pixno / W;
    XYZ camray = { { x / double(W) - 0.5,
                     y / double(H) - 0.5,
                     zoom } };
    camray.d[0] *= double(W)/double(H); // Aspect ratio correction
    camray.Normalize();
    camlookmatrix.Transform(camray);
    XYZ campix;
    RayTrace(campix, campos, camray, MAXTRACE);
    campix *= 0.5;
    resluma[y*W+x] = campix.Luma();
    // Exaggerate the colors to bring contrast better forth
    campix = (campix + contrast_offset) * contrast;
    // Clamp, and compensate for display gamma (for dithering)
    campix.ClampWithDesaturation();
    XYZ campixG = campix.Pow(Gamma);
#ifdef DO_DITHER
    XYZ qtryG = campixG;
    // Create candidate for dithering
    unsigned candlist[CandCount];
    for(unsigned i=0; i<CandCount; ++i)
    {
        unsigned k = 0;
        double b = 1e6;
        // Find closest match from palette
        for(unsigned j=0; j<16; ++j)
        {
            double a = (qtryG - PalG[j]).Squared();
            if(a < b) { b = a; k = j; }
        }
        candlist[i] = k;
        if(i+1 >= CandCount) break;
        // Compensate for error
        qtryG += (campixG - PalG[k]);
        qtryG.Clamp();
    }
    // Order candidates by luminosity
    // using insertion sort.
    for(unsigned j=1; j<CandCount; ++j)
    {
        unsigned k = candlist[j], i;
        for(i=j; i>=1 && luma[candlist[i-1]] > luma[k]; --i)
            candlist[i] = candlist[i-1];
        candlist[i] = k;
    }
    // Draw pixel (use BIOS).
    results[y*W+x] = candlist[Dither8x8[x & 7][y & 7]];
#else
    results[y*W+x] = (unsigned(campixG.d[0] * 255) << 16)
                   + (unsigned(campixG.d[1] * 255) << 8)
                   + (unsigned(campixG.d[2] * 255) << 0);
#endif
}

/* MAIN PROGRAM */
int main()
{
//    _asm { mov ax, 0x12; int 0x10 };
    // ^  Use BIOS, set 640x480 16-color graphics mode.

    InitArealightVectors();
    #define PreInit(symbol, from) \
        checkCudaErrors(cudaMemcpyToSymbol(symbol, &from, sizeof(from)))
    PreInit(ArealightVectors, ArealightVectorsPreinit);
#ifdef DO_DITHER
    InitDither();
    PreInit(PalG, PalG_init);
    PreInit(luma, lumainit);
    PreInit(Dither8x8, Dither8x8_init);
#endif
    PreInit(Planes, PlanesPreinit);
    PreInit(Spheres, SpheresPreinit);
    PreInit(Lights, LightsPreinit);
    #undef PreInit
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize,2500));

    XYZ camangle      = { {0,0,0} };
    XYZ camangledelta = { {-.005, -.011, -.017} };
    XYZ camlook       = { {0,0,0} };
    XYZ camlookdelta  = { {-.001, .005, .004} };

    double zoom = 46.0, zoomdelta = 0.99;
    double contrast = 32, contrast_offset = -0.17;

    // Render
#ifdef DO_DITHER
    static unsigned char results[W*H], *p = NULL;
#else
    static unsigned      results[W*H], *p = NULL;
#endif
    static double        resluma[W*H], *L = NULL;
    checkCudaErrors(cudaMalloc((void**)&p, sizeof(results))); assert(p!=NULL);
    checkCudaErrors(cudaMalloc((void**)&L, sizeof(resluma))); assert(L!=NULL);

    for(unsigned frameno=0; frameno<9300; ++frameno)
    {
    #ifdef DO_DITHER
        gdImagePtr im = gdImageCreate(W,H);
        {for(unsigned p=0; p<16; ++p)
            gdImageColorAllocate(im, (int)(Pal[p].d[0]*255+0.5),
                                     (int)(Pal[p].d[1]*255+0.5),
                                     (int)(Pal[p].d[2]*255+0.5));}
    #else
        gdImagePtr im = gdImageCreateTrueColor(W,H);
    #endif

        // Put camera between the central sphere and the walls
        //XYZ campos = { { 0.0, 0.0, 16.0} };
        // Rotate it around the center
        //Matrix camrotatematrix, camlookmatrix;
        //camrotatematrix.InitRotate(camangle);
        //camrotatematrix.Transform(campos);
        //camlookmatrix.InitRotate(camlook);

        fprintf(stderr, "Begins frame %u; contrast %g, contrast offset %g ",
            frameno,contrast,contrast_offset); fflush(stderr);

        RenderScreen<<<Blocks,Threads,0>>> (p,L,
                                            camangle.d[0],camangle.d[1],camangle.d[2],
                                            camlook.d[0],camlook.d[1],camlook.d[2],
                                            zoom,
                                            contrast,contrast_offset);
        checkCudaErrors(cudaMemcpy(results, p, sizeof(results), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(resluma, L, sizeof(resluma), cudaMemcpyDeviceToHost));

        // Determine the contrast ratio for this frame's pixels
        double thisframe_min = 100;
        double thisframe_max = -100;
        for(unsigned y=0; y<H; ++y)
            for(unsigned x=0; x<W; ++x)
            {
                double lum = resluma[y*W+x];
                // Update frame luminosity info for automatic contrast adjuster
                if(lum < thisframe_min) thisframe_min = lum;
                if(lum > thisframe_max) thisframe_max = lum;
                // Draw pixel (use BIOS).
                unsigned color = results[y*W+x];
                /*_asm {
                    mov ax, color
                    mov ah, 0x0C
                    xor bx, bx
                    mov cx, x
                    mov dx, y
                    int 0x10
                }*/
                gdImageSetPixel(im, x,y, color);
            }

        char Buf[64]; sprintf(Buf, "trace%04d.png", frameno);
        fprintf(stderr, "Writing %s...\n", Buf);
        FILE* fp = fopen(Buf, "wb");
        gdImagePng(im, fp);
        gdImageDestroy(im);
        fclose(fp);

        // Tweak coordinates / camera parameters for the next frame
        double much = 1.0;

        // In the beginning, do some camera action (play with zoom)
        if(zoom <= 1.1)
            zoom = 1.1;
        else
        {
            if(zoom > 40) { if(zoomdelta > 0.95) zoomdelta -= 0.001; }
            else if(zoom < 3) { if(zoomdelta < 0.99) zoomdelta += 0.001; }
            zoom *= zoomdelta;
            much = 1.1 / pow(zoom/1.1, 3);
        }

        // Update the rotation angle
        camlook  += camlookdelta * much;
        camangle += camangledelta * much;

        // Dynamically readjust the contrast based on the contents
        // of the last frame
        double middle = (thisframe_min + thisframe_max) * 0.5;
        double span   = (thisframe_max - thisframe_min);
        thisframe_min = middle - span*0.60; // Avoid dark tones
        thisframe_max = middle + span*0.37; // Emphasize bright tones
        double new_contrast_offset = -thisframe_min;
        double new_contrast        = 1 / (thisframe_max - thisframe_min);
        // Avoid too abrupt changes, though
        double l = 0.85;
        if(frameno == 0) l = 0.7;
        contrast_offset = (contrast_offset*l + new_contrast_offset*(1.0-l));
        contrast        = (contrast*l + new_contrast*(1.0-l));
    }

    checkCudaErrors(cudaFree(p));
    checkCudaErrors(cudaFree(L));

//    _asm { mov ax, 0x03; int 0x10 };
    // Set 80x25 text mode.
}