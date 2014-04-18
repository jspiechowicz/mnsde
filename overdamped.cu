/*
 * Overdamped Brownian Particle
 *
 * \dot{x} = -V'(x) + a\cos(\omega t) + f + Gaussian, Poissonian and dichotomous noise
 *
 */

#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.14159265358979f

//model
__constant__ float d_amp, d_omega, d_force, d_Dg, d_Dp, d_lambda, d_mean, d_fa, d_fb, d_mua, d_mub;
__constant__ int d_comp;
float h_omega, h_fa, h_fb, h_mua, h_mub, h_mean;
int h_comp;

//simulation
float h_trans;
int h_dev, h_block, h_grid, h_spp;
long h_paths, h_periods, h_threads, h_steps, h_trigger;
__constant__ int d_spp, d_2ndorder;
__constant__ long d_paths, d_steps, d_trigger;

//output
char *h_domain;
char h_domainx, h_domainy;
float h_beginx, h_endx, h_beginy, h_endy;
int h_logx, h_logy, h_points, h_moments, h_traj, h_hist;
__constant__ char d_domainx;
__constant__ int d_points;

//vector
float *h_x, *h_w, *h_xb, *h_dx;
float *d_x, *d_w, *d_xb, *d_dx;
unsigned int *h_seeds, *d_seeds;
curandState *d_states;

size_t size_f, size_ui, size_p;
curandGenerator_t gen;

static struct option options[] = {
    {"amp", required_argument, NULL, 'a'},
    {"omega", required_argument, NULL, 'b'},
    {"force", required_argument, NULL, 'c'},
    {"Dg", required_argument, NULL, 'd'},
    {"Dp", required_argument, NULL, 'e'},
    {"lambda", required_argument, NULL, 'f'},
    {"comp", required_argument, NULL, 'g'},
    {"dev", required_argument, NULL, 'h'},
    {"block", required_argument, NULL, 'i'},
    {"paths", required_argument, NULL, 'j'},
    {"periods", required_argument, NULL, 'k'},
    {"trans", required_argument, NULL, 'l'},
    {"spp", required_argument, NULL, 'm'},
    {"algorithm", required_argument, NULL, 'n'},
    {"mode", required_argument, NULL, 'o'},
    {"domain", required_argument, NULL, 'p'},
    {"domainx", required_argument, NULL, 'q'},
    {"domainy", required_argument, NULL, 'r'},
    {"logx", required_argument, NULL, 's'},
    {"logy", required_argument, NULL, 't'},
    {"points", required_argument, NULL, 'u'},
    {"beginx", required_argument, NULL, 'v'},
    {"endx", required_argument, NULL, 'w'},
    {"beginy", required_argument, NULL, 'y'},
    {"endy", required_argument, NULL, 'z'},
    {"mean", required_argument, NULL, 'A'},
    {"fa", required_argument, NULL, 'B'},
    {"fb", required_argument, NULL, 'C'},
    {"mua", required_argument, NULL, 'D'},
    {"mub", required_argument, NULL, 'E'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -a, --amp=FLOAT         set the harmonic driving amplitude 'a' to FLOAT\n");
    printf("    -b, --omega=FLOAT       set the harmonic driving frequency '\\omega' to FLOAT\n");
    printf("    -c, --force=FLOAT       set the external bias 'F' to FLOAT\n");
    printf("    -d, --Dg=FLOAT          set the Gaussian noise intensity 'D_G' to FLOAT\n");
    printf("    -e, --Dp=FLOAT          set the Poissonian noise intensity 'D_P' to FLOAT\n");
    printf("    -f, --lambda=FLOAT      set the Poissonian kicks frequency '\\lambda' to FLOAT\n");
    printf("    -g, --comp=INT          choose between biased and unbiased Poissonian or dichotomous noise. INT can be one of:\n");
    printf("                            0: biased; 1: unbiased\n");
    printf("    -B, --fa=FLOAT          set the first state of the dichotomous noise 'F_a' to FLOAT\n");
    printf("    -C, --fb=FLOAT          set the second state of the dichotomous noise 'F_b' to FLOAT\n");
    printf("    -D, --mua=FLOAT         set the transition rate of the first state of dichotomous noise '\\mu_a' to FLOAT\n");
    printf("    -E, --mub=FLOAT         set the transition rate of the second state of dichotomous noise '\\mu_b' to FLOAT\n");
    printf("Simulation params:\n");
    printf("    -h, --dev=INT           set the gpu device to INT\n");
    printf("    -i, --block=INT         set the gpu block size to INT\n");
    printf("    -j, --paths=LONG        set the number of paths to LONG\n");
    printf("    -k, --periods=LONG      set the number of periods to LONG\n");
    printf("    -l, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -m, --spp=INT           specify how many integration steps should be calculated\n");
    printf("                            for a single period of the driving force\n");
    printf("    -n, --algorithm=STRING  sets the algorithm. STRING can be one of:\n");
    printf("                            predcorr: simplified weak order 2.0 adapted predictor-corrector\n");
    printf("                            euler: simplified weak order 1.0 regular euler-maruyama\n");
    printf("Output params:\n");
    printf("    -o, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first moment <<v>>\n");
    printf("                            trajectory: ensemble averaged <x>(t), <v>(t) and <x^2>(t), <v^2>(t)\n");
    printf("                            histogram: the final position x of all paths\n");
    printf("    -p, --domain=STRING     simultaneously scan over one or two model params. STRING can be one of:\n");
    printf("                            1d: only one parameter; 2d: two parameters at once\n");
    printf("    -q, --domainx=CHAR      sets the first domain of the moments. CHAR can be one of:\n");
    printf("                            a: amp; w: omega, f: force; D: Dg; p: Dp; l: lambda; i: fa; j: fb; m: mua; n: mub\n");
    printf("    -r, --domainy=CHAR      sets the second domain of the moments (only if --domain=2d). CHAR can be the same as above.\n");
    printf("    -s, --logx=INT          choose between linear and logarithmic scale of the domainx\n");
    printf("                            0: linear; 1: logarithmic\n");
    printf("    -t, --logy=INT          the same as above but for domainy\n");
    printf("    -u, --points=INT        set the number of samples to generate between begin and end\n");
    printf("    -v, --beginx=FLOAT      set the starting value of the domainx to FLOAT\n");
    printf("    -w, --endx=FLOAT        set the end value of the domainx to FLOAT\n");
    printf("    -y, --beginy=FLOAT      the same as --beginx, but for domainy\n");
    printf("    -z, --endy=FLOAT        the same as --endx, but for domainy\n");
    printf("    -A, --mean=FLOAT        if is nonzero, fix the mean value of Poissonian noise or dichotomous noise to FLOAT, matters only for domains p, l, i, j, m or n\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    float ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:y:z:A:B:C:D:E", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_amp, &ftmp, sizeof(float));
                break;
            case 'b':
                h_omega = atof(optarg);
                cudaMemcpyToSymbol(d_omega, &h_omega, sizeof(float));
                break;
            case 'c':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_force, &ftmp, sizeof(float));
                break;
            case 'd':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dg, &ftmp, sizeof(float));
                break;
            case 'e':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dp, &ftmp, sizeof(float));
                break;
            case 'f':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_lambda, &ftmp, sizeof(float));
                break;
            case 'g':
                h_comp = atoi(optarg);
                cudaMemcpyToSymbol(d_comp, &h_comp, sizeof(int));
                break;
            case 'h':
                itmp = atoi(optarg);
                cudaSetDevice(itmp);
                break;
            case 'i':
                h_block = atoi(optarg);
                break;
            case 'j':
                h_paths = atol(optarg);
                cudaMemcpyToSymbol(d_paths, &h_paths, sizeof(long));
                break;
            case 'k':
                h_periods = atol(optarg);
                break;
            case 'l':
                h_trans = atof(optarg);
                break;
            case 'm':
                h_spp = atoi(optarg);
                cudaMemcpyToSymbol(d_spp, &h_spp, sizeof(int));
                break;
            case 'n':
                if ( !strcmp(optarg, "predcorr") )
                    itmp = 1;
                else if ( !strcmp(optarg, "euler") )
                    itmp = 0;
                cudaMemcpyToSymbol(d_2ndorder, &itmp, sizeof(int));
                break;
            case 'o':
                if ( !strcmp(optarg, "moments") ) {
                    h_moments = 1;
                    h_traj = 0;
                    h_hist = 0;
                } else if ( !strcmp(optarg, "trajectory") ) {
                    h_moments = 0;
                    h_traj = 1;
                    h_hist = 0;
                } else if ( !strcmp(optarg, "histogram") ) {
                    h_moments = 0;
                    h_traj = 0;
                    h_hist = 1;
                }
                break;
            case 'p':
                h_domain = optarg;
                break;
            case 'q':
                h_domainx = optarg[0]; 
                cudaMemcpyToSymbol(d_domainx, &h_domainx, sizeof(char));
                break;
            case 'r':
                h_domainy = optarg[0];
                break;
            case 's':
                h_logx = atoi(optarg);
                break;
            case 't':
                h_logy = atoi(optarg);
                break;
            case 'u':
                h_points = atoi(optarg);
                cudaMemcpyToSymbol(d_points, &h_points, sizeof(int));
                break;
            case 'v':
                h_beginx = atof(optarg);
                break;
            case 'w':
                h_endx = atof(optarg);
                break;
            case 'y':
                h_beginy = atof(optarg);
                break;
            case 'z':
                h_endy = atof(optarg);
                break;
            case 'A':
                h_mean = atof(optarg);
                cudaMemcpyToSymbol(d_mean, &h_mean, sizeof(float));
                break;
            case 'B':
                h_fa = atof(optarg);
                cudaMemcpyToSymbol(d_fa, &h_fa, sizeof(float));
                break;
            case 'C':
                h_fb = atof(optarg);
                cudaMemcpyToSymbol(d_fb, &h_fb, sizeof(float));
                break;
            case 'D':
                h_mua = atof(optarg);
                cudaMemcpyToSymbol(d_mua, &h_mua, sizeof(float));
                break;
            case 'E':
                h_mub = atof(optarg);
                cudaMemcpyToSymbol(d_mub, &h_mub, sizeof(float));
                break;
        }
    }
}

__global__ void init_dev_rng(unsigned int *d_seeds, curandState *d_states)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(d_seeds[idx], idx, 0, &d_states[idx]);
}

__device__ float drift(float l_x, float l_w, float l_amp, float l_force)
{
    return -2.0f*PI*cosf(2.0f*PI*l_x) + l_amp*cosf(l_w) + l_force;
}

__device__ float diffusion(float l_Dg, float l_dt, int l_2ndorder, curandState *l_state)
{
    if (l_Dg != 0.0f) {
        float r = curand_uniform(l_state);
        if (l_2ndorder) {
            if ( r <= 1.0f/6 ) {
                return -sqrtf(6.0f*l_Dg*l_dt);
            } else if ( r > 1.0f/6 && r <= 2.0f/6 ) {
                return sqrtf(6.0f*l_Dg*l_dt);
            } else {
                return 0.0f;
            }
        } else {
            if ( r <= 0.5f ) {
                return -sqrtf(2.0f*l_Dg*l_dt);
            } else {
                return sqrtf(2.0f*l_Dg*l_dt);
            }
        }
    } else {
        return 0.0f;
    }
}

__device__ float adapted_jump_poisson(int &npcd, int pcd, float l_lambda, float l_Dp, int l_comp, float l_dt, curandState *l_state)
{
    if (l_Dp != 0.0f) {
        float comp = sqrtf(l_Dp*l_lambda)*l_dt;
        if (pcd <= 0) {
            float ampmean = sqrtf(l_lambda/l_Dp);
           
            npcd = (int) floor( -logf( curand_uniform(l_state) )/l_lambda/l_dt + 0.5f );

            if (l_comp) {
                return -logf( curand_uniform(l_state) )/ampmean - comp;
            } else {
                return -logf( curand_uniform(l_state) )/ampmean;
            }
        } else {
            npcd = pcd - 1;
            if (l_comp) {
                return -comp;
            } else {
                return 0.0f;
            }
        }
    } else {
        return 0.0f;
    }
}

__device__ float adapted_jump_dich(int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt, curandState *l_state)
{
    if (l_fa != 0.0f) {
        if (dcd <= 0) {
            if (dst == 0) {
                ndst = 1; 
                ndcd = (int) floor( -logf( curand_uniform(l_state) )/l_mub/l_dt + 0.5f );
                return l_fb;
            } else {
                ndst = 0;
                ndcd = (int) floor( -logf( curand_uniform(l_state) )/l_mua/l_dt + 0.5f );
                return l_fa;
            }
        } else {
            ndcd = dcd - 1;
            if (dst == 0) {
                return l_fa;
            } else {
                return l_fb;
            }
        }
    } else {
        return 0.0f;
    }
}

__device__ float regular_jump_poisson(float l_lambda, float l_Dp, int l_comp, float l_dt, curandState *l_state)
{
    if (l_Dp != 0.0f) {
        float mu, ampmean, comp, s;
        int i;
        unsigned int n;

        mu = l_lambda*l_dt;
        ampmean = sqrtf(l_lambda/l_Dp);
        comp = sqrtf(l_Dp*l_lambda)*l_dt;
        n = curand_poisson(l_state, mu);
        s = 0.0f;
            for (i = 0; i < n; i++) {
                s += -logf( curand_uniform(l_state) )/ampmean;
            }
        if (l_comp) s -= comp;
        return s;
    } else {
        return 0.0f;
    }
}

__device__ void predcorr(float &corrl_x, float l_x, float &corrl_w, float l_w, int &npcd, int pcd, curandState *l_state, \
                         float l_amp, float l_omega, float l_force, float l_Dg, int l_2ndorder, float l_Dp, float l_lambda, int l_comp, \
                         int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    float l_xt, l_xtt, l_wt, l_wtt, predl_x, predl_w;

    l_xt = drift(l_x, l_w, l_amp, l_force);
    l_wt = l_omega;

    predl_x = l_x + l_xt*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    predl_w = l_w + l_wt*l_dt;

    l_xtt = drift(predl_x, predl_w, l_amp, l_force);
    l_wtt = l_omega;

    predl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    predl_w = l_w + 0.5f*(l_wt + l_wtt)*l_dt;

    l_xtt = drift(predl_x, predl_w, l_amp, l_force);
    l_wtt = l_omega;

    corrl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state)*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state) + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt, l_state);
    corrl_w = l_w + 0.5f*(l_wt + l_wtt)*l_dt;
}

__device__ void eulermaruyama(float &nl_x, float l_x, float &nl_w, float l_w, curandState *l_state, \
                         float l_amp, float l_omega, float l_force, float l_Dg, int l_2ndorder, float l_Dp, float l_lambda, int l_comp, \
                         int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
/* simplified weak order 1.0 regular euler-maruyama scheme 
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 508, 
  C. Kim, E. Lee, P. Talkner, and P.Hanggi; Phys. Rev. E 76; 011109; 2007 ) 
*/ 
{
    float l_xt, l_wt;

    l_xt = l_x + ( drift(l_x, l_w, l_amp, l_force) + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state) )*l_dt 
               + diffusion(l_Dg, l_dt, l_2ndorder, l_state) 
               + regular_jump_poisson(l_lambda, l_Dp, l_comp, l_dt, l_state);
    l_wt = l_w + l_omega*l_dt;

    nl_x = l_xt;
    nl_w = l_wt;
}

__device__ void fold(float &nx, float x, float y, float &nfc, float fc)
//reduce periodic variable to the base domain
{
    nx = x - floor(x/y)*y;
    nfc = fc + floor(x/y)*y;
}

__global__ void run_moments(float *d_x, float *d_w, float *d_xb, float *d_dx, curandState *d_states)
//actual moments kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x, l_w, l_xb, l_dx; 
    curandState l_state;

    //cache path and model parameters in local variables
    l_x = d_x[idx];
    l_w = d_w[idx];
    l_xb = d_xb[idx];
    l_state = d_states[idx];

    float l_amp, l_omega, l_force, l_Dg, l_Dp, l_lambda, l_mean, l_fa, l_fb, l_mua, l_mub;
    int l_comp;

    l_amp = d_amp;
    l_omega = d_omega;
    l_force = d_force;
    l_Dg = d_Dg;
    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_mean = d_mean;
    l_comp = d_comp;
    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;

    //run simulation for multiple values of the system parameters
    long ridx = (idx/d_paths) % d_points;
    l_dx = d_dx[ridx];

    switch(d_domainx) {
        case 'a':
            l_amp = l_dx;
            break;
        case 'w':
            l_omega = l_dx;
            break;
        case 'f':
            l_force = l_dx;
            break;
        case 'D':
            l_Dg = l_dx;
            break;
        case 'p':
            l_Dp = l_dx;
            if (l_mean != 0.0f) l_lambda = (l_mean*l_mean)/l_Dp;
            break;
        case 'l':
            l_lambda = l_dx;
            if (l_mean != 0.0f) l_Dp = (l_mean*l_mean)/l_lambda;
            break;
        case 'i':
            l_fa = l_dx;
            if (l_comp == 1) {
                l_fb = -l_fa*l_mub/l_mua;
            } else if (l_mean != 0.0f) {
                l_fb = (l_mean*(l_mua + l_mub) - l_fa*l_mub)/l_mua;
            }
            break;
        case 'j':
            l_fb = l_dx;
            if (l_comp == 1) {
                l_fa = -l_fb*l_mua/l_mub;
            } else if (l_mean != 0.0f) {
                l_fa = (l_mean*(l_mua + l_mub) - l_fb*l_mua)/l_mub;
            }
            break;
        case 'm':
            l_mua = l_dx;
            if (l_comp == 1) {
                l_mub = -l_fb*l_mua/l_fa;
            } else if (l_mean != 0.0f) {
                l_mub = (l_fb - l_mean)*l_mua/(l_mean - l_fa);
            }
            break;
        case 'n':
            l_mub = l_dx;
            if (l_comp == 1) {
                l_mua = -l_fa*l_mub/l_fb;
            } else if (l_mean != 0.0f) {
                l_mua = (l_fa - l_mean)*l_mub/(l_mean - l_fb);
            }
            break;
    }

    //step size & number of steps
    float l_dt;
    long l_steps, l_trigger, i;

    if (l_fa != 0.0f) {
        float tmp, taua, taub;

        taua = 1.0f/l_mua;
        taub = 1.0f/l_mub;
        tmp = 2.0f*PI/l_omega;

        if (taua < taub) {
            if (taua < tmp) {
                l_dt = taua/d_spp;
            } else {
                l_dt = tmp/d_spp;
            }
        } else {
            if (taub < tmp) {
                l_dt = taub/d_spp;
            } else {
                l_dt = tmp/d_spp;
            }
        }
    } else {
        l_dt = 2.0f*PI/l_omega/d_spp;
    }

    l_steps = d_steps;
    l_trigger = d_trigger;

    //counters for folding
    float xfc, wfc;
    
    xfc = 0.0f;
    wfc = 0.0f;

    int l_2ndorder, pcd, dcd, dst;
    l_2ndorder = d_2ndorder;

    //jump countdowns
    if (l_2ndorder && l_Dp != 0.0f) pcd = (int) floor( -logf( curand_uniform(&l_state) )/l_lambda/l_dt + 0.5f );

    if (l_fa != 0.0f) {
        float rn;
        rn = curand_uniform(&l_state);

        if (rn < 0.5f) {
            dst = 0;
            dcd = (int) floor( -logf( curand_uniform(&l_state) )/l_mua/l_dt + 0.5f);
        } else {
            dst = 1;
            dcd = (int) floor( -logf( curand_uniform(&l_state) )/l_mub/l_dt + 0.5f);
        }
    }
    
    for (i = 0; i < l_steps; i++) {

        //algorithm
        if (l_2ndorder) {
            predcorr(l_x, l_x, l_w, l_w, pcd, pcd, &l_state, l_amp, l_omega, l_force, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, \
                     dcd, dcd, dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
        } else {
            eulermaruyama(l_x, l_x, l_w, l_w, &l_state, l_amp, l_omega, l_force, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, \
                     dcd, dcd, dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
        }
        
        //fold path parameters
        if ( fabs(l_x) > 1.0f ) {
            fold(l_x, l_x, 1.0f, xfc, xfc);
        }

        if ( l_w > (2.0f*PI) ) {
            fold(l_w, l_w, (2.0f*PI), wfc, wfc);
        }

        if (i == l_trigger) {
            l_xb = l_x + xfc;
        }

    }

    //write back path parameters to the global memory
    d_x[idx] = l_x + xfc;
    d_w[idx] = l_w;
    d_xb[idx] = l_xb;
    d_states[idx] = l_state;
}

__global__ void run_traj(float *d_x, float *d_w, curandState *d_states)
//actual trajectory kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x, l_w; 
    curandState l_state;

    //cache path and model parameters in local variables
    l_x = d_x[idx];
    l_w = d_w[idx];
    l_state = d_states[idx];

    float l_amp, l_omega, l_force, l_Dg, l_Dp, l_lambda, l_fa, l_fb, l_mua, l_mub;
    int l_comp;

    l_amp = d_amp;
    l_omega = d_omega;
    l_force = d_force;
    l_Dg = d_Dg;
    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_comp = d_comp;
    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;

    //step size & number of steps
    float l_dt;
    long l_steps, i;

    if (l_fa != 0.0f) {
        float tmp, taua, taub;

        taua = 1.0f/l_mua;
        taub = 1.0f/l_mub;
        tmp = 2.0f*PI/l_omega;

        if (taua < taub) {
            if (taua < tmp) {
                l_dt = taua/d_spp;
            } else {
                l_dt = tmp/d_spp;
            }
        } else {
            if (taub < tmp) {
                l_dt = taub/d_spp;
            } else {
                l_dt = tmp/d_spp;
            }
        }
    } else {
        l_dt = 2.0f*PI/l_omega/d_spp;
    }

    l_steps = d_steps;

    //counters for folding
    float xfc, wfc;
    
    xfc = 0.0f;
    wfc = 0.0f;

    int l_2ndorder, pcd, dcd, dst;
    l_2ndorder = d_2ndorder;

    //jump countdowns
    if (l_2ndorder && l_Dp != 0.0f) pcd = (int) floor( -logf( curand_uniform(&l_state) )/l_lambda/l_dt + 0.5f );

    if (l_fa != 0.0f) {
        float rn;
        rn = curand_uniform(&l_state);

        if (rn < 0.5f) {
            dst = 0;
            dcd = (int) floor( -logf( curand_uniform(&l_state) )/l_mua/l_dt + 0.5f);
        } else {
            dst = 1;
            dcd = (int) floor( -logf( curand_uniform(&l_state) )/l_mub/l_dt + 0.5f);
        }
    }

    for (i = 0; i < l_steps; i++) {

        //algorithm
        if (l_2ndorder) {
            predcorr(l_x, l_x, l_w, l_w, pcd, pcd, &l_state, l_amp, l_omega, l_force, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, \
                     dcd, dcd, dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
        } else {
            eulermaruyama(l_x, l_x, l_w, l_w, &l_state, l_amp, l_omega, l_force, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, \
                     dcd, dcd, dst, dst, l_fa, l_fb, l_mua, l_mub, l_dt);
        }
        
        //fold path parameters
        if ( fabs(l_x) > 1.0f ) {
            fold(l_x, l_x, 1.0f, xfc, xfc);
        }

        if ( l_w > (2.0f*PI) ) {
            fold(l_w, l_w, (2.0f*PI), wfc, wfc);
        }

    }

    //write back path parameters to the global memory
    d_x[idx] = l_x + xfc;
    d_w[idx] = l_w;
    d_states[idx] = l_state;
}

void prepare()
//prepare simulation
{
    //grid size
    h_paths = (h_paths/h_block)*h_block;
    h_threads = h_paths;

    if (h_moments) h_threads *= h_points;

    h_grid = h_threads/h_block;

    //number of steps
    if (h_traj) {
        h_steps = h_spp;
    } else {
        h_steps = h_periods*h_spp;
    }
    cudaMemcpyToSymbol(d_steps, &h_steps, sizeof(long));
     
    //host memory allocation
    size_f = h_threads*sizeof(float);
    size_ui = h_threads*sizeof(unsigned int);
    size_p = h_points*sizeof(float);

    h_x = (float*)malloc(size_f);
    h_w = (float*)malloc(size_f);
    h_seeds = (unsigned int*)malloc(size_ui);

    //create & initialize host rng
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerate(gen, h_seeds, h_threads);
 
    //device memory allocation
    cudaMalloc((void**)&d_x, size_f);
    cudaMalloc((void**)&d_w, size_f);
    cudaMalloc((void**)&d_seeds, size_ui);
    cudaMalloc((void**)&d_states, h_threads*sizeof(curandState));

    //copy seeds from host to device
    cudaMemcpy(d_seeds, h_seeds, size_ui, cudaMemcpyHostToDevice);

    //initialization of device rng
    init_dev_rng<<<h_grid, h_block>>>(d_seeds, d_states);

    free(h_seeds);
    cudaFree(d_seeds);

    //moments specific requirements
    if (h_moments) {
        h_trigger = h_steps*h_trans;
        cudaMemcpyToSymbol(d_trigger, &h_trigger, sizeof(long));

        h_xb = (float*)malloc(size_f);
        h_dx = (float*)malloc(size_p);

        float dxtmp = h_beginx;
        float dxstep = (h_endx - h_beginx)/h_points;

        long i;
        
        //set domainx
        for (i = 0; i < h_points; i++) {
            if (h_logx) {
                h_dx[i] = pow(10.0f, dxtmp);
            } else {
                h_dx[i] = dxtmp;
            }
            dxtmp += dxstep;
        }
        
        cudaMalloc((void**)&d_xb, size_f);
        cudaMalloc((void**)&d_dx, size_p);
    
        cudaMemcpy(d_dx, h_dx, size_p, cudaMemcpyHostToDevice);
    }
}

void copy_to_dev()
{
    cudaMemcpy(d_x, h_x, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size_f, cudaMemcpyHostToDevice);
    if (h_moments) {
        cudaMemcpy(d_xb, h_xb, size_f, cudaMemcpyHostToDevice);
    }
}

void copy_from_dev()
{
    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w, d_w, size_f, cudaMemcpyDeviceToHost);
    if (h_moments) {
        cudaMemcpy(h_xb, d_xb, size_f, cudaMemcpyDeviceToHost);
    }
}

void initial_conditions()
//set initial conditions for path parameters
{
    curandGenerateUniform(gen, h_x, h_threads); //x in (0,1]
    curandGenerateUniform(gen, h_w, h_threads);

    long i;

    for (i = 0; i < h_threads; i++) {
        h_w[i] *= 2.0f*PI; //w in (0,2\pi]
    }

    if (h_moments) {
        memset(h_xb, 0, size_f);
    }
    
    copy_to_dev();
}

void moments(float *av)
//calculate the first moment of v
{
    float sx, sxb, dt, tempo, tmp, taua, taub;
    int i, j;

    cudaMemcpy(h_x, d_x, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xb, d_xb, size_f, cudaMemcpyDeviceToHost);

    for (j = 0; j < h_points; j++) {
        sx = 0.0f;
        sxb = 0.0f;

        for (i = 0; i < h_paths; i++) {
            sx += h_x[j*h_paths + i];
            sxb += h_xb[j*h_paths + i];
        }

        if (h_domainx == 'w') {
            tempo = 2.0f*PI/h_dx[j];
            dt = tempo/h_spp;
        } else {
            tempo = 2.0f*PI/h_omega;
            dt = tempo/h_spp;
        }

        if (h_domainx == 'm') {
            taua = 1.0f/h_dx[j];
            taub = 1.0f/h_mub;

            if (h_comp) {
                tmp = 1.0f/(-h_fb*h_dx[j]/h_fa);
            } else if (h_mean != 0.0f) {
                tmp = (h_fb - h_mean)*h_dx[j]/(h_mean - h_fa);
            } else {
                tmp = taub;
            }

            if (taua <= tmp) {
                if (taua < tempo) {
                    dt = taua/h_spp;
                } else {
                    dt = tempo/h_spp;
                }
            } else {
                if (tmp < tempo) {
                    dt = tmp/h_spp;
                } else {
                    dt = tempo/h_spp;
                }
            }
        } else if (h_domainx == 'n') {
            taua = 1.0f/h_mua;
            taub = 1.0f/h_dx[j];

            if (h_comp) {
                tmp = 1.0f/(-h_fa*h_dx[j]/h_fb);
            } else if (h_mean != 0.0f) {
                tmp = (h_fa - h_mean)*h_dx[j]/(h_mean - h_fb);
            } else {
                tmp = taua;
            }

            if (taub <= tmp) {
                if (taub < tempo) {
                    dt = taub/h_spp;
                } else {
                    dt = tempo/h_spp;
                }
            } else {
                if (tmp < tempo) {
                    dt = tmp/h_spp;
                } else {
                    dt = tempo/h_spp;
                }
            }
        } else if (h_fa != 0.0f) {
            taua = 1.0f/h_mua;
            taub = 1.0f/h_mub;

            if (taua < taub) {
                if (taua < tempo) {
                    dt = taua/h_spp;
                } else {
                    dt = tempo/h_spp;
                }
            } else {
                if (taub < tempo) {
                    dt = taub/h_spp;
                } else {
                    dt = tempo/h_spp;
                }
            }
        }

        av[j] = (sx - sxb)/( (1.0f - h_trans)*h_steps*dt )/h_paths;

    }
}

void ensemble_average(float *h_x, float &sx)
//calculate ensemble average
{
    int i;

    sx = 0.0f;

    for (i = 0; i < h_threads; i++) {
        sx += h_x[i];
    }

    sx /= h_threads;
}

void finish()
//free memory
{

    free(h_x);
    free(h_w);
    
    curandDestroyGenerator(gen);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_states);
    
    if (h_moments) {
        free(h_xb);
        free(h_dx);

        cudaFree(d_xb);
        cudaFree(d_dx);
    }
}

int main(int argc, char **argv)
{
    parse_cla(argc, argv);
    if (!h_moments && !h_traj && !h_hist) {
        usage(argv);
        return -1;
    }

    prepare();
    
    initial_conditions();
    
    //asymptotic long time average velocity <<v>>
    if (h_moments) {
        float *av;
        int i;

        av = (float*)malloc(size_p);

        if ( !strcmp(h_domain, "1d") ) {
            run_moments<<<h_grid, h_block>>>(d_x, d_w, d_xb, d_dx, d_states);
            moments(av);

            printf("#%c <<v>>\n", h_domainx);
            for (i = 0; i < h_points; i++) {
                printf("%e %e\n", h_dx[i], av[i]);
            }

        } else {
            float h_dy, dytmp, dystep;
            int j;
            
            dytmp = h_beginy;
            dystep = (h_endy - h_beginy)/h_points;
            
            printf("#%c %c <<v>>\n", h_domainx, h_domainy);
            
            for (i = 0; i < h_points; i++) {
                if (h_logy) {
                    h_dy = pow(10.0f, dytmp);
                } else {
                    h_dy = dytmp;
                }

                switch(h_domainy) {
                    case 'a':
                        cudaMemcpyToSymbol(d_amp, &h_dy, sizeof(float));
                        break;
                    case 'w':
                        h_omega = h_dy;
                        cudaMemcpyToSymbol(d_omega, &h_omega, sizeof(float));
                        break;
                    case 'f':
                        cudaMemcpyToSymbol(d_force, &h_dy, sizeof(float));
                        break;
                    case 'D':
                        cudaMemcpyToSymbol(d_Dg, &h_dy, sizeof(float));
                        break;
                    case 'p':
                        cudaMemcpyToSymbol(d_Dp, &h_dy, sizeof(float));
                        break;
                    case 'l':
                        cudaMemcpyToSymbol(d_lambda, &h_dy, sizeof(float));
                        break;
                    case 'i':
                        cudaMemcpyToSymbol(d_fa, &h_dy, sizeof(float));
                        break;
                    case 'j':
                        cudaMemcpyToSymbol(d_fb, &h_dy, sizeof(float));
                        break;
                    case 'm':
                        h_mua = h_dy;
                        cudaMemcpyToSymbol(d_mua, &h_mua, sizeof(float));
                        break;
                    case 'n':
                        h_mub = h_dy;
                        cudaMemcpyToSymbol(d_mub, &h_mub, sizeof(float));
                        break;
                }

                run_moments<<<h_grid, h_block>>>(d_x, d_w, d_xb, d_dx, d_states);
                moments(av);
                
                for (j = 0; j < h_points; j++) {
                    printf("%e %e %e\n", h_dx[j], h_dy, av[j]);
                }

                //blank line for plotting purposes
                printf("\n");

                initial_conditions();

                dytmp += dystep;
            }
        }

        free(av);
    }

    //ensemble averaged trajectory <x>(t)
    if (h_traj) {
        float t, sx;
        int i;

        printf("#t <x>\n");

        for (i = 0; i < h_periods; i++) {
            run_traj<<<h_grid, h_block>>>(d_x, d_w, d_states);
            copy_from_dev();
            t = i*2.0f*PI/h_omega;
            ensemble_average(h_x, sx);
            printf("%e %e\n", t, sx);
        }
    }

    //the final position x of all paths
    if (h_hist) {
        int i;

        run_traj<<<h_grid, h_block>>>(d_x, d_w, d_states);
        copy_from_dev();

        printf("#x v\n");
        
        for (i = 0; i < h_threads; i++) {
            printf("%e\n", h_x[i]); 
        }
    }

    finish();

    return 0;
}
