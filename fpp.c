/******************************************************************************/
/******** Compute the first passage properties of a foraging particle. ********/
/******** example:                                                     ********/
/*** ./fpp conf-1                                                           ***/
/***                                                                        ***/
/******** Compile with: gcc -O3 fpp.c -o fpp                           ********/
/***                                                                        ***/
/*** Input: path to the configuration file with                             ***/
/***        model and simulation parameters                                 ***/
/*** Output: file with the statistics                                       ***/
/***                                                                        ***/
/*** Configuration file consists of three blocks                            ***/
/***      [model]                                                           ***/
/***      [simulation]                                                      ***/
/***      [result]                                                          ***/
/***        model and simulation parameters                                 ***/
/***      - Distribution of time intervals.                                 ***/
/***      - Parameters for time interval distribution.                      ***/
/***      - Distribution of replenishments.                                 ***/
/***     - 'ic': Initial position (X0).                                     ***/
/***     - 'drift': Drift     (alpha).                                      ***/
/***     - 'n_steps': Number of Metropolis steps.                           ***/
/***     - 'traj_len': Maximum trajectory length.                           ***/
/***     - 'r_traj': Jumps to change per Metropolis step.                   ***/
/***     - 'imp_s': Importance sampling (exponential tilt).                 ***/
/***     - the observable to study 'n' or 'T'                               ***/
/***     - 'hist': min and max values of the histogram                      ***/
/***     - 'n_bins': number of bins in the histogram                        ***/
/*** Output:                                                                ***/
/***     - file with the statistics:                                        ***/
/***        - first line: parameters of the simulation                      ***/
/***        - second line: mean and variance                                ***/
/***        - following lines contain four quantities separated by space    ***/
/***            1. center of the bin                                        ***/
/***            2. number of samples in the bin                             ***/
/***            3. sum of relative log likelihood ratios                    ***/
/***            4. Log likelihood ratio corresponding to the bin center     ***/
/***        - the probability accumulated in the bin is                     ***/
/***            exp("3") * exp(-"4") / n_steps                              ***/
/***                                                                        ***/
/*** Copyright (c) 2024 Ivan Burenev                                        ***/
/******************************************************************************/

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

typedef struct parameters {
    /* the structure containing all the parameters for simulation */

    /* properties of the underlying process */
    double alpha;               /* drift velocity */
    double X0;                  /* initial position */

    /* dynamics */
    /* time intervals */
    char t_distribution[20];  /* distribution type */
    int tau_n_parameters;       /* number of parameters in the distribution */
    double t_parameters[10];  /* array of parameters */

    /* energy replenishments */
    char M_distribution[20];    /* distribution type */
    int M_n_parameters;         /* number of parameters in the distribution */
    double M_parameters[10];    /* array of parameters */

    /* properties of the Metropolis algorithm */
    int trajectory_length;      /* maximum length of the trajectory */
    char observable;            /* quantity of interest, 'T' or 'n'
                                   T -- lifetime of the particle
                                   n -- number of jumps before death */
    char tilt_type[100];        /* type of the tilt in IS scheme */
    double theta_is;            /* importance sampling quasi temperature */
    long long n_steps;          /* number of steps in metropolis simulation */
    int n_changes;              /* number of jumps to change per step */
    char filename[100];         /* name of the file where the last
                                    trajectory is stored */

    /* parameters of the output */
    double x_min, x_max;        /* endpoints of the histogram */
    int n_bins;                 /* number of bins in the histogram */
    double delta;               /* width of the bin */
} Parameters;


typedef struct result_data {
    /* results */
    double mean_n, variance_n;      /* the mean and the variance of n */
    double mean_tau, variance_tau;  /* the mean and the variance of tau */
    int *hist_counts;           /* number of samples in the bin */
    double *bin_centers;        /* centers of the bins */
    double *hist_weighted;      /* weighted histogram */
    long long overshoot;        /* number of trajectories with no first
                                   passage happening */
    long long acc;              /* number of accepted metropolis moves */
    double T_trust;             /* trust threshold (minimal time with
                                   overshooting) */
    char filename_data[100];    /* name of the file to store the histogram */
} Result_data;

typedef struct trajectory {
    double *waiting_times;      /* waiting times */
    double *jumps;              /* jumps */
    double T_fp;                /* first passage time */
    int n_fp;                   /* number of jumps before death */
} Trajectory;



#define MAX_LINE_LENGTH 256



/******************************************************************************
 *  @brief Routine loading the parameters from the configuration file.
 *
 *  This function reads a configuration file line by line, parsing and
 *  assigning values to the `parameters` structure based on section headers
 *  and key-value pairs. It supports three configuration sections
 *  (`[model]`, `[simulation]`, and `[result]`).
 *
 *  The function handles the following sections and their keys:
 *  - **[model]**
 *    - `drift`: Sets the drift velocity (`alpha`).
 *    - `X0`: Sets the initial position.
 *    - `trajectory_length`: Defines the maximum trajectory length.
 *    - `t_distribution`: Specifies the time interval distribution type and
 *      its associated parameter.
 *    - `M_distribution`: Specifies the energy replenishment distribution
 *      type and its associated parameter.
 *  - **[simulation]**
 *    - `n_changes`: Defines the number of trajectory changes per Metropolis
 *      step.
 *    - `importance_sampling`: Specifies the importance sampling type and
 *      its parameter.
 *    - `n_steps`: Sets the total number of Metropolis steps.
 *  - **[result]**
 *    - `observable`: Specifies the quantity to observe (`T` or `n`).
 *    - `hist_min`, `hist_max`: Define the histogram range.
 *    - `n_bins`: Specifies the number of histogram bins.
 *
 *  @param [in] filename
 *      Path to the configuration file.
 *  @param [out] parameters
 *      Pointer to the `Parameters` structure that will be populated with
 *      the parsed values from the configuration file.
 *
 *  @return int
 *      - `0` if the configuration was loaded successfully.
 *      - `-1` if an error occurred (e.g., file could not be opened or invalid
 *        configuration values were detected).
 *
 *  @details
 *  - The function skips empty lines and comments (lines starting with `#`).
 *  - Section headers are enclosed in square brackets (e.g., `[model]`).
 *  - Key-value pairs are parsed and validated. Unknown keys or invalid values
 *    will result in an error and terminate the parsing process.
 *  - For distribution types, only predefined options (`exponential`,
 *    `half_gaussian`, `fixed`, `uniform`) are accepted.
 *
 *  @note
 *  - Errors are reported to `stderr` using `perror` and `fprintf`.
 *****************************************************************************/
int load_parameters(const char *filename,
                    Parameters *parameters) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Could not open configuration file");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    char section[MAX_LINE_LENGTH] = "";

    while (fgets(line, sizeof(line), file)) {
        /* Remove newline characters */
        line[strcspn(line, "\n")] = 0;

        /* Skip empty lines and comments */
        if (line[0] == '#' || strlen(line) == 0) continue;

        /* Detect section headers */
        if (line[0] == '[') {
            sscanf(line, "[%[^]]", section);
            continue;
        }

        /* Parsing key-value pairs */
        if (strcmp(section, "model") == 0) {
            if (strstr(line, "drift") == line) {
                sscanf(line, "drift = %lf", &parameters->alpha);
            } else if (strstr(line, "X0") == line) {
                sscanf(line, "X0 = %lf", &parameters->X0);
            } else if (strstr(line, "trajectory_length") == line) {
                sscanf(line, "trajectory_length = %d", &parameters->trajectory_length);
            } else if (strstr(line, "t_distribution") == line) {
                // note that sscanf breaks on the space
                sscanf(line, "t_distribution = %s", parameters->t_distribution);
                if (strcmp(parameters->t_distribution, "exponential") == 0 ||
                    strcmp(parameters->t_distribution, "half_gaussian") == 0 ||
                    strcmp(parameters->t_distribution, "fixed") == 0 ||
                    strcmp(parameters->t_distribution, "uniform") == 0) {
                    // all four distributions have one parameter
                    parameters->tau_n_parameters = 1;

                    char param_str[MAX_LINE_LENGTH];
                    sscanf(line, "t_distribution = %*s [%[^]]", param_str);
                    parameters->t_parameters[0] = atof(param_str);

                    } else {
                        fprintf(stderr, "Error: Unknown t_distribution %s\n",
                                         parameters->t_distribution);
                        fclose(file);
                        return -1;
                    }
            } else if (strstr(line, "M_distribution") == line) {
                sscanf(line, "M_distribution = %s", parameters->M_distribution);
                if (strcmp(parameters->M_distribution, "exponential") == 0 ||
                    strcmp(parameters->M_distribution, "half_gaussian") == 0 ||
                    strcmp(parameters->M_distribution, "fixed") == 0 ||
                    strcmp(parameters->M_distribution, "uniform") == 0) {

                    parameters->M_n_parameters = 1;
                    char param_str[MAX_LINE_LENGTH];
                    sscanf(line, "M_distribution = %*s [%[^]]", param_str);
                    parameters->M_parameters[0] = atof(param_str);
                    } else {
                        fprintf(stderr, "Error: Unknown M_distribution %s\n",
                                parameters->M_distribution);
                        fclose(file);
                        return -1;
                    }
            }
        } else if (strcmp(section, "simulation") == 0) {
            if (strstr(line, "n_changes") == line) {
                sscanf(line, "n_changes = %d", &parameters->n_changes);
            } else if (strstr(line, "importance_sampling") == line) {
                sscanf(line, "importance_sampling = %s", parameters->tilt_type);

                if (strcmp(parameters->tilt_type, "exponential") == 0) {
                    char param_str[MAX_LINE_LENGTH];
                    sscanf(line, "importance_sampling = %*s [%[^]]", param_str);
                    parameters->theta_is = atof(param_str);
                } else if (strcmp(parameters->tilt_type, "none") == 0) {
                    parameters->theta_is = 0;
                } else {
                    fprintf(stderr, "Error: Unknown importance_sampling type %s\n",
                            parameters->tilt_type);
                    fclose(file);
                    return -1;
                }
            } else if (strstr(line, "n_steps") == line) {
                sscanf(line, "n_steps = %lld", &parameters->n_steps);
            }
        } else if (strcmp(section, "result") == 0) {
            if (strstr(line, "observable") == line) {
                sscanf(line, "observable = %c", &parameters->observable);
            } else if (strstr(line, "hist_min") == line) {
                sscanf(line, "hist_min = %lf", &parameters->x_min);
            } else if (strstr(line, "hist_max") == line) {
                sscanf(line, "hist_max = %lf", &parameters->x_max);
            } else if (strstr(line, "n_bins") == line) {
                sscanf(line, "n_bins = %d", &parameters->n_bins);
            }
        }
    }

    fclose(file);
    return 0;
}



/******************************************************************************
 * @brief Generate a random variable from an exponential distribution.
 *
 * This function generates a random variable `x` from an exponential
 * distribution with probability density function:
 *
 *      p(x) = λ * exp(-λ * x)  for x >= 0
 *
 * where λ is the rate parameter of the distribution.
 *
 * @param [in] lambda
 *      The rate parameter ( λ>0 ) of the exponential distribution.
 *
 * @return double
 *      A random variable sampled from the exponential distribution.
 *
 * @details
 * - The function uses the inverse transform sampling method, which
 *   transforms a uniform random variable `u` into an exponential random
 *   variable.
 * - The transformation is given by:
 *
 *      x = - log(1 - u) / λ
 *
 *   where `u` is a uniformly distributed random variable in the range [0, 1).
 *****************************************************************************/
double generate_exponential(const double lambda)
{
    const double u = rand() / (RAND_MAX + 1.0);
    return - log(1 - u) / lambda;
}



/******************************************************************************
 * @brief Generate a random variable from a uniform distribution.
 *
 * This function generates a random variable `x` from a uniform distribution
 * defined over the interval `[low, high]` with probability density function:
 *
 *      p(x) = 1 / (high - low)  for low <= x <= high
 *
 * @param [in] low
 *      The lower bound of the uniform distribution interval.
 * @param [in] high
 *      The upper bound of the uniform distribution interval. It must satisfy
 *      `high > low`.
 *
 * @return double
 *      A random variable sampled from the uniform distribution.
 *
 * @details
 * - The function uses a uniform random variable `u` in the range [0, 1) to
 *   generate the output using the formula:
 *
 *      x = low + u * (high - low)
 *
 *****************************************************************************/
double generate_uniform(const double low, const double high) {
    const double u = rand() / (RAND_MAX + 1.0);
    return low + u * (high - low);
}



/******************************************************************************
 * @brief Generate a random variable from a half-Gaussian distribution.
 *
 * This function generates a random variable `x` from a half-Gaussian
 * distribution with probability density function:
 *
 *      p(x) = sqrt(2 / π) * λ * exp(-λ² * x² / 2)  for x >= 0
 *
 * where λ is the shape parameter of the distribution.
 *
 * @param [in] lambda
 *      The shape parameter (λ) of the half-Gaussian distribution.
 *
 * @return double
 *      A random variable sampled from the half-Gaussian distribution.
 *
 * @details
 * - The function uses the Box-Muller transform to generate standard normal
 *   variables and then scales them to create a half-Gaussian distribution
 *   with required variance.
 * - The procedure involves:
 *   1. Generating two independent uniform random variables `u1` and `u2`
 *      in the range [0, 1).
 *   2. Computing a standard normal variable `z0` using the Box-Muller
 *      transform:
 *
 *          z0 = sqrt(-2 * log(u1)) * cos(2π * u2)
 *
 *   3. Scaling and taking the absolute value:
 *
 *          x = |z0 / λ|
 *****************************************************************************/
double generate_half_gaussian(const double lambda) {
    const double u1 = rand() / (RAND_MAX + 1.0);
    const double u2 = rand() / (RAND_MAX + 1.0);

    // Box-Muller transform
    const double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return fabs( z0 / lambda );
}



/******************************************************************************
 * @brief Generate a jump value based on a specified probability distribution.
 *
 * This function generates a jump `M` according to the
 * distribution specified in the simulation parameters. It supports several
 * distribution types: fixed, exponential, uniform, and half-Gaussian.
 *
 * @param [in] parameters
 *      Pointer to the `Parameters` structure containing the simulation
 *      settings. The relevant fields are:
 *      - `M_distribution`: The type of distribution for jumps.
 *        Supported values are:
 *          - `"fixed"`: Fixed jumps.
 *          - `"exponential"`: Exponential distribution.
 *          - `"uniform"`: Uniform distribution.
 *          - `"half_gaussian"`: Half-Gaussian distribution.
 *      - `M_parameters`: Array of parameters for the specified distribution.
 *        The required parameters depend on the distribution type:
 *          - `"fixed"`: [M0] (single fixed value).
 *          - `"exponential"`: [λ] (rate parameter).
 *          - `"uniform"`: [b] (upper bound; lower bound is 0).
 *          - `"half_gaussian"`: [λ] (variance parameter).
 *
 * @return double
 *      A random jump value sampled from the specified distribution.
 *
 * @details
 * - **Fixed jumps**: Returns a constant value:
 *
 *        M = M_parameters[0]
 *
 * - **Exponential jumps**: Uses `generate_exponential` to sample from:
 *
 *        p(M) = λ * exp(-λ * M)  for M >= 0
 *
 * - **Uniform jumps**: Uses `generate_uniform` to sample from:
 *
 *        p(M) = 1 / b  for 0 <= M <= b
 *
 * - **Half-Gaussian jumps**: Uses `generate_half_gaussian` to sample from:
 *
 *        p(M) = sqrt(2 / π) * λ * exp(-λ² * M² / 2)  for M >= 0
 *
 * @note
 * - If an unknown distribution type is specified, the function will return `0`
 *****************************************************************************/
double generate_M(const Parameters *parameters)
{
    if (strcmp(parameters->M_distribution, "fixed") == 0) {
        return  parameters->M_parameters[0];
    }
    if (strcmp(parameters->M_distribution, "exponential") == 0) {
        return  generate_exponential(parameters->M_parameters[0]);
    }
    if (strcmp(parameters->M_distribution, "uniform") == 0) {
        return  generate_uniform(0,  parameters->M_parameters[0]);
    }
    if (strcmp(parameters->M_distribution, "half_gaussian") == 0) {
        return  generate_half_gaussian(parameters->M_parameters[0]);
    }

    return 0;
}



/******************************************************************************
 * @brief Generate a single time interval based on a specified probability
 *        distribution.
 *
 * This function generates a time interval `t` according to the distribution
 * specified in the simulation parameters. It supports several distribution
 * types: fixed, exponential, uniform, and half-Gaussian.
 *
 * @param [in] parameters
 *      Pointer to the `Parameters` structure containing the simulation
 *      settings. The relevant fields are:
 *      - `t_distribution`: The type of distribution for time intervals.
 *        Supported values are:
 *          - `"fixed"`: Fixed time intervals.
 *          - `"exponential"`: Exponential distribution.
 *          - `"uniform"`: Uniform distribution.
 *          - `"half_gaussian"`: Half-Gaussian distribution.
 *      - `t_parameters`: Array of parameters for the specified distribution.
 *        The required parameters depend on the distribution type:
 *          - `"fixed"`: [t0] (single fixed value).
 *          - `"exponential"`: [λ] (rate parameter).
 *          - `"uniform"`: [b] (upper bound; lower bound is 0).
 *          - `"half_gaussian"`: [λ] (shape parameter).
 *
 * @return double
 *      A random time interval sampled from the specified distribution.
 *
 * @details
 * - **Fixed time intervals**: Returns a constant value:
 *
 *        t = t_parameters[0]
 *
 * - **Exponential time intervals**: Uses `generate_exponential` to sample from:
 *
 *        p(t) = λ * exp(-λ * t)  for t >= 0
 *
 * - **Uniform time intervals**: Uses `generate_uniform` to sample from:
 *
 *        p(t) = 1 / b  for 0 <= t <= b
 *
 * - **Half-Gaussian time intervals**: Uses `generate_half_gaussian` to sample
 *   from:
 *
 *        p(t) = sqrt(2 / π) * λ * exp(-λ² * t² / 2)  for t >= 0
 *
 * @note
 * - If an unknown distribution type is specified, the function will return `0`
 *****************************************************************************/
double generate_t(const Parameters *parameters)
{
    if (strcmp(parameters->t_distribution, "fixed") == 0) {
        return  parameters->t_parameters[0];
    }
    if (strcmp(parameters->t_distribution, "exponential") == 0) {
        return  generate_exponential(parameters->t_parameters[0]);
    }
    if (strcmp(parameters->t_distribution, "uniform") == 0) {
        return  generate_uniform(0, parameters->t_parameters[0]);
    }
    if (strcmp(parameters->t_distribution, "half_gaussian") == 0) {
        return  generate_half_gaussian(parameters->t_parameters[0]);
    }
    return 0;
}



/*** compute first passage properties                                       ***/
/*** the results are stored in T_fp and n_fp                                ***/
/*** return: 1 -- first passage happens                                     ***/
/***            T_fp -- lifetime;    n_fp -- last jump with positive energy ***/
/***        -1 -- no first passage                                          ***/
/***            T_fp -- total time of the process;  n_fp -- length of the trajectory    ***/
/*** NB: n_fp > 0 (the definition of the model)                             ***/
int compute_fp(int *n_fp, double *T_fp,
               const Parameters parameters,
               double const *M, double const *tau)
{
    double X_current = parameters.X0;

    /* Initialize first passage properties */
    *T_fp = 0;
    *n_fp = 0;

    /* Check when the trajectory reaches zero */
    for (int i = 0; i < parameters.trajectory_length; i++) {
        X_current += M[i] - parameters.alpha * tau[i];
        *T_fp += tau[i];
        *n_fp += 1;
        if (X_current <= 0) { /* Check whether the trajectory reached zero */
            *T_fp += X_current / parameters.alpha;
            return 1;
        }
    }

    return -1;
}

/*** log of the likelihood ratio P(T,n) / Q(T,n)                            ***/
/***    P(T,n) -- original distribution                                     ***/
/***    Q(T,n) -- biased distribution used in importance sampling           ***/
/*** the biased distribution is specified in simulation parameters          ***/
double ln_w(const int n_fp, const double T_fp,
            const Parameters *simulation_parameters) {
    switch (simulation_parameters->observable) {
        default:
            return 0;
        case 'n':
            return simulation_parameters->theta_is * (double) n_fp;
        case 'T':
            return simulation_parameters->theta_is * T_fp;
    }
}

/*** initialize the simulation by loading the command line parameters       ***/
int initialize_simulation(const int argc, char *argv[],
                          Parameters *simulation_parameters) {

    if (load_parameters(argv[1], simulation_parameters) == -1) {
        fprintf(stderr, "Error while passing arguments\n");
        return -1;
    }

    simulation_parameters->delta =
            (simulation_parameters->x_max - simulation_parameters->x_min)
                    / (double) simulation_parameters->n_bins;;

    /* if the quantity of interest is discrete, bin width should be integer */
    if ( simulation_parameters->observable == 'n') {
        simulation_parameters->delta = ceil(simulation_parameters->delta);
    }


    /* create a directories for the output */
    mkdir("metropolis_conf", S_IRWXU | S_IRWXG | S_IRWXO);
    sprintf(simulation_parameters->filename, /* file with trajectory */
            "metropolis_conf/"
            "%s-%.2f-%s-%.2f-X0=%.3f-traj_len=%d-IS=%.4f-%c-conf",
            simulation_parameters->t_distribution,
            simulation_parameters->t_parameters[0],
            simulation_parameters->M_distribution,
            simulation_parameters->M_parameters[0],
            simulation_parameters->X0,
            simulation_parameters->trajectory_length,
            simulation_parameters->theta_is,
            simulation_parameters->observable);

    return 0;
}


int initialize_result(const Parameters *simulation_parameters,
                      Result_data *result_data){

    mkdir("metropolis_data", S_IRWXU | S_IRWXG | S_IRWXO);
    sprintf(result_data->filename_data, /* file with final histograms */
                "metropolis_data/"
                "%s-%.2f-%s-%.2f-X0=%.3f-traj_len=%d-IS=%.4f-%c-hist",
                simulation_parameters->t_distribution,
                simulation_parameters->t_parameters[0],
                simulation_parameters->M_distribution,
                simulation_parameters->M_parameters[0],
                simulation_parameters->X0,
                simulation_parameters->trajectory_length,
                simulation_parameters->theta_is,
                simulation_parameters->observable);

    result_data->mean_n = 0;
    result_data->variance_n = 0;
    result_data->mean_tau = 0;
    result_data->variance_tau = 0;
    result_data->T_trust = -1;
    result_data->acc = 0;
    result_data->overshoot = 0;

    result_data->hist_counts =
            (int *) malloc( simulation_parameters->n_bins * sizeof(int) );
    result_data->bin_centers =
            (double *) malloc( simulation_parameters->n_bins * sizeof(double) );
    result_data->hist_weighted =
            (double *) malloc( simulation_parameters->n_bins * sizeof(double) );


    /* fill the histograms with zeros and compute the center of the bins */
    for(int i = 0; i < simulation_parameters->n_bins; i++) {
        result_data->hist_weighted[i] = 0;
        result_data->hist_counts[i] = 0;
        result_data->bin_centers[i] = simulation_parameters->x_min
                                    + simulation_parameters->delta * (double) i
                                    + 0.5 * simulation_parameters->delta;

        /* for the discrete observable center the bins */
        if (simulation_parameters->observable == 'n') {
            result_data->bin_centers[i] += -.5;
        }
    }

    return 0;
}

int initialize_trajectory(Trajectory trajectory,
                            const Parameters *simulation_parameters) {

    /* load the trajectory from the file or generate it from scratch */
    FILE *fptr = fopen(simulation_parameters->filename, "r");
    if (fptr == NULL) {
        for (int i=0; i < simulation_parameters->trajectory_length; i++) {
            trajectory.waiting_times[i] = generate_t(simulation_parameters);
            trajectory.jumps[i] = generate_M(simulation_parameters);
        }
    } else {
        for (int i =0; i < simulation_parameters->trajectory_length; i++) {
            fscanf(fptr, "%lf %lf",
                          &trajectory.jumps[i], &trajectory.waiting_times[i]);
        }
        fclose(fptr);
    }
    /* compute first passage properties */
    compute_fp(&trajectory.n_fp, &trajectory.T_fp,
        *simulation_parameters, trajectory.jumps,trajectory.waiting_times);
    return 1;
}




int save_results(const Parameters *simulation_parameters,
                const Result_data *result_data) {

    FILE *fptr = fopen(result_data->filename_data, "w");

    /* metadata */
    fprintf(fptr,"# X0: %f \n",            simulation_parameters->X0);
    fprintf(fptr,"# n_steps: %lld \n",     simulation_parameters->n_steps);
    fprintf(fptr,"# acc: %lld \n",         result_data->acc);
    fprintf(fptr,"# overshoot: %lld \n",   result_data->overshoot);
    fprintf(fptr,"# theta_is: %f \n",      simulation_parameters->theta_is);
    fprintf(fptr,"# observable: %c \n",    simulation_parameters->observable);
    fprintf(fptr,"# mean_n: %f \n",        result_data->mean_n);
    fprintf(fptr,"# variance_n: %f \n",    result_data->variance_n);
    fprintf(fptr,"# mean_tau: %f \n",      result_data->mean_tau);
    fprintf(fptr,"# variance_tau: %f \n",  result_data->variance_tau);
    fprintf(fptr,"# T_trust: %f \n",       result_data->T_trust);

    /* histograms */
    for(int bin_id = 0; bin_id < simulation_parameters->n_bins; bin_id++){
        const double average_ln_w =  ln_w(result_data->bin_centers[bin_id],
                                          result_data->bin_centers[bin_id],
                                          simulation_parameters)
                    + log(simulation_parameters->n_steps)
                    + log(result_data->bin_centers[1]-result_data->bin_centers[0]);
        fprintf(fptr, "%f  %d  %f  %f  \n",
                result_data->bin_centers[bin_id],
                result_data->hist_counts[bin_id],
                result_data->hist_weighted[bin_id],
                average_ln_w);
    }

    fclose(fptr);
    return 1;
}

/* metropolis step */
/* perform one step and update trajectory */
/* if parameters.n_changes == -1, then the trajectory is generated from scratch */
/* return 1 if the trajectory is changed and 0 if it stayed the same */

/* in the case where the trajectory is generated from scratch
 * it is better to compute the first passage properties on the fly
 * as it is much faster
 */

/* in the case where only part of the trajectory is updated, we update the trajectory
 * and then revert the changes if needed. This is because the acceptance rate is higher than 50%
 * therofore this approach is more efficient.
 */
int metropolis_step(
    Trajectory trajectory,
    const Parameters simulation_parameters,
    int *indices_to_change,
    double *t_old,
    double *M_old
    ) {
    int step_accepted = 0;
    bool overshoot = false; // flag to keep track of the overshooting in the importance sampling scheme

    if (simulation_parameters.n_changes == -1 ) {
        /* trajectory is generated from scratch */
        trajectory.n_fp = 0;
        trajectory.T_fp = 0;
        double X_current = simulation_parameters.X0;

        for (int i=0; i < simulation_parameters.trajectory_length; i++) {
            trajectory.waiting_times[i] = generate_t(&simulation_parameters);
            trajectory.jumps[i] = generate_M(&simulation_parameters);

            X_current += trajectory.jumps[i] - simulation_parameters.alpha * trajectory.waiting_times[i];

            trajectory.T_fp += trajectory.waiting_times[i];
            trajectory.n_fp += 1;

            /* first passage terminates generation */
            if (X_current <= 0) {
                trajectory.T_fp += X_current / simulation_parameters.alpha;
                break;
            }
        }

        step_accepted = 1;
    }

    if (simulation_parameters.n_changes > 0 ){
        const double T_fp_old = trajectory.T_fp;
        const int n_fp_old = trajectory.n_fp;


        int min_jump_index = simulation_parameters.trajectory_length;


        /* generate new steps  */
        for(int i = 0; i < simulation_parameters.n_changes; i++) {
            /* pick a random index [0, trajectory_length) */
            indices_to_change[i] = rand()
                                   % simulation_parameters.trajectory_length;

            /* store old jumps */
            t_old[i] = trajectory.waiting_times[indices_to_change[i]];
            M_old[i] = trajectory.jumps[indices_to_change[i]];

            /* generate new jumps */
            trajectory.waiting_times[indices_to_change[i]] = generate_t(&simulation_parameters);
            trajectory.jumps[indices_to_change[i]] = generate_M(&simulation_parameters);

            /* update the minimum index of the jump if necessary*/
            if (indices_to_change[i] < min_jump_index) {
                min_jump_index = indices_to_change[i];
            }
        }

        double pAcc = 1;

        /* if all updates happened after the trajectory reached zero,
         * then the first passage properties remain the same
         * and hence the Metropolis move is always accepted
         * else we need to compute the acceptance probability */
        if (min_jump_index <= n_fp_old) {
            int first_passage = compute_fp(&trajectory.n_fp, &trajectory.T_fp,
                                            simulation_parameters,
                                            trajectory.jumps,trajectory.waiting_times);

            /* if there is no first passage then in the importance sampling the proposed move is always rejected */
            if ((first_passage == -1) && (simulation_parameters.theta_is != 0)) {
                pAcc = 0;
                overshoot = true;
            } else {
                pAcc *= exp( ln_w(trajectory.n_fp, trajectory.T_fp, &simulation_parameters)
                            - ln_w(n_fp_old, T_fp_old, &simulation_parameters) );
            }

            /* accept/reject step */
            if( rand() / (RAND_MAX + 1.0) < pAcc ){
                step_accepted = 1;
            } else {
                /* the new trajectory is rejected
                                 * and the changes should be reverted */
                step_accepted = 0;
                for(int i = simulation_parameters.n_changes - 1; i >=0; i--) {
                    /* NB as the indices may repeat, the loop should iterate in
                     * the opposite direction to revert the changes properly. */
                    trajectory.waiting_times[indices_to_change[i]] = t_old[i];
                    trajectory.jumps[indices_to_change[i]] = M_old[i];
                }

                /* restore first passage properties to its original values */
                trajectory.T_fp = T_fp_old;
                trajectory.n_fp = n_fp_old;
            }
        }
    }

    if (overshoot) { return -1; }
    return step_accepted;
}

int main(int argc, char *argv[]) {
    /**************************************************************************/
    /************************ Initialization Routine **************************/
    /**************************************************************************/

    Parameters simulation_parameters;
    Result_data result_data;
    Trajectory trajectory;

    if (initialize_simulation(argc, argv,
                              &simulation_parameters) == -1){
        fprintf(stderr, "Error while initializing simulation\n");
                              }
    initialize_result(&simulation_parameters, &result_data);

    /* allocate the memory for the trajectory */
    trajectory.waiting_times = (double *)
        malloc( simulation_parameters.trajectory_length * sizeof(double) );
    trajectory.jumps = (double *)
        malloc( simulation_parameters.trajectory_length * sizeof(double) );
    trajectory.n_fp = 0;
    trajectory.T_fp = 0;

    initialize_trajectory(trajectory, &simulation_parameters);

    int changes_array_size  = simulation_parameters.n_changes;
    if (changes_array_size < 0){ changes_array_size = 1; }

    int *indices_to_change = (int *) malloc(changes_array_size * sizeof(int));
    double *t_old  = (double *) malloc( changes_array_size * sizeof(double) );
    double *M_old  = (double *) malloc( changes_array_size * sizeof(double) );

    /**************************************************************************/
    /************************  Metropolis algorithm  **************************/
    /**************************************************************************/

    /* timers to estimate the execution time and print the progress */
    clock_t time_start, time_current;
    time_start = clock();
    int time_taken_s, time_left_s;
    double it_per_sec;


    for (long long step=0; step < simulation_parameters.n_steps; step++) {

        /* simple command line progress bar */
        if( ceil( (double) 100 * step / simulation_parameters.n_steps)
            == floor( (double) 100 * step / simulation_parameters.n_steps ) ){
            time_current = clock();
            time_taken_s = ( (int) time_current - time_start) / CLOCKS_PER_SEC;
            it_per_sec = step / time_taken_s;
            long long steps_left = simulation_parameters.n_steps - step;
            time_left_s = (int) floor( (double) steps_left / it_per_sec );

            printf("In progress %d %% time elapsed %d s time left %d s",
                    (int) (100 * (step+1) / simulation_parameters.n_steps),
                    time_taken_s, time_left_s );
            fflush(stdout);
            printf("\r");
        }

        /* the main part of the Metropolis */
        const int step_is_accepted = metropolis_step(trajectory, simulation_parameters, indices_to_change, t_old, M_old);
        if (step_is_accepted == -1) {
            /* if there is an overshoot in the importance sampling scheme */
            result_data.overshoot ++;
        }
        if (step_is_accepted == 1) {
            result_data.acc ++;

            if (trajectory.n_fp == simulation_parameters.trajectory_length) {
                if ( result_data.T_trust == -1
                    || result_data.T_trust > trajectory.T_fp ){
                    result_data.T_trust = trajectory.T_fp;
                    }
            }
        }


        /* update the mean values */
        result_data.mean_n += trajectory.n_fp
                            / (double) simulation_parameters.n_steps;
        result_data.mean_tau += trajectory.T_fp
                            / (double) simulation_parameters.n_steps;
        /* update the variance */
        /* NB! this is actually second moment, the shift is done later */
        result_data.variance_n += trajectory.n_fp * trajectory.n_fp
                            / (double) simulation_parameters.n_steps;

        result_data.variance_tau += trajectory.T_fp * trajectory.T_fp
                / (double) simulation_parameters.n_steps;

        /* update the result with the obtained values */
        double observable = 0;

        if (simulation_parameters.observable=='n') {
            observable = (double) trajectory.n_fp;
        }
        if(simulation_parameters.observable=='T') {
            observable = trajectory.T_fp ;
        }

        /* compute the id of the bin in the histogram */
        int bin_id = (int) floor( (observable - simulation_parameters.x_min)
                                  / simulation_parameters.delta );
        if(bin_id >= 0 && bin_id < simulation_parameters.n_bins) {
            /* counts is increased by one */
            result_data.hist_counts[bin_id] += 1;
            /* in the weighted histogram we rescale the weight by a constant
             * to avoid numerical overflow */
            double average_ln_w =  ln_w(result_data.bin_centers[bin_id],
                                        result_data.bin_centers[bin_id],
                                        &simulation_parameters);
            double ln_w_current =  ln_w(trajectory.n_fp, trajectory.T_fp, &simulation_parameters);
            result_data.hist_weighted[bin_id] += exp( - ln_w_current
                                                      + average_ln_w);
        }

    }

    /* now we shift the value of the variance */
    result_data.variance_n = result_data.variance_n
                            - result_data.mean_n * result_data.mean_n;
    result_data.variance_tau = result_data.variance_tau
                            - result_data.mean_tau * result_data.mean_tau;
    printf("\n"); /* for the progress bar to stop */


    /*************************************************************************/
    /*************************  Saving the output  ***************************/
    /*************************************************************************/

    /* basic information is printed in the command line */
    if(result_data.overshoot>0) {
        printf("Warning: overshoot,"
                "consider increasing the maximum length of the trajectory");
    }

    printf("acceptance rate: %.2f\n",
            (double) result_data.acc
                / (double) simulation_parameters.n_steps);

    printf("output saved in: %s", result_data.filename_data);

    /* save the histogram */
    save_results(&simulation_parameters, &result_data);

    /* save trajectory */
    FILE *fptr = fopen(simulation_parameters.filename, "w");
    for (int i =0; i<simulation_parameters.trajectory_length; i++) {
        fprintf(fptr, "%f %f \n", trajectory.jumps[i], trajectory.waiting_times[i]);
    }
    fclose(fptr);

    /* free all dynamically allocated memory to avoid leaks */
    free(trajectory.waiting_times);
    free(trajectory.jumps);

    /* free the memory for the update steps */
    free(indices_to_change);
    free(t_old);
    free(M_old);

    /* free the memory in the histogram */
    free(result_data.bin_centers);
    free(result_data.hist_counts);
    free(result_data.hist_weighted);

    return 0;
}
