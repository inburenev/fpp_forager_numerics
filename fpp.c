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
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

typedef struct parameters {
    /* the structure containing all the parameters for simulation */

    /* properties of the underlying process */
    double alpha;               /* energy decay rate */
    double X0;                  /* initial position */

    /* dynamics */
    /* time intervals */
    char tau_distribution[20];  /* distribution type */
    int tau_n_parameters;       /* number of parameters in the distribution */
    double tau_parameters[10];  /* array of parameters */

    /* energy replenishments */
    char M_distribution[20];    /* distribution type */
    int M_n_parameters;         /* number of parameters in the distribution */
    double M_parameters[10];    /* array of parameters */

    /* properties of the Metropolis algorithm */
    int trajectory_length;      /* maximum length of the trajectory */
    char observable;            /* quantity of interest, 'T' or 'n'
                                   T -- lifetime of the particle
                                   n -- number of jumps before death */
    char tilt_type[10];         /* type of the tilt in IS scheme */
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
    double *tau, *M;            /* time intervals and replenishments */
    double n_fp, T_fp;          /* first passage properties */
} Trajectory;



#define MAX_LINE_LENGTH 256
/*** loads the simulation parameters in from the configuration file         ***/
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
        // Remove newline characters
        line[strcspn(line, "\n")] = 0;

        // Skip empty lines and comments
        if (line[0] == '#' || strlen(line) == 0) continue;

        // Detect section headers
        if (line[0] == '[') {
            sscanf(line, "[%[^]]", section);
            continue;
        }

        // Parsing key-value pairs
        if (strcmp(section, "model") == 0) {
            if (strstr(line, "drift") == line) {
                sscanf(line, "drift = %lf", &parameters->alpha);
            } else if (strstr(line, "X0") == line) {
                sscanf(line, "X0 = %lf", &parameters->X0);
            } else if (strstr(line, "trajectory_length") == line) {
                sscanf(line, "trajectory_length = %d", &parameters->trajectory_length);
            } else if (strstr(line, "tau_distribution") == line) {
                // note that sscanf breaks on the space
                sscanf(line, "tau_distribution = %s", parameters->tau_distribution);
                if (strcmp(parameters->tau_distribution, "exponential") == 0 ||
                    strcmp(parameters->tau_distribution, "half_gaussian") == 0 ||
                    strcmp(parameters->tau_distribution, "fixed") == 0 ||
                    strcmp(parameters->tau_distribution, "uniform") == 0) {
                    // all four distributions have one parameter
                    parameters->tau_n_parameters = 1;

                    char param_str[MAX_LINE_LENGTH];
                    sscanf(line, "tau_distribution = %*s [%[^]]", param_str);
                    parameters->tau_parameters[0] = atof(param_str);

                    } else {
                        fprintf(stderr, "Error: Unknown tau_distribution %s\n",
                                         parameters->tau_distribution);
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


/*** generate random variable from an exponential distribution              ***/
/*** probability density: p(x) = lambda * exp( - lambda * x )               ***/
double generate_exponential(const double lambda)
{
    const double u = rand() / (RAND_MAX + 1.0);
    return - log(1 - u) / lambda;
}

/*** generate random variable from uniform distribution on [a,b]            ***/
double generate_uniform(const double low, const double high) {
    const double u = rand() / (RAND_MAX + 1.0);
    return low + u * (high - low);
}

/*** generate random variable from half gaussian distribution               ***/
/*** probability density:                                                   ***/
/***        p(x) =  sqrt(2/pi) * lambda * exp( - lambda^2 * x^2 / 2 )       ***/
double generate_half_gaussian(const double lambda) {
    const double u1 = rand() / (RAND_MAX + 1.0);
    const double u2 = rand() / (RAND_MAX + 1.0);

    // Box-Muller transform
    const double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return fabs( z0 / lambda );
}





/*** generate a replenishment                                               ***/
/*** the probability distribution is specified in simulation parameters     ***/
double generate_M(const Parameters *parameters)
{
    /* fixed replenishment */
    if (strcmp(parameters->M_distribution, "fixed") == 0) {
        return  parameters->M_parameters[0];
    }
    /* exponential replenishment */
    if (strcmp(parameters->M_distribution, "exponential") == 0) {
        return  generate_exponential(parameters->M_parameters[0]);
    }
    /* uniform replenishment */
    if (strcmp(parameters->M_distribution, "uniform") == 0) {
        return  generate_uniform(0,  parameters->M_parameters[0]);
    }
    /* uniform replenishment */
    if (strcmp(parameters->M_distribution, "half_gaussian") == 0) {
        return  generate_half_gaussian(parameters->M_parameters[0]);
    }

    return 0;
}

/*** generate a single time interval                                        ***/
/*** the probability distribution is specified in simulation parameters     ***/
double generate_tau(const Parameters *parameters)
{
    /* fixed replenishment */
    if (strcmp(parameters->tau_distribution, "fixed") == 0) {
        return  parameters->tau_parameters[0];
    }
    /* exponential replenishment */
    if (strcmp(parameters->tau_distribution, "exponential") == 0) {
        return  generate_exponential(parameters->tau_parameters[0]);
    }
    /* uniform replenishment */
    if (strcmp(parameters->tau_distribution, "uniform") == 0) {
        return  generate_uniform(0, parameters->tau_parameters[0]);
    }
    /* uniform replenishment */
    if (strcmp(parameters->tau_distribution, "half_gaussian") == 0) {
        return  generate_half_gaussian(parameters->tau_parameters[0]);
    }
    return 0;
}

/*** compute first passage properties                                       ***/
/*** the results are stored in T_fp and n_fp                                ***/
/*** return: 1 -- first passage happens                                     ***/
/***            T_fp -- lifetime;    n_fp -- last jump with positive energy ***/
/***        -1 -- no first passage                                          ***/
/***            T_fp -- total time; n_fp -- length                          ***/
/*** NB: n_fp > 0 (the definition of the model)                             ***/
int compute_fp(int *n_fp, double *T_fp,
               const Parameters parameters,
               double const *M, double const *tau)
{
    double E_current = parameters.X0;

    /* Initialize first passage properties */
    *T_fp = 0;
    *n_fp = 0;

    /* Check when the trajectory reaches zero */
    for (int i = 0; i < parameters.trajectory_length; i++) {
        E_current += M[i] - parameters.alpha * tau[i];
        *T_fp += tau[i];
        *n_fp += 1;
        if (E_current <= 0) { /* Check whether the trajectory reached zero */
            *T_fp += E_current / parameters.alpha;
            return 1;
        }
    }

    *n_fp += 1;
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
            simulation_parameters->tau_distribution,
            simulation_parameters->tau_parameters[0],
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
                simulation_parameters->tau_distribution,
                simulation_parameters->tau_parameters[0],
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

int initialize_trajectory(double *tau, double *M,
                          const Parameters *simulation_parameters) {
    /* load or create the trajectory */
    FILE *fptr = fopen(simulation_parameters->filename, "r");
    if (fptr == NULL) {
        for (int i=0; i < simulation_parameters->trajectory_length; i++) {
            tau[i] = generate_tau(simulation_parameters);
            M[i] = generate_M(simulation_parameters);
        }
    } else {
        for (int i =0; i < simulation_parameters->trajectory_length; i++) {
            fscanf(fptr,"%lf %lf", &M[i], &tau[i]);
        }
        fclose(fptr);
    }
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


int main(int argc, char *argv[]) {

    /**************************************************************************/
    /************************ Initialization Routine **************************/
    /**************************************************************************/

    double *M, *tau;
    double T_fp = 0;
    int n_fp = 0;
    int passage_happens = 1;            /* 1 does happen, -1 does not happen  */

    Parameters simulation_parameters;
    Result_data result_data;

    if (initialize_simulation(argc, argv,
                              &simulation_parameters) == -1){
        fprintf(stderr, "Error while initializing simulation\n");
    }
    initialize_result(&simulation_parameters, &result_data);


    tau = (double *) malloc( simulation_parameters.trajectory_length
                            * sizeof(double) );
    M = (double *) malloc( simulation_parameters.trajectory_length
                            * sizeof(double) );
    initialize_trajectory(tau, M, &simulation_parameters);

    passage_happens = compute_fp(&n_fp, &T_fp, simulation_parameters, M,tau );



    /**************************************************************************/
    /************************  Metropolis algorithm  **************************/
    /**************************************************************************/

    /* timers to estimate the execution time and print the progress */
    clock_t time_start, time_current;
    time_start = clock();
    int time_taken_s, time_left_s;
    double it_per_sec;


    /* auxiliary arrays to store the trajectory before the change */
    int *indices_to_change = (int *) malloc(simulation_parameters.n_changes
                                            * sizeof(int));
    double *tau_old  = (double *) malloc( simulation_parameters.n_changes
                                            * sizeof(double) );
    double *M_old  = (double *) malloc( simulation_parameters.n_changes
                                            * sizeof(double) );


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
        if (simulation_parameters.n_changes == -1 ) {
            n_fp = 0;
            T_fp = 0;
            double E_current = simulation_parameters.X0;
            // n_changes = -1 create the new trajectory at each step
            for (int i=0; i < simulation_parameters.trajectory_length; i++) {
                tau[i] = generate_tau(&simulation_parameters);
                M[i] = generate_M(&simulation_parameters);
                E_current += M[i] - simulation_parameters.alpha * tau[i];
                T_fp += tau[i];
                n_fp += 1;
                if (E_current <= 0) { /* Check whether the trajectory reached zero */
                    break;
                }
            }
            T_fp += E_current / simulation_parameters.alpha;

            if (n_fp == simulation_parameters.trajectory_length) {
                if ( result_data.T_trust == -1
                    || result_data.T_trust > T_fp ){
                    result_data.T_trust = T_fp;
                }
            }

            result_data.acc ++;
        } else {
            /* store old first passage properties */
            const double T_fp_old = T_fp;
            const int n_fp_old = n_fp;

            /* first we perform a change of jumps */
            int min_jump_index = simulation_parameters.trajectory_length;
            for(int i = 0; i < simulation_parameters.n_changes; i++) {
                /* pick a random index [0, trajectory_length) */
                indices_to_change[i] = rand()
                                       % simulation_parameters.trajectory_length;

                /* store old jumps */
                tau_old[i] = tau[indices_to_change[i]];
                M_old[i] = M[indices_to_change[i]];

                /* generate new jumps */
                tau[indices_to_change[i]] = generate_tau(&simulation_parameters);
                M[indices_to_change[i]] = generate_M(&simulation_parameters);

                /* update the minimum index of the jump if necessary*/
                if (indices_to_change[i] < min_jump_index) {
                    min_jump_index = indices_to_change[i];
                }
            }


            if (min_jump_index > n_fp_old) {
                /* if all updates happened after the trajectory reached zero,
                 * then the first passage properties remain the same
                 * and hence the Metropolis move is always accepted */
                result_data.acc ++;
            } else {
                /* at least some of the changes have happened before the
                 * trajectory has reached zero, so we need to update
                 * the first passage properties */
                double pAcc = 1;
                if(compute_fp(&n_fp, &T_fp, simulation_parameters, M,tau ) == -1 ) {
                    /* NB when we call compute_fp, n_fp and T_fp are updated */
                    /* if there is no first passage */
                    result_data.overshoot ++;

                    if (result_data.T_trust > T_fp || result_data.T_trust == -1) {
                        result_data.T_trust = T_fp;
                    }

                    if (simulation_parameters.theta_is != 0) {
                        /* in the importance sampling scheme overshoot = reject the move */
                        pAcc = 0;
                    }
                }


                /* compute the acceptance probability */
                pAcc *= exp( ln_w(n_fp, T_fp, &simulation_parameters)
                            - ln_w(n_fp_old, T_fp_old, &simulation_parameters) );


                if( rand() / (RAND_MAX + 1.0) < pAcc ){
                    result_data.acc ++;
                } else {
                    /* the new trajectory is rejected
                     * and the changes should be reverted */

                    for(int i = simulation_parameters.n_changes - 1; i >=0; i--) {
                        /* NB as the indices may repeat, the loop should iterate in
                         * the opposite direction to revert the changes properly. */
                        tau[indices_to_change[i]] = tau_old[i];
                        M[indices_to_change[i]] = M_old[i];
                    }

                    /* restore first passage properties to its original values */
                    T_fp = T_fp_old;
                    n_fp = n_fp_old;
                }

            }
        }

        /* if there were no overshoot we update the results */
        if (n_fp < simulation_parameters.trajectory_length) {
            /* update the result with the obtained values */
            double observable = 0;

            if (simulation_parameters.observable=='n') {
                observable = (double) n_fp;
            }
            if(simulation_parameters.observable=='T') {
                observable = T_fp ;
            }

            /* update the mean values */
            result_data.mean_n += n_fp
                                / (double) simulation_parameters.n_steps;
            result_data.mean_tau += T_fp
                                / (double) simulation_parameters.n_steps;
            /* update the variance */
            /* NB! this is actually second moment, the shift is done later */
            result_data.variance_n += n_fp * n_fp
                                / (double) simulation_parameters.n_steps;

            result_data.variance_tau += T_fp * T_fp
                    / (double) simulation_parameters.n_steps;


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
                double ln_w_current =  ln_w(n_fp, T_fp, &simulation_parameters);
                result_data.hist_weighted[bin_id] += exp( - ln_w_current
                                                          + average_ln_w);
            }
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

    /* save the histogram and the trajectory */
    save_results(&simulation_parameters, &result_data);

    /* trajectory */
    FILE *fptr = fopen(simulation_parameters.filename, "w");
    for (int i =0; i<simulation_parameters.trajectory_length; i++) {
        fprintf(fptr, "%f %f \n", M[i], tau[i]);
    }
    fclose(fptr);

    /* free all dynamically allocated memory to avoid leaks */
    free(tau);
    free(M);
    free(result_data.bin_centers);
    free(result_data.hist_counts);
    free(result_data.hist_weighted);
    free(indices_to_change);
    free(M_old);
    free(tau_old);

    return 0;
}