/******************************************************************************/
/******** Compute the first passage properties of a foraging particle. ********/
/******** example:                                                     ********/
/*** ./fpp exp_tau 1 fixed_M 0.5 ic 10 rate 1 n_steps 10000000              ***/
/***        traj_len 1000 r_traj 1 imp_s 0 n hist 0 1000 n_bins 1001        ***/
/***                                                                        ***/
/******** Compile with: gcc -O3 fpp.c -o fpp                           ********/
/***                                                                        ***/
/*** Input: model and simulation parameters                                 ***/
/***      - Distribution of time intervals.                                 ***/
/***      - Parameters for time interval distribution.                      ***/
/***      - Distribution of replenishments.                                 ***/
/***     - 'ic': Initial energy (E0).                                       ***/
/***     - 'rate': Decay rate (alpha).                                      ***/
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

typedef struct simulation_parameters {
    /* the structure containing all the parameters for simulation */

    /* properties of the underlying process */
    /* initial conditions */
    double alpha;               /* energy decay rate */
    double E0;                  /* initial energy */

    /* dynamics */
    /* time intervals */
    char tau_process[10];       /* distribution name */
    int tau_n_parameters;       /* number of parameters in the distribution */
    double tau_parameters[10];  /* array of parameters */

    /* energy replenishments */
    char M_process[10];         /* distribution name */
    int M_n_parameters;         /* number of parameters in the distribution */
    double M_parameters[10];    /* array of parameters */

    /* properties of the Metropolis algorithm */
    int trajectory_length;      /* maximum length of the trajectory */
    char quantity_of_interest;  /* quantity of interest, 'T' or 'n'
                                   T -- lifetime of the particle
                                   n -- number of jumps before death */
    double beta_is;             /* importance sampling parameter */
    long long n_steps;          /* number of steps in metropolis simulation */
    int n_changes;              /* number of jumps to change per step */
    char filename[100];         /* name of the file where the last
                                    trajectory is stored */

    /* parameters of the output */
    double x_min, x_max;        /* endpoints of the histogram */
    int n_bins;                 /* number of bins in the histogram */
    double delta;               /* width of the bin */
} Simulation_parameters;


typedef struct result_data {
    /* results */
    double mean, variance;      /* the mean and the variance of the data */
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


/*** parse command line arguments  ***/
// TODO: rewrite this function in a nicer way (using getopt)
int parse_arguments(int argc, char *argv[],
                    Simulation_parameters *simulation_parameters)
{
    /*
     * this function parses arguments into the parameters of the simulation
     * return -1 if the input is wrong
     */

    if (argc < 2 ) {
        fprintf(stderr, "Wrong input parameters \n");
        return -1;
    }

    int argument_index = 1;
    /* we start with reading the parameters of the model */
    // first we should have the process for time intervals
    // for example: exp_tau 1
    if (strcmp(argv[argument_index], "exp_tau") == 0) {
        strcpy(simulation_parameters->tau_process, argv[argument_index]);
        simulation_parameters->tau_n_parameters = 1;
        if (argument_index + 1 < argc) {
            argument_index++;
            simulation_parameters->tau_parameters[0] = atof(argv[argument_index]);
        }
    } else {
        fprintf(stderr, "Invalid process for time intervals.\n");
        return -1;
    }
    argument_index++;


    // then we should have the process for replenishment
    // for example: exp_M 1 or fixed_M 0.5
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "exp_M") == 0
            || strcmp(argv[argument_index], "fixed_M") == 0) {
            strcpy(simulation_parameters->M_process, argv[argument_index]);
            simulation_parameters->M_n_parameters = 1;
            if (argument_index + 1 < argc) {
                argument_index++;
                simulation_parameters->M_parameters[0]
                                        = atof(argv[argument_index]);
            }
            }
    } else {
        fprintf(stderr, "Invalid process for energy replenishments intervals.\n");
        return -1;
    }
    argument_index++;



    // after that we have initial conditions
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "ic") == 0) {
            if (argument_index + 1 < argc) {
                argument_index++;
                simulation_parameters->E0 = atof(argv[argument_index]);
            }
        }
    } else {
        fprintf(stderr, "Invalid initial conditions.\n");
        return -1;
    }
    argument_index++;

    // then the rate of the decay
    if (argument_index  < argc) {
        if (strcmp(argv[argument_index], "rate") == 0) {
            if (argument_index + 1 < argc) {
                argument_index++;
                simulation_parameters->alpha = atof(argv[argument_index]);
            }
        }
    } else {
        fprintf(stderr, "Invalid decay rate.\n");
        return -1;
    }
    argument_index++;

    /* now the model is defined and we proceed to the metropolis parameters */
    // the number of steps in metropolis
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "n_steps") == 0) {
            if (argument_index + 1 < argc) {
                argument_index++;
                simulation_parameters->n_steps = atoll(argv[argument_index]);
            }
        }
    } else {
        fprintf(stderr, "Invalid number of steps.\n");
        return -1;
    }
    argument_index++;

    // maximum length of the trajectory
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "traj_len") == 0) {
            if (argument_index + 1 < argc) {
                argument_index++;
                simulation_parameters->trajectory_length
                                      = atoi(argv[argument_index]);
            }
        }
    } else {
        fprintf(stderr, "Invalid number of steps.\n");
        return -1;
    }
    argument_index++;

    // number of jumps to change per metropolis step
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "r_traj") == 0) {
            if (argument_index + 1 < argc) {
                argument_index++;
                simulation_parameters->n_changes = atoi(argv[argument_index]);
            }
        }
    } else {
        fprintf(stderr, "Invalid number of jumps to change.\n");
        return -1;
    }
    argument_index++;

    // importance sampling parameters
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "imp_s") == 0) {
            if (argument_index + 2 < argc) {
                argument_index++;
                simulation_parameters->beta_is = atof(argv[argument_index]);
                argument_index++;
                simulation_parameters->quantity_of_interest
                                      = argv[argument_index][0];
            }
        }
    } else {
        fprintf(stderr, "Invalid importance sampling parameters.\n");
        return -1;
    }
    argument_index++;

    /* the parameters of the histogram */
    // min and max max_value in the histogram
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "hist") == 0) {
            if (argument_index + 2 < argc) {
                argument_index++;
                simulation_parameters->x_min = atof(argv[argument_index]);
                argument_index++;
                simulation_parameters->x_max = atof(argv[argument_index]);
            }
        }
    } else {
        fprintf(stderr, "Invalid parameters of histogram.\n");
        return -1;
    }
    argument_index++;

    // number of bins in the histogram
    if (argument_index < argc) {
        if (strcmp(argv[argument_index], "n_bins") == 0) {
            if (argument_index + 1 < argc) {
                argument_index++;
                simulation_parameters->n_bins = atoi(argv[argument_index]);
            }
        }
    } else {
        fprintf(stderr, "Invalid parameters of histogram.\n");
        return -1;
    }

    return 0;
}


/*** generate random variable from an exponential distribution              ***/
/*** probability density: p(x) = lambda * exp( - lambda * x )               ***/
double generate_exponential(double lambda)
{
    const double u = rand() / (RAND_MAX + 1.0);
    return -log(1 - u) / lambda;
}


/*** generate a replenishment                                               ***/
/*** the probability distribution is specified in simulation parameters     ***/
double generate_M(const Simulation_parameters *simulation_parameters)
{
    /* fixed replenishment */
    if (strcmp(simulation_parameters->M_process, "fixed_M") == 0) {
        /* fixed replenishment */
        return  simulation_parameters->M_parameters[0];
    }
    /* exponential replenishment */
    if (strcmp(simulation_parameters->M_process, "exp_M") == 0) {
        return  generate_exponential(simulation_parameters->M_parameters[0]);
    }
    return 0;
}

/*** generate a single time interval                                        ***/
/*** the probability distribution is specified in simulation parameters     ***/
double generate_tau(const Simulation_parameters *simulation_parameters)
{
    /* generate a single time interval */
    /* the probability distribution is defined in process_parameters */
    if (strcmp(simulation_parameters->tau_process, "exp_tau") == 0) {
        /* exponential time intervals */
        return generate_exponential(simulation_parameters->tau_parameters[0]);
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
               const Simulation_parameters simulation_parameters,
               double const *M, double const *tau)
{
    double E_current = simulation_parameters.E0;

    /* Initialize first passage properties */
    *T_fp = 0;
    *n_fp = 0;

    /* Check when the trajectory reaches zero */
    for (int i = 0; i < simulation_parameters.trajectory_length; i++) {
        E_current += M[i] - simulation_parameters.alpha * tau[i];
        *T_fp += tau[i];
        *n_fp += 1;
        if (E_current <= 0) { /* Check whether the trajectory reached zero */
            *T_fp += E_current / simulation_parameters.alpha;
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
            const Simulation_parameters simulation_parameters) {
    switch (simulation_parameters.quantity_of_interest) {
        default:
            return 0;
        case 'n':
            return simulation_parameters.beta_is * (double) n_fp;
        case 'T':
            return simulation_parameters.beta_is * T_fp;
    }
}

/*** initialize the simulation by loading the command line parameters       ***/
int initialize_simulation(const int argc, char *argv[],
                          Simulation_parameters *simulation_parameters) {

    if (parse_arguments(argc, argv,
                        simulation_parameters) == -1) {
        fprintf(stderr, "Error while passing arguments\n");
        return -1;
    }

    simulation_parameters->delta =
            (simulation_parameters->x_max - simulation_parameters->x_min)
                    / (double) simulation_parameters->n_bins;;

    /* if the quantity of interest is discrete, bin width should be integer */
    if ( simulation_parameters->quantity_of_interest == 'n') {
        simulation_parameters->delta = ceil(simulation_parameters->delta);
    }


    /* create a directories for the output */
    mkdir("metropolis_conf", S_IRWXU | S_IRWXG | S_IRWXO);
    sprintf(simulation_parameters->filename, /* file with trajectory */
            "metropolis_conf/"
            "%s-%.2f-%s-%.2f-E0=%.0f-traj_len=%d-IS=%.8f-%c-conf",
            simulation_parameters->tau_process,
            simulation_parameters->tau_parameters[0],
            simulation_parameters->M_process,
            simulation_parameters->M_parameters[0],
            simulation_parameters->E0,
            simulation_parameters->trajectory_length,
            simulation_parameters->beta_is,
            simulation_parameters->quantity_of_interest);

    return 0;
}


int initialize_result(const Simulation_parameters *simulation_parameters,
                      Result_data *result_data){

    mkdir("metropolis_data", S_IRWXU | S_IRWXG | S_IRWXO);
    sprintf(result_data->filename_data, /* file with final histograms */
                "metropolis_data/"
                "%s-%.2f-%s-%.2f-E0=%.0f-traj_len=%d-IS=%.8f-%c-hist",
                simulation_parameters->tau_process,
                simulation_parameters->tau_parameters[0],
                simulation_parameters->M_process,
                simulation_parameters->M_parameters[0],
                simulation_parameters->E0,
                simulation_parameters->trajectory_length,
                simulation_parameters->beta_is,
                simulation_parameters->quantity_of_interest);

    result_data->mean = 0;
    result_data->variance = 0;
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
        if (simulation_parameters->quantity_of_interest == 'n') {
            result_data->bin_centers[i] += -.5;
        }
    }

    return 0;
}

int initialize_trajectory(double *tau, double *M,
                          const Simulation_parameters *simulation_parameters) {
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




int main(int argc, char *argv[]) {

    /**************************************************************************/
    /************************ Initialization Routine **************************/
    /**************************************************************************/
    double *M, *tau;
    double T_fp = 0;
    int n_fp = 0;
    int passage_happens = 1;            /* 1 does happen, -1 does not happen  */

    Simulation_parameters simulation_parameters;
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



    /*************************************************************************/
    /************************  Metropolis algorithm  *************************/
    /*************************************************************************/

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
        /* store old first passage properties */
        double T_fp_old = T_fp;
        int n_fp_old = n_fp;

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

                if (simulation_parameters.beta_is != 0) {
                    /* in the importance sampling scheme overshoot =  reject the move */
                    pAcc = 0;
                }
            }


            /* compute the acceptance probability */
            pAcc *= exp( ln_w(n_fp, T_fp, simulation_parameters)
                        - ln_w(n_fp_old, T_fp_old, simulation_parameters) );


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

        /* if there were no overshoot we update the results */
        if (n_fp < simulation_parameters.trajectory_length) {
            /* update the result with the obtained values */
            double observable = 0;

            if (simulation_parameters.quantity_of_interest=='n') {
                observable = (double) n_fp;
            }
            if(simulation_parameters.quantity_of_interest=='T') {
                observable = T_fp ;
            }

            /* update the mean and the variance */
            result_data.mean += observable
                                / (double) simulation_parameters.n_steps;
            /* at the moment this is not the variance but the mean of the square
             * we will subtract the square of the mean later */
            result_data.variance += observable * observable
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
                                            simulation_parameters);
                double ln_w_current =  ln_w(n_fp, T_fp, simulation_parameters);
                result_data.hist_weighted[bin_id] += exp( - ln_w_current
                                                          + average_ln_w);
            }
        }
    }

    /* now we shift the value of the variance */
    result_data.variance = result_data.variance
                            - result_data.mean * result_data.mean;
    printf("\n"); /* for the progress bar to stop */


    /*************************************************************************/
    /*************************  Saving the output  ***************************/
    /*************************************************************************/

    /* basic information is printed in the command line */
    if(result_data.overshoot>0) {
        printf("Warning: mean and variance are wrong due to overshoot, "
                "consider increasing the maximum length of the trajectory");
    }
    printf("mean: %.3f variance: %.3f:\n",
            result_data.mean, result_data.variance);
    printf("acceptance rate: %.2f\n",
            (double) result_data.acc
                / (double) simulation_parameters.n_steps);
    printf("overshoot rate: %.2f\n",
            (double) result_data.overshoot
                / (double) simulation_parameters.n_steps);

    /* trajectory */
    FILE *fptr;
    fptr = fopen(simulation_parameters.filename, "w");
    for (int i =0; i<simulation_parameters.trajectory_length; i++) {
        fprintf(fptr, "%f %f \n", M[i], tau[i]);
    }
    fclose(fptr);

    /* results */
    fptr = fopen(result_data.filename_data, "w");
    fprintf(fptr, "# E0: %f n_steps: %lld acc: %lld overshoot: %lld "
                  "T_trust: %f  beta_is: %f observable: %c \n",
                    simulation_parameters.E0,
                    simulation_parameters.n_steps,
                    result_data.acc,
                    result_data.overshoot,
                    result_data.T_trust,
                    simulation_parameters.beta_is,
                    simulation_parameters.quantity_of_interest);

    fprintf(fptr, "# mean: %f variance: %f \n",
                   result_data.mean, result_data.variance);

    for(int bin_id = 0; bin_id < simulation_parameters.n_bins; bin_id++){
        double average_ln_w =  ln_w(result_data.bin_centers[bin_id],
                                    result_data.bin_centers[bin_id],
                                    simulation_parameters);
        fprintf(fptr, "%f  %d  %f  %f  \n",
                result_data.bin_centers[bin_id],
                result_data.hist_counts[bin_id],
                result_data.hist_weighted[bin_id],
                average_ln_w);
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