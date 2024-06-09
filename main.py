import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def sir_model(t, y, params):
    """
    SIR model equations
    """
    beta, gamma, sigma, rho, alpha_IQ, alpha_QR = params
    
    S, I, R, E, Q= y
    dSdt = -beta * S * I  + rho * R
    dEdt = beta * S * I  - sigma * E
    dIdt = sigma * E - gamma * I - alpha_IQ*I
    dRdt = gamma * I - rho * R + alpha_QR*Q
    dQdt = alpha_IQ*I - alpha_QR*Q
    return [dSdt, dIdt, dRdt, dEdt, dQdt]

def generate_data(params, initial_conditions, t_points, add_noise=False, seasonal_amplitude=2000, seasonal_period=7):
    """
    Generate simulated data based on SIR model with immunity waning
    """
    beta, gamma, sigma, rho, alpha_IQ, alpha_QR = params
    solution = solve_ivp(sir_model, [t_start, t_end], initial_conditions, args=(params,), t_eval=t_points)
    S_data, I_data, R_data, E_data, Q_data = solution.y

    if add_noise:
        noise_S = np.random.normal(0, 10, len(S_data))  # Increase variance of noise
        S_data += noise_S
        S_data = np.maximum(S_data, 0)  # Ensure infected data doesn't go below 0

        noise_E = np.random.normal(0, 60, len(E_data))  # Increase variance of noise
        E_data += noise_E
        E_data = np.maximum(E_data, 0)  # Ensure exposed data doesn't go below 0

        noise_I = np.random.normal(0, 600, len(I_data))  # Increase variance of noise
        seasonal_variation = seasonal_amplitude * np.sin(2 * np.pi * t_points / seasonal_period)  # Seasonality
        I_data += noise_I + seasonal_variation
        I_data = np.maximum(I_data, 0)  # Ensure infected data doesn't go below 0

        noise_R = np.random.normal(0, 300, len(R_data))
        R_data += noise_R
        R_data = np.maximum(R_data, 0)  # Ensure recovered data doesn't go below 0

    # Round infected values to integers
    I_data = np.round(I_data).astype(int)

    # Store data in a DataFrame
    data = pd.DataFrame({
        'Time': t_points,
        'Susceptible': S_data,
        'Exposed': E_data,
        'Infected': I_data,
        'Recovered': R_data,
        'Quarantine': Q_data,
    })

    # Save DataFrame to CSV
    data.to_csv('simulated_data_3.csv', index=False)

    return S_data, E_data, I_data, R_data, Q_data, 

def loss_function(params):
    """
    Loss function to minimize during parameter estimation
    """
    beta, gamma, sigma, rho, alpha_IQ, alpha_QR = params
    S_data, E_data, I_data, R_data, Q_data = generate_data(params, initial_conditions, t_points)
    error = np.sum((I_data - I_observed) ** 2)
    return error

# Parameters for data generation
N = 100000
initial_conditions = [N - 1, 1, 0]  # [S0, I0, R0]
prob_of_infecting = 1/200
avg_no_contacts_per_individual = 50
beta_true = prob_of_infecting * avg_no_contacts_per_individual / N
gamma_true = 1/14    # Estimated recovery rate
sigma_true = 1/7    # Estimated incubation rate
rho_true = 1/60  # Estimated immunity waning rate

#ADDED
alpha_IQ_true = 1/28
alpha_QR_true = 1/14 

initial_conditions = [N - 1, 1, 0, 0, 0]  # Initial number of susceptible, infected, and recovered individuals

# Time points 
t_start = 0
t_end = 360
t_step = 1
t_points = np.arange(t_start, t_end, t_step)

params_true = beta_true, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true

# Generate observed data based on true parameters
S_observed, E_observed, I_observed, R_observed, Q_observed = generate_data(params_true, initial_conditions, t_points)

# Parameter estimation
beta_guess = beta_true  # Initial guess for transmission rate
gamma_guess = 1/20  # Initial guess for recovery rate
sigma_guess = 1/10  # Initial guess for incubation rate
rho_guess = 1/120  # Initial guess for re-susceptibility rate

#ADDED
alpha_IQ_guess = 1/10
alpha_QR_quess = 1/10


initial_guess = [beta_guess, gamma_guess, sigma_guess, rho_guess, alpha_IQ_guess, alpha_QR_quess]


#Initial guess for scenario 1

initial_guess_1 = [1e-6, 1e-2, 1e-1, 1e-2, 0.1, 0.1]  # Initial guess for no restrictions
initial_guess_2 = [2e-6, 1e-2, 1e-1, 1e-2, 0.1, 0.1]  # Initial guess for restrictions
initial_guess_3 = [5e-6, 1e-2, 1e-1, 1e-2, 0.1, 0.1]  # Initial guess for masks

#Initial guess for scenario 2

initial_guess_seasonal = [1e-6, 1e-2, 1e-1, 1e-2, 0.1, 0.1]


# Initial guesses for the parameters including vaccination rate beta_values = [0.5e-6, 1e-6, 2e-6]

initial_guess_vaccination = [2e-6, 0.07, 0.14, 0.016, 0.1, 0.1, 1e-3]


# Perform parameter estimation using optimization
result = minimize(loss_function, initial_guess, method='Nelder-Mead')
estimated_params = result.x
beta_estimated, gamma_estimated, sigma_estimated, rho_estimated, alpha_IQ_estimated, alpha_QR_estimated = estimated_params

# Generate simulated data based on estimated parameters
S_estimated, E_estimated, I_estimated, R_estimated, Q_estimated = generate_data(estimated_params, initial_conditions, t_points,
                                                      add_noise=False)




def plot_sir_model_estimated(t_points, observed, estimated, params_true, params_estimated, ax, mode):
    if mode != 3:
        S_observed, E_observed, I_observed, R_observed, Q_observed = observed 
        IS_estimated, E_estimated, I_estimated, R_estimated, Q_estimated = estimated
        beta_true, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true = params_true
        beta_estimated, gamma_estimated, sigma_estimated, rho_estimated, alpha_IQ_estimated, alpha_QR_estimated = params_estimated
    elif mode == 3:
        S_observed, E_observed, I_observed, R_observed, Q_observed = observed 
        S_estimated, I_estimated, E_estimated, R_estimated, Q_estimated = estimated
        beta_true, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true, nu_true = params_true
        beta_estimated, gamma_estimated, sigma_estimated, rho_estimated, alpha_IQ_estimated, alpha_QR_estimated, nu_estimated = params_estimated
    
    if mode == 1:
        ax.plot(t_points, S_estimated, '--', label='Susceptible (Estimated)')
        ax.plot(t_points, E_estimated, '--', label='Exposed (Estimated)')
        ax.plot(t_points, R_estimated, '--', label='Recovered (Estimated)')
        ax.plot(t_points, Q_estimated, '--', label='Quarantine (Estimated)')
        ax.plot(t_points, I_estimated, '--', label='Infected (Estimated)')

    elif mode == 2:
        ax.plot(t_points_seasonal, I_estimated, '--', label='Infected (Seasonal) (Estimated)')
        ax.plot(t_points_seasonal, R_estimated, '--', label='Recovered (Seasonal) (Estimated)')
        ax.plot(t_points_seasonal, S_estimated, '--', label='Susceptible (Seasonal) (Estimated)')
        ax.plot(t_points_seasonal, E_estimated, '--', label='Exposed (Seasonal) (Estimated)')
        ax.plot(t_points_seasonal, Q_estimated, '--', label='Quarantine (Seasonal) (Estimated)')

    elif mode == 3:
        ax.plot(t_points, S_estimated, '--', label='Susceptible (Estimated)')
        ax.plot(t_points, E_estimated, '--', label='Exposed (Estimated)')
        ax.plot(t_points, R_estimated, '--', label='Recovered (Estimated)')
        ax.plot(t_points, Q_estimated, '--', label='Quarantine (Estimated)')
        ax.plot(t_points, I_estimated, '--', label='Infected (Estimated)')

    
    # Axis labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Individuals')
    ax.set_title('SIR Model Parameter Estimation')
    # Legend
    ax.legend()

    # Print parameters
    print(f"True Parameters: beta = {beta_true}, gamma = {gamma_true}, sigma = {sigma_true}, rho = {rho_true}")
    print(f"Estimated Parameters: beta = {beta_estimated}, gamma = {gamma_estimated}, sigma = {sigma_estimated}, rho = {rho_estimated}")



def plot_sir_model(t_points, observed, params_true, ax, parameter_text=''):
    S_observed, E_observed, I_observed, R_observed, Q_observed = observed 
    beta_true, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true = params_true

    # Plotting on the provided axes 'ax'
    ax.plot(t_points, I_observed, label='Infected (Observed)')
    ax.plot(t_points, R_observed, label='Recovered (Observed)')
    ax.plot(t_points, Q_observed, label='Quarantine (Observed)')
    ax.plot(t_points, E_observed, label='Exposed (Observed)')
    ax.plot(t_points, S_observed, label='Susceptible (Observed)')

    # Axis labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Individuals')

    # Set title considering parameter_text
    if parameter_text:
        title = f'SIR Model: {parameter_text}'
    else:
        title = 'SIR Model Parameter Estimation'
    ax.set_title(title)

    # Legend
    ax.legend()

    # Optionally print parameters for debugging or logging
    print(f"True Parameters: beta = {beta_true}, gamma = {gamma_true}, sigma = {sigma_true}, rho = {rho_true}")
observed = S_observed, E_observed, I_observed, R_observed, Q_observed
estimated = S_estimated, E_estimated, I_estimated, R_estimated, Q_estimated
params_estimated = beta_estimated, gamma_estimated, sigma_estimated, rho_estimated, alpha_IQ_estimated, alpha_QR_estimated


# CHOOSE ONE 
scenario_option = 'Hospitalizations and social restrictions'
#scenario_option = 'Seasonal'
#scenario_option = 'Vaccination'

# CHOOSE ONE 
#estimate_params_option = True
estimate_params_option = False

if scenario_option == 'Hospitalizations and social restrictions':

    if estimate_params_option == True:
        # Estimate parameters for each scenario with different initial guesses
        result_no_restrictions = minimize(loss_function, initial_guess_1, method='Nelder-Mead')
        estimated_params_no_restrictions = result_no_restrictions.x

        result_restrictions = minimize(loss_function, initial_guess_2, method='Nelder-Mead')
        estimated_params_restrictions = result_restrictions.x

        result_masks = minimize(loss_function, initial_guess_3, method='Nelder-Mead')
        estimated_params_masks = result_masks.x

        # Generate simulated data based on estimated parameters for each scenario
        S_estimated_no_restrictions, E_estimated_no_restrictions, I_estimated_no_restrictions, R_estimated_no_restrictions, Q_estimated_no_restrictions = generate_data(estimated_params_no_restrictions, initial_conditions, t_points, add_noise=False)
        S_estimated_restrictions, E_estimated_restrictions, I_estimated_restrictions, R_estimated_restrictions, Q_estimated_restrictions = generate_data(estimated_params_restrictions, initial_conditions, t_points, add_noise=False)
        S_estimated_masks, E_estimated_masks, I_estimated_masks, R_estimated_masks, Q_estimated_masks = generate_data(estimated_params_masks, initial_conditions, t_points, add_noise=False)


    
    # Set up figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust the figsize based on your needs

    # Parameters for scenario without restrictions
    prob_of_infecting = 0.02
    avg_no_contacts_per_individual = 100
    params_true_1 = beta_true, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true
    S_observed_1, E_observed_1, I_observed_1, R_observed_1, Q_observed_1 = generate_data(params_true_1, initial_conditions, t_points)
    observed_1 = S_observed_1, E_observed_1, I_observed_1, R_observed_1, Q_observed_1 
    # Plot for No Restrictions
    plot_sir_model(t_points, [S_observed_1, E_observed_1, I_observed_1, R_observed_1, Q_observed_1], params_true_1, ax=axes[0])
    axes[0].set_title("Without Restrictions")
    if estimate_params_option:
        plot_sir_model_estimated(t_points, [S_observed_1, E_observed_1, I_observed_1, R_observed_1, Q_observed_1], [S_estimated_no_restrictions, E_estimated_no_restrictions, I_estimated_no_restrictions, R_estimated_no_restrictions, Q_estimated_no_restrictions], params_true_1, estimated_params_no_restrictions, axes[0], 1)

    # Scenario with school and public place closures
    prob_of_infecting = 0.02
    avg_no_contacts_per_individual = 10
    beta_true_2 = prob_of_infecting * avg_no_contacts_per_individual / N
    params_true_2 = beta_true_2, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true
    S_observed_2, E_observed_2, I_observed_2, R_observed_2, Q_observed_2 = generate_data(params_true_2, initial_conditions, t_points)
    observed_2 = S_observed_2, E_observed_2, I_observed_2, R_observed_2, Q_observed_2
    # Text detailing the changes in parameters
    parameter_text = f'prob_of_infecting = {prob_of_infecting}, avg_no_contacts_per_individual = {avg_no_contacts_per_individual}'
    # Plot for Restrictions
    plot_sir_model(t_points, [S_observed_2, E_observed_2, I_observed_2, R_observed_2, Q_observed_2], params_true_2, ax=axes[1], parameter_text=parameter_text)
    axes[1].set_title("Closed Schools and Public Places")
    if estimate_params_option:
        plot_sir_model_estimated(t_points, [S_observed_2, E_observed_2, I_observed_2, R_observed_2, Q_observed_2], [S_estimated_restrictions, E_estimated_restrictions, I_estimated_restrictions, R_estimated_restrictions, Q_estimated_restrictions], params_true_2, estimated_params_restrictions, axes[1], 1)


    # Parameters for scenario with masks
    prob_of_infecting = 0.01
    avg_no_contacts_per_individual = 50
    beta_true_3 = prob_of_infecting * avg_no_contacts_per_individual / N
    params_true_3 = beta_true_3, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true
    S_observed_3, E_observed_3, I_observed_3, R_observed_3, Q_observed_3 = generate_data(params_true_3, initial_conditions, t_points)
    observed_3 = S_observed_3, E_observed_3, I_observed_3, R_observed_3, Q_observed_3
    # Plot for Masks
    plot_sir_model(t_points, [S_observed_3, E_observed_3, I_observed_3, R_observed_3, Q_observed_3], params_true_3, ax=axes[2])
    axes[2].set_title("Masks")
    if estimate_params_option:
        plot_sir_model_estimated(t_points, [S_observed_3, E_observed_3, I_observed_3, R_observed_3, Q_observed_3], [S_estimated_masks, E_estimated_masks, I_estimated_masks, R_estimated_masks, Q_estimated_masks], params_true_3, estimated_params_masks, axes[2], 1)


    # Display the plots
    plt.tight_layout()
    plt.show()


#function for reasonal changes in Beta
def sir_model_seasonal(t, y, params):
    beta, gamma, sigma, rho, alpha_IQ, alpha_QR = params
    
    # Adjust beta periodically for seasonal variation
    if (t % 365) < 90:  # Winter
        beta *= 1.5  # Increase beta by 50%
    elif (t % 365) < 180:  # Spring
        beta *= 1.2  # Increase beta by 20%
    elif (t % 365) < 270:  # Summer
        beta *= 0.8  # Decrease beta by 20%
    else:  # Fall
        beta *= 1.0  # Normal beta

    S, I, R, E, Q = y

    dSdt = -beta * S * I + rho * R
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I - alpha_IQ * I
    dRdt = gamma * I - rho * R + alpha_QR * Q
    dQdt = alpha_IQ * I - alpha_QR * Q
    return [dSdt, dIdt, dRdt, dEdt, dQdt]   

if scenario_option == 'Seasonal':

    if estimate_params_option:
        # Estimate parameters for seasonal scenario
        result_seasonal = minimize(loss_function, initial_guess_seasonal, method='Nelder-Mead')
        estimated_params_seasonal = result_seasonal.x
        
        # Generate simulated data based on estimated parameters for seasonal scenario
        S_estimated_seasonal, E_estimated_seasonal, I_estimated_seasonal, R_estimated_seasonal, Q_estimated_seasonal = generate_data(estimated_params_seasonal, initial_conditions, t_points, add_noise=False)

    # Parameters for seasonal data generation
    params_seasonal = [beta_true, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true]
    t_points_seasonal = np.arange(t_start, t_end, t_step)  # Same time points as your original setup
    
    # Solve the SIR model with seasonal changes
    solution_seasonal = solve_ivp(sir_model_seasonal, [t_start, t_end], initial_conditions, args=(params_seasonal,), t_eval=t_points_seasonal)
    
    # Extract data
    S_seasonal, E_seasonal, I_seasonal, R_seasonal, Q_seasonal = solution_seasonal.y
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a single subplot for the seasonal variation
    
    # Plot estimated data
    ax.plot(t_points_seasonal, I_seasonal, label='Infected (Seasonal)')
    ax.plot(t_points_seasonal, R_seasonal, label='Recovered (Seasonal)')
    ax.plot(t_points_seasonal, S_seasonal, label='Susceptible (Seasonal)')
    ax.plot(t_points_seasonal, E_seasonal, label='Exposed (Seasonal)')
    ax.plot(t_points_seasonal, Q_seasonal, label='Quarantine (Seasonal)')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of Individuals')
    ax.set_title('Seasonal Variation in SIR Model')
    ax.legend()
    
    # If estimation of parameters is enabled, plot the estimated model as well
    if estimate_params_option:
        plot_sir_model_estimated(t_points_seasonal, [S_seasonal, E_seasonal, I_seasonal, R_seasonal, Q_seasonal], 
                                  [S_estimated_seasonal, E_estimated_seasonal, I_estimated_seasonal, R_estimated_seasonal, Q_estimated_seasonal], 
                                  params_seasonal, estimated_params_seasonal, ax, 2)
    
    plt.show()



def sir_model_vaccination(t, y, params):
    beta, gamma, sigma, rho, alpha_IQ, alpha_QR, nu = params

    S, E, I, R, Q = y

    dSdt = -beta * S * I - nu * S + rho * R
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I - alpha_IQ * I
    dRdt = gamma * I - rho * R + alpha_QR * Q + nu * S
    dQdt = alpha_IQ * I - alpha_QR * Q
    return [dSdt, dEdt, dIdt, dRdt, dQdt]


def generate_data_vaccination(params, initial_conditions, t_points, add_noise=False, seasonal_amplitude=2000, seasonal_period=7):
    """
    Generate simulated data based on SIR model with vaccination
    """
    beta, gamma, sigma, rho, alpha_IQ, alpha_QR, nu = params
    solution = solve_ivp(sir_model_vaccination, [t_points[0], t_points[-1]], initial_conditions, args=(params,), t_eval=t_points)
    S_data, E_data, I_data, R_data, Q_data = solution.y

    if add_noise:
        noise_S = np.random.normal(0, 10, len(S_data))
        S_data += noise_S
        S_data = np.maximum(S_data, 0)

        noise_E = np.random.normal(0, 60, len(E_data))
        E_data += noise_E
        E_data = np.maximum(E_data, 0)

        noise_I = np.random.normal(0, 600, len(I_data))
        seasonal_variation = seasonal_amplitude * np.sin(2 * np.pi * t_points / seasonal_period)
        I_data += noise_I + seasonal_variation
        I_data = np.maximum(I_data, 0)

        noise_R = np.random.normal(0, 300, len(R_data))
        R_data += noise_R
        R_data = np.maximum(R_data, 0)

    I_data = np.round(I_data).astype(int)

    data = pd.DataFrame({
        'Time': t_points,
        'Susceptible': S_data,
        'Exposed': E_data,
        'Infected': I_data,
        'Recovered': R_data,
        'Quarantine': Q_data,
    })

    data.to_csv('simulated_data_vaccination.csv', index=False)

    return S_data, E_data, I_data, R_data, Q_data

def loss_function_vaccination(params):
    """
    Loss function to minimize during parameter estimation for vaccination scenario
    """
    beta, gamma, sigma, rho, alpha_IQ, alpha_QR, nu = params
    S_data, E_data, I_data, R_data, Q_data = generate_data_vaccination(params, initial_conditions, t_points)
    error = np.sum((I_data - I_observed) ** 2)
    return error


if scenario_option == 'Vaccination':

    if estimate_params_option:
        # Initial guess for parameters including vaccination rate
        initial_guess_vaccination = [1e-6, 1e-2, 1e-1, 1e-2, 0.1, 0.1, 1e-3]
        
        # Estimate parameters for vaccination scenario
        result_vaccination = minimize(loss_function_vaccination, initial_guess_vaccination, method='Nelder-Mead')
        estimated_params_vaccination = result_vaccination.x
        
        # Generate simulated data based on estimated parameters for vaccination scenario
        S_estimated_vaccination, E_estimated_vaccination, I_estimated_vaccination, R_estimated_vaccination, Q_estimated_vaccination = generate_data_vaccination(estimated_params_vaccination, initial_conditions, t_points, add_noise=False)

    # Parameters for vaccination data generation
    params_vaccination = [beta_true, gamma_true, sigma_true, rho_true, alpha_IQ_true, alpha_QR_true, 1e-3]
    t_points_vaccination = np.arange(t_start, t_end, t_step)  # Same time points as your original setup
    
    # Solve the SIR model with vaccination
    solution_vaccination = solve_ivp(sir_model_vaccination, [t_start, t_end], initial_conditions, args=(params_vaccination,), t_eval=t_points_vaccination)
    
    # Extract data
    S_vaccination, E_vaccination, I_vaccination, R_vaccination, Q_vaccination = solution_vaccination.y
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a single subplot for the vaccination scenario
    
    # Plot estimated data
    ax.plot(t_points_vaccination, I_vaccination, label='Infected (Vaccination)')
    ax.plot(t_points_vaccination, R_vaccination, label='Recovered (Vaccination)')
    ax.plot(t_points_vaccination, S_vaccination, label='Susceptible (Vaccination)')
    ax.plot(t_points_vaccination, E_vaccination, label='Exposed (Vaccination)')
    ax.plot(t_points_vaccination, Q_vaccination, label='Quarantine (Vaccination)')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of Individuals')
    ax.set_title('Impact of Vaccination on Epidemic Spread')
    ax.legend()
    
    # If estimation of parameters is enabled, plot the estimated model as well
    if estimate_params_option:
        plot_sir_model_estimated(t_points_vaccination, [S_vaccination, E_vaccination, I_vaccination, R_vaccination, Q_vaccination], 
                                  [S_estimated_vaccination, E_estimated_vaccination, I_estimated_vaccination, R_estimated_vaccination, Q_estimated_vaccination], 
                                  params_vaccination, estimated_params_vaccination, ax, 3)
    
    plt.show()