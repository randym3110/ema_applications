#pip install ema_workbench

# libreries for EMA
from ema_workbench import (RealParameter, ScalarOutcome, Constant, ArrayOutcome,
                           Model, MultiprocessingEvaluator, ema_logging, perform_experiments,
                           load_results, save_results)
from ema_workbench.analysis import dimensional_stacking
from ema_workbench.analysis.plotting import lines

import os
import tarfile

# libreries for simulation
import numpy as np
import pandas as pd
import matplotlib as plot
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math
from datetime import datetime

#creates a global variable to store all simulations, DO NOT CHANGE
global result_list
result_list = []

# hydrological model
def shaman(u_climate_sce, c_var, hmax_var, imax_var, alfa_var,
           frac_tree, k_vol):
    
    #load the scenario file and store it in "datos"
    file_name = 'datos_' +str(round(u_climate_sce)) + '.csv'
    date_parser = lambda x: datetime.strptime(x, '%d/%m/%Y')
    datos = pd.read_csv(file_name, parse_dates = ['fecha'], date_parser=date_parser)
    
    #====================
    # MODELS
    
    # function of hydrological model
    def hydro_model(Hmax_par, Imax_par, c_par, alfa_par, dt_par, Ho_par, Vo_par, area_par, cob):
            
        for i in range(len(datos)):

            Pi = c_par * (Hmax_par - Ho_par)

            Di = Hmax_par - Ho_par + datos.loc[i, "etp_" + cob]

            if datos.loc[i, "pp"] <= Pi or datos.loc[i, "pp"] + Di - (2 * Pi) < 0:
                Ti = 0
            else:
                Ti = ((datos.loc[i, "pp"] - Pi)**2)/(datos.loc[i, "pp"] + Di - (2 * Pi))

            Temporali = Ho_par + datos.loc[i, "pp"] - Ti - datos.loc[i, "etp_" + cob]
            Hi = max(0, Temporali)

            Temporali = Ho_par + datos.loc[i, "pp"] - Ti
            Etri = min(Temporali, datos.loc[i, "etp_" + cob])

            Ii = Imax_par * (Ti / (Ti + Imax_par))

            Asupi = Ti - Ii

            Ri = area_par * Ii * 1000

            Vi = (Vo_par * np.exp(-1 * alfa_par * dt_par / 2)) + (Ri / alfa_par)*(1 - np.exp(-1 * alfa_par * dt_par / 2))

            if Vo_par <= Vi:
                Asubi = 0
            else:        
                Asubi = Vo_par - Vi + Ri

            datos.loc[i,'Qsub_' + cob]  = Asubi / (datos.loc[i, "dia"] * 24 * 3600)

            datos.loc[i, 'Qsup_' + cob] = Asupi * area_par * 1000 / (datos.loc[i, "dia"] * 24 * 3600)

            #restart values for next time period
            Ho_par = Hi
            Vo_par = Vi
            
    # function for reservoir
    def reservoir(Volmax, dead_per, Height):
        
        Surface = Volmax / Height
        Vo_emb = 0.3 * Volmax
                
        datos["pp_m3"] = datos["pp_o"] * Surface / 1000
        datos["evap_m3"] = datos["evap"] * Surface / 1000
        datos["Qsim_m3"] = datos["Qsim"] * 3600 * 24

        #DAM model
        threshold = 185 * 3600 * 24 #m2
        
        for i in range(len(datos)):
            
            if datos.loc[i, "Qsim_m3"] < threshold:
                Vsal = datos.loc[i, "Qsim_m3"]
                Vspill = 0
                Vfin = max(0, Vo_emb - datos.loc[i, "evap_m3"])
            else:
                Vsal = threshold
                Vres = datos.loc[i, "Qsim_m3"] - threshold + Vo_emb
                
                if Vres > datos.loc[i, "evap_m3"]:
                    Vres_evap = Vres - datos.loc[i, "evap_m3"]
                    
                    if Vres_evap >= Volmax:
                        Vspill = Vres_evap - Volmax
                        Vfin = Volmax
                    else:
                        Vspill = 0
                        Vfin = Vres_evap
                        
                else:
                    Vres_evap = 0
                    Vspill = 0
                    Vfin = 0

            datos.loc[i, "Vfull"] = Vspill + Vsal
            Vo_emb = Vfin

    #====================
    #calibration parameters
    Hmax = 106.508789479969 * (hmax_var)
    Imax = 128.365878555163 * (imax_var)
    cf = 0.600833085831643 * (c_var)
    cp = 0.458942107557758 * (c_var)
    alfa = 0.835030698098722
    Ho = 10
    Vo = 725320.002930137
    dt = 1.78134620412316
    datos["pp"] = datos["pp_o"] * 1
    kc_tree = 0.88
    kc_past = 0.41
    
    # permanent data (km2)
    area = 1999
    
    area1_tree = area * frac_tree
    area2_pasture = area - area1_tree
    frac_pasture = 1 - frac_tree
    
    # original fractions
    frac_o_tree = 0.428860927953542
    frac_o_pasture = 1 - frac_o_tree
    
    #=====================
    # SIMULATION STARTS

    # global for hydro simulation
    datos["etp_tree"] = datos["eto"] * kc_tree
    datos["etp_past"] = datos["eto"] * kc_past
            
    hydro_model(Hmax_par = Hmax * 1,
                Imax_par = Imax * 1,
                c_par = cf,
                alfa_par = alfa * 1,
                dt_par = dt,
                Ho_par = Ho,
                Vo_par = Vo,
                area_par = area1_tree,
                cob = "tree")

    hydro_model(Hmax_par = Hmax * 1,
                Imax_par = Imax * 1,
                c_par = cp,
                alfa_par = alfa * 1,
                dt_par = dt,
                Ho_par = Ho,
                Vo_par = Vo,
                area_par = area2_pasture,
                cob = "past") 
    
    #Full runoff simulation
    datos["Qsim"] = datos["Qsub_tree"] + datos["Qsub_past"] + datos["Qsup_tree"] + datos["Qsup_past"]
    
    #simulate the reservoir
    reservoir(Volmax = 1000000 * k_vol,
              dead_per = 0.05,
              Height = 5)
    
    datos["Vspill_m3/s"] = datos["Vfull"] / 3600 / 24
        
    #============
    #end of simulation
    
    #============
    # metrics for EMA
    
    #Calculates probabilities and return period value columns
    #data = {"nro": list(range(1, 14))} #for historic simulation
    data = {"nro": list(range(1, 30))} #for scenario simulation
    ret_per = pd.DataFrame(data)
    #ret_per["prob"] = ret_per["nro"] / (13 + 1) #for historic simulation
    ret_per["prob"] = ret_per["nro"] / (29 + 1) #for scenario simulation
    ret_per["RT_yr"] = 1 / ret_per["prob"]
    
    #extracts only the date and Qsim from the large simulation dataset
    dat = pd.DataFrame()
    dat["fecha"] = datos["fecha"]
    dat["Qsim"] = datos["Vspill_m3/s"]
    
    #groups the values by year and finds the maximum value per year
    max_pp_o_by_year = dat.groupby(dat['fecha'].dt.year)['Qsim'].max()
    
    #sorts descending the max value per year
    max_pp_o_by_year_sorted = max_pp_o_by_year.sort_values(ascending=False)
    
    # convert max_pp_o_by_year_sorted (time series type) into a dataframe with columns "fecha" and "max"
    max_df = max_pp_o_by_year_sorted.reset_index().rename(columns={'Qsim': 'max'})
    
    # combines the return period column and max value per year in one single dataframe
    max_df["RT_yr"] = ret_per["RT_yr"]
    
    # sets a threshold for the return period in years
    umbral = 2
       
    # Find the closest Return period to umbral
    closest_index = (max_df["RT_yr"]-umbral).abs().idxmin()
    
    # Get the corresponding value
    rp_d = max_df.loc[closest_index, "max"]
    
    # price per policy
    # price for water reservoir
    price_res = 7.5083 * (k_vol ** 0.5324) * 1000000 * 0.66 #price in USD
    
    # price for reforestation / pasture planting
    if frac_tree < frac_o_tree:
        price_cover = (frac_pasture - frac_o_pasture) * area * 100 * 500 * 0.66 # change in pasture extension in km2 converted to ha and multiplied by USD/ha
    else:
        price_cover = (frac_tree - frac_o_tree) * area * 100 * 1000 * 0.66 # change in forest extension in km2 converted to ha and multiplied by USD/ha
    
    price_policy = price_res + price_cover
    
    #accumulate simulations
    export = pd.DataFrame(datos)
    result_list.append(export)
    
    #========
    # return results for EMA analysis
    
    return {'return_period' : rp_d, 'price_policy' : price_policy}

#=====================
#runs model to test historic
shaman(u_climate_sce = 12,
       frac_tree = 0.51820704161536,
       k_vol = 11.7514591866802,
       c_var = 0.997986097451476, hmax_var =1.01668825296601, imax_var = 1.0546684650956, alfa_var = 1.06453815180111
       )

#=====================
# EMA
# the head lines allow a visualization of the model run and identifying when it ends
if __name__ == '__main__':
    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)

    #takes the model to run
    model = Model('simpleModel', function = shaman)  # instantiate the model
        
    # specify process model parameters  xRCP=None, xClimateModel=None
    model.uncertainties = [RealParameter('u_climate_sce', 0.6, 11.8),
                          RealParameter('c_var', 0.9, 1.1),
                          RealParameter('hmax_var', 0.9, 1.1),
                           RealParameter('imax_var', 0.9, 1.1),
                           RealParameter('alfa_var', 0.9, 1.1)
                          ]
    
    # specify polices IntegerParameter
    model.levers = [RealParameter('frac_tree', 0.04, 0.81),
                   RealParameter('k_vol', 0.01, 19.8)
                   ]
   
    # specify outcomes
    model.outcomes = [ScalarOutcome('return_period'),
                     ScalarOutcome('price_policy')]

    # runs the model several times
    results = perform_experiments(model, 70, 70)
print('end!')
