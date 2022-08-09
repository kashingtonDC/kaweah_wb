import os
import ee
import datetime
import time
import sklearn
import importlib

import geopandas as gp
import pandas as pd
import numpy as np
import rsfuncs as rs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from sklearn import preprocessing

from tqdm import tqdm_notebook as tqdm

ee.Initialize()

# Load shapefile 
shp = gp.read_file('../shape/study_area/c2vsim_sub_18.shp')

# Make EE objects from shapefiles 
area = rs.gdf_to_ee_poly(shp)

# Load Small watersheds shapefile, dissolve, and simplify it slightly 
sw_shp = gp.read_file('../shape/study_area/small_sheds.shp').dissolve().explode()
sw_area = rs.gdf_to_ee_poly(sw_shp)

# Load RS data dict from rsfuncs.py
data = rs.load_data()

# Set start/end
strstart = '2001-01-01'
strend = '2021-08-31'

startdate = datetime.datetime.strptime(strstart, "%Y-%m-%d")
enddate = datetime.datetime.strptime(strend, "%Y-%m-%d")

print("-------" * 10)
print("Processing Runoff Data")
print("-------" * 10)

# R
tc_r = rs.calc_monthly_sum(data['tc_r'], startdate, enddate, area)
ecmwf_r = rs.calc_monthly_sum(data['ecmwf_r'], startdate, enddate, area)
ecmwf_r = rs.calc_monthly_sum(data['ecmwf_r'], startdate, enddate, area)

fldas_ssr = rs.calc_monthly_sum(data['fldas_ssr'], startdate, enddate, area)
fldas_bfr = rs.calc_monthly_sum(data['fldas_bfr'], startdate, enddate, area)

gldas_ssr = rs.calc_monthly_sum(data['gldas_ssr'], startdate, enddate, area)
gldas_bfr = rs.calc_monthly_sum(data['gldas_bfr'], startdate, enddate, area)

# Sum the base flow and surface runoff 
gldas_r = pd.DataFrame(pd.concat([gldas_bfr, gldas_ssr], axis = 1).sum(axis =1))
gldas_r.columns = ['gldas_r']
fldas_r = pd.DataFrame(pd.concat([fldas_bfr, fldas_ssr], axis = 1).sum(axis =1))
fldas_r.columns = ['fldas_r']


# SWe
print("-------" * 10)
print("Processing SWE Data")
print("-------" * 10)

gldas_swe = rs.calc_monthly_mean(data['gldas_swe'], startdate, enddate, area)
fldas_swe = rs.calc_monthly_mean(data['fldas_swe'],startdate, enddate, area)
dmet_swe = rs.calc_monthly_mean(data['dmet_swe'],startdate, enddate, area)
tc_swe = rs.calc_monthly_mean(data['tc_swe'],startdate, enddate, area)

# SM
print("-------" * 10)
print("Processing Soil Moisture Data")
print("-------" * 10)

# Smos
smos_ssm = rs.calc_monthly_mean(data['smos_ssm'], "2010-01-01", "2019-12-31", area)
smos_susm = rs.calc_monthly_mean(data['smos_susm'],"2010-01-01", "2019-12-31", area)
smos_smp = rs.calc_monthly_mean(data['smos_smp'],"2010-01-01", "2019-12-31", area)
smos_sm = pd.concat([smos_ssm, smos_susm], axis = 1).sum(axis =1)

# SMAP
smap_ssm = rs.calc_monthly_mean(data['smap_ssm'], '2015-04-01', enddate, area)
smap_susm = rs.calc_monthly_mean(data['smap_susm'],'2015-04-01', enddate, area)
smap_smp = rs.calc_monthly_mean(data['smap_smp'],'2015-04-01', enddate, area)

# TC
tc_sm = rs.calc_monthly_mean(data['tc_sm'], startdate, enddate, area)

# GLDAS
gldas_rzsm = rs.calc_monthly_mean(data['gldas_rzsm'], startdate, enddate, area)
gldas_gsm1 = rs.calc_monthly_mean(data['gsm1'], startdate, enddate, area)
gldas_gsm2 = rs.calc_monthly_mean(data['gsm2'], startdate, enddate, area)
gldas_gsm3 = rs.calc_monthly_mean(data['gsm3'], startdate, enddate, area)
gldas_gsm4 = rs.calc_monthly_mean(data['gsm4'], startdate, enddate, area)
gldas_sm = pd.concat([gldas_gsm1,gldas_gsm2,gldas_gsm3,gldas_gsm4], axis = 1).sum(axis =1)

# FLDAS 
fldas_fsm1 = rs.calc_monthly_mean(data['fsm1'], startdate, enddate, area)
fldas_fsm2 = rs.calc_monthly_mean(data['fsm2'], startdate, enddate, area)
fldas_fsm3 = rs.calc_monthly_mean(data['fsm3'], startdate, enddate, area)
fldas_fsm4 = rs.calc_monthly_mean(data['fsm4'], startdate, enddate, area)


# Combine SM
gldas_sm = pd.DataFrame(pd.concat([gldas_gsm1,gldas_gsm2,gldas_gsm3,gldas_gsm4], axis = 1).sum(axis =1))
gldas_sm.columns = ['gldas_sm']
fldas_sm = pd.concat([fldas_fsm1,fldas_fsm2,fldas_fsm3,fldas_fsm4], axis = 1).sum(axis =1)
fldas_sm.columns = ['fldas_sm']
smap_sm = pd.DataFrame(pd.concat([smap_ssm, smap_susm], axis = 1).sum(axis =1))
smap_sm.columns = ['smap_sm']
smos_sm = pd.DataFrame(pd.concat([smos_ssm, smos_susm], axis = 1).sum(axis =1))
smos_sm.columns = ['smos_sm']

print("-------" * 10)
print("Processing Precip Data")
print("-------" * 10)

# Precip
gpm = rs.calc_monthly_sum(data['gpm'], startdate, enddate, area)
prism = rs.calc_monthly_sum(data['prism'], startdate, enddate, area)
dmet = rs.calc_monthly_sum(data['dmet'], startdate, enddate, area)
chirps = rs.calc_monthly_sum(data['chirps'], startdate, enddate, area)
psn = rs.calc_monthly_sum(data['persiann'], startdate, enddate, area)

print("-------" * 10)
print("Processing AET Data")
print("-------" * 10)

# Aet
modis_aet = rs.calc_monthly_sum(data['modis_aet'], startdate, enddate, area)
gldas_aet = rs.calc_monthly_sum(data['gldas_aet'], startdate, enddate, area)
tc_aet = rs.calc_monthly_sum(data['tc_aet'], startdate, enddate, area)
fldas_aet = rs.calc_monthly_sum(data['fldas_aet'], startdate, enddate, area)

print("-------" * 10)
print("Processing PET Data")
print("-------" * 10)

# PET
gldas_pet = rs.calc_monthly_sum(data['gldas_pet'], startdate, enddate, area)
modis_pet = rs.calc_monthly_sum(data['modis_pet'], startdate, enddate, area)
nldas_pet = rs.calc_monthly_sum(data['nldas_pet'], startdate, enddate, area)
tc_pet = rs.calc_monthly_sum(data['tc_pet'], startdate, enddate, area)
gmet_eto = rs.calc_monthly_sum(data['gmet_eto'], startdate, enddate, area)


# Process the dataframes 
pdfs = {"p_prism":prism, "p_gpm":gpm,"p_dmet":dmet, "p_chirps": chirps, "p_psn":psn}
aetdfs = {"aet_modis":modis_aet, "aet_gldas":gldas_aet, "aet_tc":tc_aet, "aet_fldas":fldas_aet }
petdfs = {"pet_modis":modis_pet, "pet_gldas":gldas_pet, "pet_tc":tc_pet, "pet_nldas":nldas_pet, 'pet_gmet':gmet_eto }
smdfs = {"sm_smos": smos_sm, "sm_smap": smap_sm, "sm_tc": tc_sm, "sm_gldas": gldas_sm}
rdfs = {"r_tc": tc_r, "r_gldas": gldas_r, "r_fldas": fldas_r, "r_ecmwf": ecmwf_r}
swedfs = {'swe_gldas': gldas_swe, 'swe_fldas': fldas_swe, 'swe_dmet':dmet_swe, "swe_tc":tc_swe}

master_df = []

for i in [pdfs, aetdfs, petdfs, smdfs, rdfs, swedfs]:
    for k,v in i.items():
        # print(k,v.columns)
        newdf = v
        newdf.columns = [k]
        master_df.append(newdf)

finout = pd.concat(master_df, axis = 1).astype(float)

finout.to_csv('../data/RS_analysis_dat.csv')
