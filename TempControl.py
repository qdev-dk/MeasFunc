# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:04:34 2022

@author: Triton 9 acq
"""
#%% Initialise and read channels
from time import sleep
import time
from qcodes.instrument_drivers.Lakeshore.Model_372 import Model_372
ls = Model_372('lakeshore_372', 'TCPIP::192.168.15.105::7777::SOCKET')
for ch in ls.channels:
    print(f'Temperature of {ch.short_name} ({"on" if ch.enabled() else "off"}): {ch.temperature()} {ch.units()}')
 
h = ls.sample_heater

h.wait_cycle_time(0.5)
h.wait_tolerance(0.1)
h.wait_equilibration_time(1.5)
#%% desacivate all channels except MC and magnet

for ch in ls.channels:
    ch.enabled(False)
ls.ch08.enabled(True)
ls.ch01.enabled(False)
ls.ch13.enabled(True)

for ch in ls.channels:
    print(f'Temperature of {ch.short_name} ({"on" if ch.enabled() else "off"}): {ch.temperature()} {ch.units()}')
    
#%% Define PID and input channels
h.P(10)
h.I(20)
h.D(0)
h.output_range('1mA')
h.input_channel(8)

print('P', h.P())
print('I', h.I())
print('D', h.D())
print('Output Range', h.output_range())
print('Input Channel', h.input_channel())

#%% prepare for T control
h.setpoint(0.0)
h.mode('closed_loop')
#%% define gotoT
def GotoT(T):
    if T<0.030:
        h.output_range('1mA')
    elif T<0.1:
        h.output_range('3.16mA')
    elif T<0.3:
        h.output_range('10mA')
    elif T<0.8:
        h.output_range('31.6mA')
    else:
        print('T outside range')
        T=0
            
    
    h.setpoint(T)
    sleep(1)
    print(time.ctime())
    print('waiting for stability')
    h.wait_until_set_point_reached()
    
    print('stable')
    print(f'T = {ls.ch08.temperature()}')
    print(time.ctime())
    
#%% Magnet control
def MagCheck():
    Flag=0
    MT=ls.ch13.temperature()
    if MT>4.6:
        Flag=1
    return Flag
#%% Measurement:
#tls=np.arange(0.020, 0.500, 0.010)
tls=np.arange(0.020, 0.050, 0.010)
for i, temp in enumerate(tls):
    MOK=MagCheck()
    if MOK==0:

        GotoT(temp)
        print('Measurement')
        print(f'T = {ls.ch08.temperature()}')
        print(Lockin1.ResX(), Lockin2.ResX(), Lockin3.ResX(), Lockin4.ResX(), Lockin5.ResX())
        #magnet.bpar(0.1)
        #print('Magnetizing...')
        #print(time.ctime())
        #sleep(1)
        #magnet.bpar(0)
        #sleep(1)
        #do1d(magnet.bpar, 0, -0.05, 51, 0.11, Lockin1.ResX, Lockin2.ResX, Lockin3.ResX, Lockin4.ResX, Lockin5.ResX)
        #sleep(1)
        #print('Complete')
        #print(time.ctime())
    else:
        print('magnet too warm')
        print(f'T = {ls.ch13.temperature()}')
        pause(30)
        magnet.bpar(0)
        
        break
h.setpoint(0.0)

#%%
tls=np.arange(0.020, 0.510, 0.010)
maglist = np.linspace(0, -0.05, 51)
from qcodes.dataset import Measurement, experiments

meas = Measurement(exp=myexp,
                   station=scfg,
                   name='test_temp_map_example')

meas.register_parameter(ls.ch08.temperature)
meas.register_parameter(magnet.bpar)
meas.register_parameter(Lockin1.ResX,setpoints=(ls.ch08.temperature,magnet.bpar))
meas.register_parameter(Lockin2.ResX,setpoints=(ls.ch08.temperature,magnet.bpar))
meas.register_parameter(Lockin3.ResX,setpoints=(ls.ch08.temperature,magnet.bpar))
meas.register_parameter(Lockin4.ResX,setpoints=(ls.ch08.temperature,magnet.bpar))
meas.register_parameter(Lockin5.ResX,setpoints=(ls.ch08.temperature,magnet.bpar))

with meas.run() as datasaver:
    for temp in tls:
        print('Magnetizing...')
        print(time.ctime())
        magnet.bpar(0.1)
        sleep(1)
        magnet.bpar(0.0)
        sleep(1)
        GotoT(temp)
        MOK=MagCheck()
        if MOK==0: 
            for mag in maglist:
                magnet.bpar(mag)
                datasaver.add_result(
                    (ls.ch08.temperature,ls.ch08.temperature()),
                    (magnet.bpar,magnet.bpar()),
                    (Lockin1.ResX,Lockin1.ResX()),
                    (Lockin2.ResX,Lockin2.ResX()),
                    (Lockin3.ResX,Lockin3.ResX()),
                    (Lockin4.ResX,Lockin4.ResX()),
                    (Lockin5.ResX,Lockin5.ResX()),
                    )
        else:
            print('magnet too warm')
            print(f'T = {ls.ch13.temperature()}')
            pause(30)
            magnet.bpar(0) 
            break
    thisdata = datasaver.dataset
            
h.setpoint(0.0)            
