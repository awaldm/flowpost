#/usr/bin/python


###############################################################################
# USER SETUP

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/DDES_dt100_md_cK/sol/surface/plt/'
case_name = 'DDES_v38_dt100_md_cK'
zonelist = [12] # 12
planelist = ['eta0283']
start_i = 28500 # 27120
end_i = 31800

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_md_CFL2_eigval015_pswitch1_symm_tau2017/sol/surface/plt/'
case_name = 'DDES_v38h_dt100_md_CFL2_eigval015_pswitch1_symm_tau2017'
zonelist = [11]
planelist = ['eta0283']
start_i = 7400
end_i = 9990

plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_md_CFL2_eigval02_pswitch1_tau2014/sol/surface/plt/'
case_name = 'DDES_v38h_dt100_md_CFL2_eigval02_pswitch1_symm_tau2014'
zonelist = [10]
planelist = ['eta0201']
start_i = 1000
end_i = 5600

plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_md_CFL2_eigval02_pswitch1_chim_tau2014/sol/surface/plt/'
case_name = 'DDES_v38h_dt100_md_CFL2_eigval02_pswitch1_chim_tau2014'
zonelist = [11]
planelist = ['eta0283']
start_i = 1200
end_i = 5600

plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_md_CFL2_eigval02_pswitch4_tau2014/sol/surface/plt/'
case_name = 'DDES_v38h_dt100_md_CFL2_eigval02_pswitch4_symm_tau2014'
zonelist = [11]
planelist = ['eta0283']
start_i = 2200
end_i = 7600



#plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38t/DDES_dt100_md_chim_tau2017/sol/surface/plt/'
#case_name = 'DDES_v38t_dt100_md_chim_tau2017'
#zonelist = [11]
#planelist = ['eta0283']
#start_i = 2800
#end_i = 6500

#plt_path = '/home/andreas/hazelhen_WS1/v38_tet/DDES/DDES_dt100_md_symm_tau2014/sol/surface/plt/'
#case_name = 'DDES_v38t_dt100_md_symm_tau2014'
#zonelist = [10]
#planelist = ['eta0201']
#start_i = 1100
#end_i = 2800


'''
plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/DDES_dt180_md/sol/surface/plt/'
case_name = 'DDES_v38_dt100md'
zone_no = 11
start_i = 26000
end_i = 27510
#zone_no = 10 # 11
#start_i = 5000
#end_i = 7000
#plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_v38h05_dt50/sol/surface/plt/'
#case_name = 'DDES_v38h_dt50md_symm_tau2014'
#zone_no = 21 # 10 # 11
#start_i = 2050
#end_i = 5600
#zone_no = 10 # 11
#start_i = 1200
#end_i = 3200
plt_path = '/home/andreas/laki_ehrle/AW_tests/AZDES-SSG/AZDES-SSG_dt2e5_turbRoe2nd_SGDH_Lc00161/sol/surface/'
case_name = 'OAT15A_AZDES-SSG'
zone_no = 0
zonelist = [0]
start_i = 15490
end_i = 17450
plt_path = '/home/andreas/laki_ehrle/AW_tests/URANS-SSG/URANS-SSG_dt2e5_2016.2_turbRoe2nd_SGDH/sol/surface/'
case_name = 'OAT15A_URANS-SSG'
zone_no = 0
zonelist = [0]
start_i = 9640
end_i = 12880


plt_path = '/home/andreas/hazelhen_WS1/v38_tet/DDES/DDES_dt100_md_symm_tau2014/sol/surface/plt/'
case_name = 'DDES_v38t_dt100md_symm_tau2014'
zone_no = 10 # 11
start_i = 1200
end_i = 3100
#start_i = 3200
#end_i = 5600
zonelist = [10,11,12,13]


plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/URANS-SSG_turbRoe2nd/sol/surface/plt/'
case_name = 'URANS-SSG_turbRoe2nd'
zonelist = [12,13,14,15]

planelist= ['eta0201', 'eta0283', 'eta0397', 'eta0603']
zonelist = [13]
planelist = ['eta0283']

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/URANS-SAO/sol/surface/plt/surfaces/'
case_name = 'URANS-SAO'
start_i = 10000
end_i = 17260
zonelist = [4]
planelist = ['eta0283']
plt_path = '/home/andreas/hazelhen_WS1/M085/a45_URANS_v51/SAQCR/dt150/sol/surface/plt/'
case_name = 'CRM_M085_v51_a45_SAQCR_dt150'
zonelist = [3]
start_i = 1500
end_i = 2100
zone_name ='wing_ss'

plt_path = '/home/andreas/hazelhen_WS1/M085/a50_URANS_v51/SA-QCR/dt150/sol/surface/plt/'
case_name = 'CRM_M085_v51_a50_SAQCR_dt150'
#zonelist = [3]
zonelist = [2,3,4,5]
start_i = 3000
end_i = 4200
zone_name ='wing_ss'

plt_path = '/home/andreas/hazelhen_WS1/M085/a50_URANS_v51/SA-QCR/dt150_AZDES/sol/surface/plt/'
case_name = 'CRM_M085_v51_a50_SAQCR_AZDES_dt150'
zonelist = [3]
zonelist = [2,3,4,5]
start_i = 2700
end_i = 3300
zone_name ='wing_ss'
'''

plt_path = '/home/andreas/NAS_CRM2/CRM/M025/v38_h2/AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
case_name = 'CRM_v38h_AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
#zonelist = [3]
zonelist = [1,2,3,4]
zonelist = [0,1,2,3]
start_i = 2200
end_i = 4800
zone_name ='wing_ss'
datasetfile = 'v38h_wing.plt'

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_ldDLR_CFL2_eigval02_pswitch4_tau2014/sol/surface/plt/'
case_name = 'CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval02_pswitch1_symm_tau2014'
zonelist = [1,2,3,4]
zonelist = [0,1,2,3]
start_i = 3330
end_i = 6800
zone_name ='wing_ss'
datasetfile = 'v38h_wing.plt'

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/DDES_v52/dt200_ldDLR/sol/surface/plt/'
case_name = 'CRM_v52_dt200_ldDLR'
zonelist = [0,1,2,3]
start_i = 1000
end_i = 4400
zone_name ='wing_ss'
datasetfile = 'v52_wing.plt'

'''

plt_path = '/home/andreas/hazelhen_WS1/v52/DDES-SAO/dt200_LD2/sol/surface/plt/'
case_name = 'CRM_v52_dt200_LD2'
zonelist = [0,1,2,3]
start_i = 3000
end_i = 4600
zone_name ='wing_ss'


plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/IDDES_dt200_ldDLR/sol/surface/plt/'
case_name = 'CRM_v38_IDDES_dt200_ldDLR'
zonelist = [3,4,5,6]
start_i = 26000
end_i = 28400
zone_name ='wing_ss'


plt_path = '/home/andreas/hazelhen_WS1/M085/a50_URANS_v51/SSG-w/dt150/sol/surface/plt/'
case_name = 'CRM_M085_v51_a50_SSG_dt150'
zonelist = [3]
start_i = 500
end_i = 1300
zone_name ='wing_ss'

plt_path = '/home/andreas/hazelhen_WS1/M085/a45_URANS_pb_WBT0ssd/SAQCR/dt150/sol/surface/plt/'
case_name = 'CRM_M085_pbWBT0ssd_a45_SAQCR_dt150'
zonelist = [8]
start_i = 300
end_i = 1200
zone_name ='wing_ss'
'''

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_ldDLR_CFL2_eigval02_pswitch4_tau2014/sol/surface/plt/'
case_name = 'CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval02_pswitch1_symm_tau2014'
zonelist = [1,2,3,4]
zonelist = [0,1,2,3]
start_i = 3330
end_i = 6800
zone_name ='wing_ss'
datasetfile = 'v38h_wing.plt'

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
case_name = 'CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
zonelist = [1,2,3,4]
zonelist = [0,1,2,3]
start_i = 2000
end_i = 7200
zone_name ='wing_ss'
datasetfile = 'v38h_wing.plt'



plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/URANS_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
case_name = 'CRM_v38h_URANS_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
zonelist = [9,10,11,12,13,14]
zonelist = [21]
start_i = 1500
end_i = 4000
planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
planelist= ['x2']
plane = planelist[0]
datasetfile = 'v38h_etaplanes.plt'


plt_path = '/home/andreas/NAS_CRM/CRM/M085/HSS/v51/a50/dt150_URANS_md128/sol/surface/plt/'
case_name = 'CRM_M085_v51_a50_SAQCR_URANS_dt150_md128'
zonelist = [2,3,4,5]
start_i = 3000
end_i = 7200
zone_name ='wing_ss'
datasetfile = 'v51_wing.plt'

plt_path = '/home/andreas/NAS_CRM/CRM/M085/HSS/v51/a50/dt150_URANS_ldDLR/sol/surface/plt/'
case_name = 'CRM_M085_v51_a50_SAQCR_URANS_dt150_ldDLR'
zonelist = [2,3,4,5]
start_i = 3000
end_i = 5100
zone_name ='wing_ss'
datasetfile = 'v51_wing.plt'

plt_path = '/home/andreas/NAS_CRM/CRM/M085/HSS/v51/a50/dt150_AZDES_Lc007/sol/surface/plt/'
case_name = 'CRM_M085_v51_a50_SAQCR_AZDES_dt150_Lc007'
#zonelist = [3]
zonelist = [2,3,4,5]
start_i = 4900
end_i = 9900
zone_name ='wing_ss'
datasetfile = 'v51_wing.plt'
'''

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/DDES_dt100_ldDLR/sol/surface/plt/'
case_name = 'CRM_v38_DDES_dt100_ldDLR'
zonelist = [3,4,5,6]
start_i = 25750
end_i = 29500 # 29500
datasetfile = 'v38_wing.plt'
zone_name ='wing_ss'


plt_path = '/home/andreas/hazelhen_M085/pb_WBT0ssd/a50_URANS/dt150_md128/sol/surface/plt/'
case_name = 'CRM_M085_pbWBT0ssd_a50_SAQCR_URANS_dt150_md128'
zonelist = [7,8,9,10]
start_i = 4500
end_i = 7200
zone_name ='wing_ss'
datasetfile = 'pb_wing.plt'


plt_path = '/home/andreas/hazelhen_M085/pb_WBT0ssd/a50_URANS/dt150_ldDLR/sol/surface/plt/'
case_name = 'CRM_M085_pbWBT0ssd_a50_SAQCR_URANS_dt150_ldDLR'
zonelist = [7,8,9,10]
start_i = 3000
end_i = 6700
zone_name ='wing_ss'
datasetfile = 'pb_wing.plt'
'''

plt_path = '/home/andreas/NAS_CRM2/CRM/M025/v38_h2/AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
case_name = 'CRM_v38h_AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
zonelist = [9,10,11,12,13,14]
zonelist = [14]
start_i = 5000
end_i = 12000
planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
planelist= ['eta0603']
plane = planelist[0]
#datasetfile = 'v38h_etaplanes.plt'
#planelist=['wake_4']
#zonelist=[18]

plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2/sol/surface/plt/'
plt_path = '/home/andreas/NAS_CRM2/CRM/M025/v38_h2/DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2/sol/surface/plt/'
case_name = 'CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2'
zonelist = [9,10,11,12,13,14]
zonelist = [11]
start_i = 5000
end_i = 16400
#planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
planelist= ['eta0283']
plane = planelist[0]
datasetfile = 'v38h_etaplanes.plt'
#planelist=['wake_4']
#zonelist=[18]
#zonelist = [0,1,2,3]
#datasetfile = 'v38h_wing.plt'

'''
plt_path = '/home/andreas/NAS_CRM2/CRM/M025/v38_h2/AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024/sol/surface/plt/'
case_name = 'CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024'
zonelist = [9,10,11,12,13,14]
zonelist = [11]
start_i = 5000
end_i = 15600
planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
planelist= ['eta0283']
datasetfile = 'v38h_etaplanes.plt'
plane = planelist[0]
#planelist=['x2']
#zonelist=[21]
#planelist=['wake_4']
#zonelist=[18]
#zonelist = [0,1,2,3]
#datasetfile = 'v38h_wing.plt'


#plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/DDES_v52/dt200_ldDLR/sol/surface/plt/'
#case_name = 'CRM_v52_dt200_ldDLR'
#zonelist = [9,10,11,12,13,14,15,16,17]
#zonelist = [9]
#start_i = 1000
#end_i = 5300
#planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603', 'eta0727', 'eta0848', 'eta0950']
#planelist= ['eta0131']
#datasetfile = 'v52_etaplanes.plt'

plt_path = '/home/andreas/NAS_CRM2/CRM/M025/v38_hex/AoA18_Re250_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
case_name = 'AoA18_Re250_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
zonelist = [9,10,11,12,13,14]
zonelist = [11]
start_i = 2000
end_i = 7600
planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
planelist= ['eta0283']
datasetfile = 'v38h_etaplanes.plt'
plane = planelist[0]

plt_path = '/home/andreas/NAS_CRM2/CRM/M025/v38_hex/AoA18_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
case_name = 'AoA18_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
zonelist = [9,10,11,12,13,14]
zonelist = [11]
start_i = 4400 # davor kaputt
end_i = 9000
planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
planelist= ['eta0283']
datasetfile = 'v38h_etaplanes.plt'
plane = planelist[0]

plt_path = '/home/andreas/NAS_CRM2/CRM/M025/v52/DDES_dt100_ldDLR/sol/surface/plt/'
case_name = 'CRM_v52_dt100_ldDLR'
#zonelist = [9,10,11,12,13,14,15,16,17]
zonelist = [11]
start_i = 2000
end_i = 5200
#planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603', 'eta0727', 'eta0848', 'eta0950']
planelist= ['eta0283']
datasetfile = 'v52_etaplanes.plt'
'''
plt_path = '/home/andreas/laki_nieth/AOA19/DDES/SAO_dt1e5_k1024_turbAoF/sol/surface/plt/'
case_name = 'NACA0012_AoA19_DDES_SAO_dt1e5_k1024_turbAoF'
start_i = 10000
end_i = 24000
zonelist = [1]
planelist = ['periodic']
datasetfile = None

plt_path = '/home/andreas/laki_nieth/AOA19/URANS/SAO_dt1e5_k1024_turbAoF/sol/surface/plt/'
case_name = 'NACA0012_AoA19_URANS_SAO_dt1e5_k1024_turbAoF'
start_i = 16000
end_i = 20000
zonelist = [1]
planelist = ['periodic']
datasetfile = None

plt_path = '/home/andreas/laki_nieth/AOA19/URANS/SAO_dt1e5_MDk0128_convSSK_turbAoF/sol/surface/plt/'
case_name = 'NACA0012_AoA19_URANS_SAO_dt1e5_MDk0128_convSSK_turbAoF'
start_i = 17000
end_i = 20000
zonelist = [1]
planelist = ['periodic']
datasetfile = None

