#------------------------------------------------------------------Initialization------------------------------------------------------------------#
dimension 	3
boundary	p p p
units		metal
atom_style	atomic
neighbor	2.0 bin
neigh_modify	every 20 delay 0 check no  
shell "mkdir -p dump"

#-----------------------------------------------------------------Box of simulation----------------------------------------------------------------#

read_data	/path/to//AnisoScrew.lmp

#---------------------------------------------------------------Interatomic potential--------------------------------------------------------------#

pair_style    adp
pair_coeff    * * /path/to/UMo.adp.txt Mo  

timestep	0.0005
thermo		100   

#-----------------------------------------------------------------Property computes----------------------------------------------------------------#

compute		energy all pe/atom               
compute		kenergy all ke/atom                     
compute		stress all stress/atom NULL
compute 	disp all displace/atom  
#compute 	MSD all msd com yes
region		dyn_cyl cylinder x 84.1 57.1 30.1 INF INF
group		dyn_cyl region dyn_cyl
compute 	temp_dyn_cyl dyn_cyl temp/region dyn_cyl

#--------------------------------------------------------------------Relaxation-------------------------------------------------------------------#

thermo_style	custom step temp c_temp_dyn_cyl etotal pe press pxx pyy pzz vol pxy pxz pyz #c_MSD[1] c_MSD[2] c_MSD[3] c_MSD[4]

fix 		Relax all box/relax aniso 0.0
min_style	cg               
minimize 	1e-25 1e-25 5000 10000
unfix		Relax

#--------------------------------------------------------------Thermal equilibration--------------------------------------------------------------#

region		cyl cylinder x 84.5 57.1 40.1 INF INF
group		cyl region cyl

group		freez subtract cyl dyn_cyl
fix		f_b freez setforce 0.0 0.0 0.0
velocity	freez create 0 84721

fix 		nvt dyn_cyl nvt temp 1700.0 1700.0 0.01
run		5000
unfix 		nvt

region		charc_cyl cylinder x 85.85 55.81 15.1 20.1 50.1
group		charc_cyl region charc_cyl

#---------------------------------------------------------------Output visualization-------------------------------------------------------------#

dump		1 charc_cyl cfg 100 /path/to//dump/Mo.*.cfg  mass type xs ys zs  id c_energy c_stress[1] c_stress[4] c_stress[5] c_stress[6] c_disp[1] c_disp[2] c_disp[3] c_disp[4] 
dump_modify	1 element Mo

#-----------------------------------------------------------------traj for VDOS------------------------------------------------------------------#
fix		1 all nve
fix		L dyn_cyl langevin 1700 1700 0.01 699483

dump        	2 charc_cyl xyz 1 /path/to//dump/traj-Screw-1700.xyz	
dump_modify 	2 element Mo

run		2000

#-----------------------------------------------------------------Diffraction data----------------------------------------------------------------#

compute XRD charc_cyl  xrd 1.541838 Mo 2Theta 10 100 c 0.02 0.02 0.02 LP 1 manual echo        #default 0.05
fix XRD charc_cyl ave/histo/weight 1 1 1 10 100 1000 c_XRD[1] c_XRD[2] mode vector file /path/to//dump/Deg2Theta_Screw-1700.xrd

compute         SAED_Screw charc_cyl saed 0.0251  Mo Kmax 1.72  &
                Zone 0 0 0 c 0.02 0.02 0.02 &
                dR_Ewald 0.01 echo manual
            
fix         SAED_Screw charc_cyl saed/vtk 1 1 1 c_SAED_Screw file /path/to//dump/Screw-1700.saed 

run             0

unfix           XRD
uncompute       XRD
unfix           SAED_Screw
uncompute       SAED_Screw


write_restart 	/path/to//dump/screw_1700K.restart

