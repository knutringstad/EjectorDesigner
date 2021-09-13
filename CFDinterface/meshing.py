def meshingEjector (MeshLocation, ScriptRootLocation,Name,run_ICEM, write_mesh, mesh_convergence,refine,mesh_smoothing,delta, df):

    import numpy as np
    import pandas
    import math
    import matplotlib.pyplot as plt
    import subprocess


    # READ INPUTS FROM DATAFRAME

    LintletConst = df['LintletConst']
    Lsuction = df['Lsuction']
    DmotiveIn = df['DmotiveIn']
    DmotiveOut = df['DmotiveOut']
    Dthroat = df['Dthroat']
    alphaMotiveDiff = df['alphaMotiveDiff']
    alphaMotiveConv = df['alphaMotiveConv']
    Lmch = df['Lmch']
    alphaSuction = df['alphaSuction']
    Dmix = df['Dmix']
    Lmix = df['Lmix']
    alphadiff = df['alphadiff']
    DdiffOut = df['DdiffOut']
    Loutlet = df['Loutlet']
    Dsuc = df['Dsuc']
    ThicknessNozzle = df['ThicknessNozzle']
    ThicknessNozzleOuter =df['ThicknessNozzleOuter']


    Ldiff = (DdiffOut/2-Dmix/2)/(math.tan(math.radians(alphadiff/2)))
    LinletDiff = (DmotiveOut/2-Dthroat/2)/(math.tan(math.radians(alphaMotiveDiff/2)))
    LinletConv = (-Dthroat/2+DmotiveIn/2)/(math.tan(math.radians(alphaMotiveConv/2)))

    Lmotive =LinletDiff + LinletConv
    alphaMotiveOuter = alphaSuction
    


    if Dsuc > 0 :
        Ldsmax = - Lmch + (Dsuc- Dmix)/(2*math.tan(math.radians(alphaSuction/2)))
        Lsuction = min(Ldsmax,Lsuction)
        Lsuction = min(LinletDiff+LinletConv*0.8,Lsuction)
        

    Lsuction = min(LinletDiff+LinletConv*0.8,Lsuction)

    Dsuc = 2*(Lsuction + Lmch)*(math.tan(math.radians(alphaSuction/2))) + Dmix


    # MAKE ITERABLE LISTS FOR POINTS

    y = np.ndarray(13,float)
    x = np.ndarray(13,float)

    x [0] = -Lmotive-LintletConst
    x [1] = -Lmotive
    x [2] = -LinletDiff
    x [3] = 0
    x [4] = 0
    x [5] = -Lsuction
    x [6] = -Lsuction
    x [7] = Lmch 
    x [8] = Lmch + Lmix
    x [9] = Lmch + Lmix + Ldiff
    x [10] = Lmch + Lmix + Loutlet+ Ldiff
    x [11] = Lmch + Lmix + Loutlet+ Ldiff
    x [12] = -Lmotive-LintletConst

    
    y[0] = DmotiveIn/2
    y[1] = DmotiveIn/2
    y[2] = Dthroat/2
    y[3] = DmotiveOut/2
    y[4] = DmotiveOut/2+ThicknessNozzle
    y[5] = DmotiveOut/2+ThicknessNozzle+ Lsuction * math.tan(math.radians(alphaMotiveOuter/2))
    y[6] = Dsuc/2
    y[7] = Dmix/2
    y[8] = Dmix/2
    y[9] = DdiffOut/2
    y[10] = DdiffOut/2
    y[11] = 0
    y[12] = 0
  



    # ICEM .rpl SCRIPT DEFINITION

    ScriptName = '%s/AutoEjector_ICEMscript_%s.rpl' % (ScriptRootLocation,Name)
    fid=open(ScriptName,'w')

    # POINTS
    fid.write('ic_set_global geo_cad 0 toptol_userset\n') 
    fid.write('ic_set_global geo_cad 0.0 toler\n') 
    fid.write('ic_geo_new_family GEOM\n') 
    fid.write('ic_boco_set_part_color GEOM\n') 
    fid.write('ic_empty_tetin\n') 

    for i in range(13):
        fid.write("ic_point {{}} GEOM pnt.%d {%f %f 0}\n" % (i, x[i],y[i] ))

    for i in range(12):
        fid.write( 'ic_undo_group_begin\n') 
        fid.write('ic_delete_geometry curve names crv.%d 0 \n' % i) 
        fid.write('ic_curve point GEOM crv.%d {pnt.%d pnt.%d} \n' % (i,i,i+1) )
        fid.write( 'ic_undo_group_end \n') 

    fid.write( 'ic_undo_group_begin\n') 
    fid.write('ic_delete_geometry curve names crv.%d 0 \n' % 12 )
    fid.write('ic_curve point GEOM crv.%d {pnt.%d pnt.%d} \n' % (12,12,0) )
    fid.write( 'ic_undo_group_end \n') 


    # BLOCKING

    fid.write('ic_geo_new_family FLUID\n') 
    fid.write('ic_boco_set_part_color FLUID\n') 
    fid.write('ic_hex_unload_blocking \n') 
    fid.write('ic_hex_initialize_mesh 2d new_numbering new_blocking FLUID\n') 
    fid.write('ic_hex_unblank_blocks \n') 
    fid.write('ic_hex_multi_grid_level 0\n') 
    fid.write('ic_hex_projection_limit 0\n') 
    fid.write('ic_hex_default_bunching_law default 2.0\n') 
    fid.write('ic_hex_floating_grid off\n') 
    fid.write('ic_hex_transfinite_degree 1\n') 
    fid.write('ic_hex_unstruct_face_type one_tri\n') 
    fid.write('ic_hex_set_unstruct_face_method uniform_quad\n') 
    fid.write('ic_hex_set_n_tetra_smoothing_steps 20\n') 
    fid.write('ic_hex_error_messages off_minor\n') 
    fid.write('ic_undo_group_end \n') 

    # SPLITTING

    fid.write('ic_hex_split_grid 13 21 pnt.3 m GEOM FLUID VORFN\n')  #motive cut x
    fid.write('ic_hex_split_grid 34 21 pnt.7 m GEOM FLUID VORFN\n')  #mix cut x
    fid.write('ic_hex_split_grid 38 21 pnt.8 m GEOM FLUID VORFN\n')  #mix-diff cut x
    fid.write('ic_hex_split_grid 42 21 pnt.9 m GEOM FLUID VORFN\n')  #outlet cut x

    fid.write('ic_undo_group_begin \n') 
    fid.write('ic_hex_mark_blocks unmark\n') 
    fid.write('ic_hex_mark_blocks superblock 4\n') 
    fid.write('ic_hex_undo_major_start split_grid\n') 
    fid.write('ic_hex_split_grid 11 13 pnt.3 m GEOM FLUID VORFN marked\n')  #low motive cut y
    fid.write('ic_hex_undo_major_end split_grid \n') 
    fid.write('ic_undo_group_end \n') 

    fid.write('ic_undo_group_begin \n') 
    fid.write('ic_hex_mark_blocks unmark\n') 
    fid.write('ic_hex_mark_blocks superblock 21\n') 
    fid.write('ic_hex_undo_major_start split_grid\n') 
    fid.write('ic_hex_split_grid 49 13 pnt.4 m GEOM FLUID VORFN marked\n')  #high motive cut y
    fid.write('ic_hex_undo_major_end split_grid \n') 
    fid.write('ic_undo_group_end \n') 

    fid.write('ic_undo_group_begin \n') 
    fid.write('ic_hex_mark_blocks unmark\n') 
    fid.write('ic_hex_mark_blocks superblock 22 \n') 
    fid.write('ic_hex_mark_blocks superblock 4 \n') 
    fid.write('ic_hex_undo_major_start split_grid \n') 
    fid.write('ic_hex_split_grid 13 34 pnt.6 m GEOM FLUID VORFN marked \n') 
    fid.write('ic_hex_undo_major_end split_grid \n') 
    fid.write('ic_undo_group_end \n') 

    fid.write('ic_undo_group_begin \n') 
    fid.write('ic_hex_mark_blocks unmark\n') 
    fid.write('ic_hex_mark_blocks superblock 4\n') 
    fid.write('ic_hex_undo_major_start split_grid\n') 
    fid.write('ic_hex_split_grid 13 68 pnt.1 m GEOM FLUID VORFN marked\n')  #motive cut 2 x
    fid.write('ic_hex_undo_major_end split_grid \n') 
    fid.write('ic_undo_group_end \n') 

    fid.write('ic_hex_mark_blocks unmark\n') 
    fid.write('ic_hex_mark_blocks superblock 4\n') 
    fid.write('ic_hex_mark_blocks superblock 23\n') 
    fid.write('ic_hex_mark_blocks superblock 25\n') 
    fid.write('ic_hex_split_grid 33 50 0.95 m GEOM FLUID VORFN marked\n')  #motive BL cut
    fid.write('ic_hex_mark_blocks unmark\n') 

    fid.write('ic_hex_mark_blocks superblock 22\n') 
    fid.write('ic_hex_mark_blocks superblock 27\n') 
    fid.write('ic_hex_mark_blocks superblock 24\n') 
    fid.write('ic_hex_mark_blocks superblock 25\n') 
    fid.write('ic_hex_mark_blocks superblock 10\n') 
    fid.write('ic_hex_mark_blocks superblock 13\n') 
    fid.write('ic_hex_mark_blocks superblock 16\n') 
    fid.write('ic_hex_mark_blocks superblock 19\n') 
    fid.write('ic_hex_split_grid 57 13 0.95 m GEOM FLUID VORFN marked\n')  #suction BL upper cut

    fid.write('ic_hex_mark_blocks unmark\n') 
    fid.write('ic_hex_mark_blocks superblock 22\n') 
    fid.write('ic_hex_mark_blocks superblock 27\n') 
    fid.write('ic_hex_mark_blocks superblock 24\n') 
    fid.write('ic_hex_mark_blocks superblock 25\n') 
    fid.write('ic_hex_split_grid 58 90 0.05 m GEOM FLUID VORFN marked\n')  #suction BL lower cut

    fid.write('ic_undo_group_begin \n')
    fid.write('ic_hex_undo_major_start convert_to_unstruct\n')
    fid.write('ic_hex_convert_to_unstruct 21\n')
    fid.write('ic_hex_undo_major_end convert_to_unstruct\n')
    fid.write('ic_undo_group_end \n')



    # # Mesh size calculations
    # #Mesh parameters

    Dmch = 2*(math.tan(math.radians(alphaSuction/2))*Lmch + Dmix/2) 

    
    r=1.3 
    Delta_baseline = delta  #in the mixing chamber (x by x)
    DeltaMotive = Delta_baseline * Dmch/Dmix *0.9

    NyMixEstimate=(Dmix/2) / Delta_baseline
    Nybl = max(min(math.ceil(0.07*NyMixEstimate),10),3)

    aspect_mix =1.5 #aspect ratio, x size / y size

    rf = 1 
    t =0 
    for i in range(1,Nybl+1):
        t = t +  rf/r**i 


    #mixing section
    DeltaWall = Delta_baseline/(r**Nybl) 

    Nymix = math.ceil( (Dmix/2-t*Delta_baseline)/Delta_baseline) 



    Nymotive = max(math.ceil((DmotiveOut/2 - t*DeltaMotive)/ DeltaMotive ) ,2) 

    Nyremaining= max(Nymix -Nymotive,2)
    DeltaRemainder = (Dmch/2-DmotiveOut/2)/Nyremaining

    #lip and BL
    Nylip = math.ceil(ThicknessNozzle/(DeltaRemainder/1.5))
    Nylip = max(5,Nylip)

    #suction
    Nysuction = max(Nymix - Nymotive - Nylip - 2*Nybl ,2)

    L = (Dmch/2-(DmotiveOut/2+ThicknessNozzle))
    DeltaSuction = L/(Nysuction+2*t) 

    NxSuction = max( math.ceil((Lsuction)/(DeltaSuction)), 2)

    DeltaMch = (Delta_baseline* max(aspect_mix/2,1)  ) 

    NxMch = max(math.ceil(Lmch/DeltaMch) , 2)
    NxMix1 = max(math.ceil(Lmix/(Delta_baseline*aspect_mix)) , 2)

    DeltaMotiveInlet = DeltaMotive*DmotiveIn/DmotiveOut 

    NxMotive1 = max(math.ceil((LintletConst)/(DeltaMotiveInlet)) , 2)
    NxMotive2 = max(math.ceil((LinletConv)/((DeltaMotive+DeltaMotiveInlet)/2.5)) , 2)
    NxMotive3 = max(math.ceil(    (LinletDiff)/((DeltaMotive))  ) , 2)

    DeltaDiff = Delta_baseline*DdiffOut/Dmix * max(aspect_mix/2,1) 
    NxDiff = math.ceil(1.2*Ldiff/(DeltaDiff))
    NxDiffOut = math.ceil(Loutlet/(DeltaDiff)) 


    BL_s_upper = np.ndarray(5,float)
    BL_s_lower = np.ndarray(2,float)
    BL_m_upper = np.ndarray(4,float)

    BL_m_upper[3] = t*DeltaMotive *DmotiveOut/DmotiveOut 
    BL_m_upper[2] = t*DeltaMotive *Dthroat/DmotiveOut
    BL_m_upper[0] = t*DeltaMotive *DmotiveIn/DmotiveOut 
    BL_m_upper[1] = t*DeltaMotive *DmotiveIn/DmotiveOut

    BL_s_upper[0] = t*DeltaSuction 
    BL_s_upper[1] = t*DeltaSuction 
    BL_s_upper[2] = t*Delta_baseline 
    BL_s_upper[3] = t*Delta_baseline 
    BL_s_upper[4] = t*DeltaDiff 

    BL_s_lower[0] = t*DeltaSuction 
    BL_s_lower[1] = t*DeltaSuction 






    fid.write('ic_undo_group_begin\n')
    fid.write('ic_hex_mark_blocks unmark\n')
    fid.write('ic_hex_mark_blocks superblock 21\n')
    fid.write('ic_hex_mark_blocks superblock 22\n')
    fid.write('ic_hex_mark_blocks superblock 35\n')
    fid.write('ic_hex_mark_blocks superblock 29\n')
    fid.write('ic_hex_change_element_id VORFN\n')
    fid.write('ic_delete_empty_parts\n')
    fid.write('ic_undo_group_end\n')

    #  Outer verticies
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  49  }}\n'%(DmotiveIn/2)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  72  }}\n'%(DmotiveIn/2)) 
    fid.write('ic_hex_set_node_location x %f -csys global node_numbers {{  66  }}\n'%(x[2])) 
    fid.write('ic_hex_set_node_location x %f -csys global node_numbers {{  65  }}\n'%(x[2])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  66  }}\n'%(y[2])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  50  }}\n'%(DmotiveOut/2)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  58  }}\n'%(DmotiveOut/2+ThicknessNozzle)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  34  }}\n'%(Dmch/2)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  38  }}\n'%(Dmix/2)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  42  }}\n'%(Dmix/2)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  46  }}\n'%(DdiffOut/2)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  21  }}\n'%(DdiffOut/2)) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  68  }}\n'%(Dsuc/2))
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  67  }}\n'%y[5])  


    #  Inner Boundary Layer verticies

    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  77  }}\n'%(DmotiveIn/2-BL_m_upper[0])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  78  }}\n'%(DmotiveIn/2-BL_m_upper[1])) 
    fid.write('ic_hex_set_node_location x %f -csys global node_numbers {{  79  }}\n'%(x[2])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  79  }}\n'%(y[2]-BL_m_upper[2])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  80  }}\n'%(DmotiveOut/2-BL_m_upper[3])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  100  }}\n'%(DmotiveOut/2+ThicknessNozzle + BL_s_lower[1])) 

    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  90  }}\n'%(Dmch/2-BL_s_upper[1])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  91  }}\n'%(Dmix/2-BL_s_upper[2])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  92  }}\n'%(Dmix/2-BL_s_upper[3])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  93  }}\n'%(DdiffOut/2-BL_s_upper[4])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  94  }}\n'%(DdiffOut/2-BL_s_upper[4])) 
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  89  }}\n'%(Dsuc/2-BL_s_upper[0]))
    fid.write('ic_hex_set_node_location y %f -csys global node_numbers {{  99  }}\n'%(y[5]+ BL_s_lower[0]))  


    fid.write('ic_geo_set_part curve crv.12 INLET_M 0\n') 
    fid.write('ic_geo_set_part curve crv.5 INLET_S 0\n') 

    fid.write('ic_geo_set_part curve {') 
    for j in range(6,10):
        fid.write('crv.%d '%j) 
    for j in range(0,5):
        fid.write('crv.%d '%j) 
    fid.write('} WALL 0\n') 
    fid.write('ic_geo_set_part curve crv.10 OUTLET 0\n') 
    fid.write('ic_geo_set_part curve crv.11 AXIS 0\n') 


    # Assosiate

    fid.write("""ic_hex_find_comp_curve crv.12
    ic_undo_group_begin 
    ic_hex_set_edge_projection 11 77 0 1 crv.12
    ic_hex_set_edge_projection 77 49 0 1 crv.12
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.0
    ic_undo_group_begin 
    ic_hex_set_edge_projection 49 72 0 1 crv.0
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.1
    ic_undo_group_begin 
    ic_hex_set_edge_projection 72 66 0 1 crv.1
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.2
    ic_undo_group_begin 
    ic_hex_set_edge_projection 66 50 0 1 crv.2
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.3
    ic_undo_group_begin 
    ic_hex_set_edge_projection 50 58 0 1 crv.3
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.4
    ic_undo_group_begin 
    ic_hex_set_edge_projection 67 58 0 1 crv.4
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.5
    ic_undo_group_begin 
    ic_hex_set_edge_projection 99 89 0 1 crv.5
    ic_hex_set_edge_projection 67 99 0 1 crv.5
    ic_hex_set_edge_projection 89 68 0 1 crv.5
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.6
    ic_undo_group_begin 
    ic_hex_set_edge_projection 68 34 0 1 crv.6
    ic_hex_set_edge_projection 34 38 0 1 crv.6
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.7
    ic_undo_group_begin 
    ic_hex_set_edge_projection 38 42 0 1 crv.7
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.8
    ic_undo_group_begin 
    ic_hex_set_edge_projection 42 46 0 1 crv.8
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.9
    ic_undo_group_begin 
    ic_hex_set_edge_projection 46 21 0 1 crv.9
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.10
    ic_undo_group_begin 
    ic_hex_set_edge_projection 19 84 0 1 crv.10
    ic_hex_set_edge_projection 94 21 0 1 crv.10
    ic_undo_group_end 
    ic_hex_find_comp_curve crv.11
    ic_undo_group_begin 
    ic_hex_set_edge_projection 45 19 0 1 crv.11
    ic_hex_set_edge_projection 41 45 0 1 crv.11
    ic_hex_set_edge_projection 37 41 0 1 crv.11
    ic_hex_set_edge_projection 33 37 0 1 crv.11
    ic_hex_set_edge_projection 65 33 0 1 crv.11
    ic_hex_set_edge_projection 71 65 0 1 crv.11
    ic_hex_set_edge_projection 11 71 0 1 crv.11
    ic_undo_group_end \n""") 



    # # BLOCK TO FREE


    #Meshing

    # Y -direction
    fid.write('ic_hex_set_mesh 50 58 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nylip,DeltaMotive*DmotiveOut/Dthroat/r**(max(Nybl-4,1)),DeltaSuction/(r**Nybl),1.05,1.05)) #motive lip

    #suction and mix BLs
    fid.write('ic_hex_set_mesh 89 68 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaSuction,DeltaSuction/(r**Nybl),r,r) )#suction inlet
    fid.write('ic_hex_set_mesh 100 58 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaSuction,DeltaSuction/(r**Nybl),r,r) )
    fid.write('ic_hex_set_mesh 90 34 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaSuction,DeltaSuction/(r**Nybl),r,r) )
    fid.write('ic_hex_set_mesh 99 67 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaSuction,DeltaSuction/(r**Nybl),r,r) )#suction connection
    fid.write('ic_hex_set_mesh 91 38 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,Delta_baseline,DeltaWall,r,r) )
    fid.write('ic_hex_set_mesh 92 42 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,Delta_baseline,DeltaWall,r,r) ) # mixing
    fid.write('ic_hex_set_mesh 93 46 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,Delta_baseline*DdiffOut/Dmix,DeltaWall*DdiffOut/Dmix,r,r) ) #diffuser
    fid.write('ic_hex_set_mesh 94 21 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,Delta_baseline*DdiffOut/Dmix,DeltaWall*DdiffOut/Dmix,r,r) )

    #MOTIVE BL
    fid.write('ic_hex_set_mesh 77 49 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaMotive*DmotiveIn/DmotiveOut,DeltaMotive*DmotiveIn/Dthroat/(r**Nybl),r,r) )
    fid.write('ic_hex_set_mesh 78 72 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaMotive*DmotiveIn/DmotiveOut,DeltaMotive*DmotiveIn/Dthroat/(r**Nybl),r,r) )
    fid.write('ic_hex_set_mesh 79 66 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaMotive*Dthroat/DmotiveOut,DeltaMotive*Dthroat/DmotiveOut/(r**Nybl),r,r) )
    fid.write('ic_hex_set_mesh 80 50 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nybl,DeltaMotive,DeltaMotive/r**(max(Nybl-4,1)),r,r) )

    # Suction interior
    fid.write('ic_hex_set_mesh 89 99 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nysuction,0,0,r,r) )
    fid.write('ic_hex_set_mesh 100 90 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nysuction,0,0,r,r) )




    # Motive interior
    fid.write('ic_hex_set_mesh 11 77 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nymotive,0,0,r,r) )
    fid.write('ic_hex_set_mesh 71 78 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nymotive,0,0,r,r) )
    fid.write('ic_hex_set_mesh 65 79 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nymotive,0,0,r,r) )
    fid.write('ic_hex_set_mesh 33 80 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (Nymotive,0,0,r,r)  )



    # X -direction

    fid.write('ic_hex_set_mesh 68 34 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxSuction,DeltaSuction,DeltaMch,r,r) )
    fid.write('ic_hex_set_mesh 67 58 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxSuction,DeltaSuction,DeltaMch,r,r) )
    fid.write('ic_hex_set_mesh 99 100 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxSuction,DeltaSuction,DeltaMch,r,r) )
    fid.write('ic_hex_set_mesh 89 90 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxSuction,DeltaSuction,DeltaMch,r,r) )

    
    fid.write('ic_hex_set_mesh 34 38 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default copy_to_parallel unlocked\n'% (NxMch,DeltaMch,Delta_baseline*aspect_mix,r,r) )
    fid.write('ic_hex_set_mesh 38 42 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxMix1,0,0,r,r) )

    fid.write('ic_hex_set_mesh 42 46 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default copy_to_parallel unlocked\n'% (NxDiff,Delta_baseline*aspect_mix,Delta_baseline*aspect_mix*DdiffOut/Dmix,r,r) )
    fid.write('ic_hex_set_mesh 46 21 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxDiffOut,0,0,r,r) )

    fid.write('ic_hex_set_mesh 11 71 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxMotive1,0,0,r,r) )
    fid.write('ic_hex_set_mesh 71 65 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default copy_to_parallel unlocked\n'% (NxMotive2,DeltaMotiveInlet,DeltaMotive,r,r) )
    fid.write('ic_hex_set_mesh 65 33 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxMotive3,DeltaMotive,DeltaMch,r,r) )
    fid.write('ic_hex_set_mesh 66 50 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxMotive3,DeltaMotive,DeltaMch,r,r) )
    fid.write('ic_hex_set_mesh 79 80 n %d h1 %f h2 %f r1 %f r2 %f lmax 0 default unlocked\n'% (NxMotive3,DeltaMotive,DeltaMch,r,r) )






    if mesh_convergence==1:
        print('WARNING, superblocks not checked with new meshing mode')
        fid.write('ic_hex_mark_blocks unmark\n') 
        fid.write('ic_hex_mark_blocks superblock 10\n') 
        fid.write('ic_hex_mark_blocks superblock 13\n') 
        fid.write('ic_hex_mark_blocks superblock 16\n') 
        fid.write('ic_hex_mark_blocks superblock 19\n') 
        fid.write('ic_hex_mark_blocks superblock 22\n') 
        fid.write('ic_hex_mark_blocks superblock 23\n') 
        fid.write('ic_hex_mark_blocks superblock 24\n') 
        fid.write('ic_hex_mark_blocks superblock 25\n') 
        fid.write('ic_hex_mark_blocks superblock 26\n') 
        fid.write('ic_hex_mark_blocks superblock 27\n') 
        fid.write('ic_hex_mark_blocks superblock 28\n') 
        fid.write('ic_hex_mark_blocks superblock 29\n') 
        fid.write('ic_hex_mark_blocks superblock 30\n') 
        fid.write('ic_hex_mark_blocks superblock 31\n') 
        fid.write('ic_hex_mark_blocks superblock 32\n') 
        fid.write('ic_hex_mark_blocks superblock 33\n') 
        fid.write('ic_hex_mark_blocks superblock 34\n') 
        fid.write('ic_hex_mark_blocks superblock 35\n') 
        fid.write('ic_hex_mark_blocks superblock 36\n') 
        fid.write('ic_hex_mark_blocks superblock 37\n') 
        fid.write('ic_hex_mark_blocks superblock 38\n') 
        fid.write('ic_hex_mark_blocks superblock 39\n') 
        fid.write('ic_hex_mark_blocks superblock 4\n') 
        fid.write('ic_hex_mark_blocks superblock 40\n') 
        fid.write('ic_hex_mark_blocks superblock 41\n') 
        fid.write('ic_hex_mark_blocks superblock 42\n') 


    if refine==3:
        fid.write('ic_hex_set_refinement -1 3\n') 
    if refine==2:
        fid.write('ic_hex_set_refinement -1 2\n') 
    elif refine == 0.5:
        fid.write('ic_hex_set_refinement -1 1/2\n') 
    elif refine == 0.25:
        fid.write('ic_hex_set_refinement -1 1/4\n') 


    if mesh_smoothing ==1:
        fid.write(' ic_hex_smooth 5 GEOM FLUID INLET_M INLET_S WALL OUTLET AXIS elliptic iter_srf 10 iter_vol 5 exp_srf 0.0 exp_vol 0.0 niter_post 3 limit_post 0.2 smooth_type 202 nfix_layers -1 rebunch_edges 0 treat_unstruct 0 stabilize_srf 1.0 stabilize_vol 2.0 ortho_distance_srf 0 ortho_distance_vol 0 surface_fitting 1 keep_per_geom 1\n') 


    if (write_mesh==1):
    #  Producing mesh
        fid.write('ic_hex_create_mesh GEOM FLUID INLET_M INLET_S WALL OUTLET AXIS proj 2 dim_to_mesh 2 nproc 20\n') 
        fid.write('ic_boco_solver {ANSYS Fluent}\n') 
        fid.write('ic_solver_mesh_info {ANSYS Fluent}\n') 
        fid.write('ic_boco_save %s/ansys.fbc\n'%ScriptRootLocation) 
        fid.write('ic_boco_save_atr %s/ansys.atr\n'%ScriptRootLocation) 

        fid.write('ic_hex_write_file %s/hex.uns GEOM FLUID INLET_M INLET_S WALL OUTLET AXIS proj 2 dim_to_mesh 2 no_boco\n'%ScriptRootLocation) 
        fid.write('ic_unload_mesh \n') 
        fid.write('ic_delete_empty_parts \n') 
        fid.write('ic_uns_load %s/hex.uns 3 0 {} 1\n'%ScriptRootLocation) 
        fid.write('ic_uns_update_family_type visible {WALL INLET_M FLUID GEOM AXIS OUTLET ORFN INLET_S} {!LINE_2 QUAD_4} update 0\n') 
        fid.write('ic_boco_solver \n') 
        fid.write('ic_boco_clear_icons \n') 


        fid.write('ic_exec {C:/Program Files/ANSYS Inc/v195/icemcfd/win64_amd/icemcfd/output-interfaces/fluent6} -dom %s/hex.uns -b %s/ansys.fbc -dim2d %s/%s\n'% (ScriptRootLocation,ScriptRootLocation,MeshLocation,Name)) 
        fid.write('ic_uns_num_couplings \n') 
        fid.write('ic_undo_group_begin \n') 
        fid.write('ic_uns_create_diagnostic_edgelist 1\n') 
        fid.write('ic_uns_diagnostic subset all diag_type uncovered fix_fam FIX_UNCOVERED diag_verb {Uncovered faces} fams {{}} busy_off 1 quiet 1\n') 
        fid.write('ic_uns_create_diagnostic_edgelist 0\n') 
        fid.write('ic_undo_group_end \n') 
        fid.write('ic_uns_min_metric Quality {{}} {{}}\n') 
        fid.write('ic_unload_mesh\n') 
        fid.write('ic_delete_empty_parts\n') 
        fid.write('ic_uns_load %s/hex.uns 3 0 {{}} 1\n'%ScriptRootLocation) 
        fid.write('ic_uns_update_family_type visible {WALL INLET_M FLUID GEOM AXIS OUTLET ORFN INLET_S} {!LINE_2 QUAD_4} update 0\n') 
        fid.write('ic_boco_solver\n') 
        fid.write('ic_boco_clear_icons\n') 

    fid.close()

    # RUNNING ICEM PROGRAM
    if (run_ICEM == 1):
        command ='"C:/Program Files/ANSYS Inc/v192/icemcfd/win64_amd/bin/icemcfd.bat" -batch -script %s/AutoEjector_ICEMscript_%s.rpl & ' % (ScriptRootLocation,Name) 
        subprocess.call(command)



