#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 20:51:41 2025

@author: milosims
"""

#-----
#imports
import numpy as np
import matplotlib.pyplot as plt
#-----
#functions 
Masses = [2,2,2]
def N_body():
    ax = plt.figure().add_subplot(projection='3d')
    # initial positions and velocities for partciles 
    p1 = 0.347111
    p2 = 0.532728
    point1 = [-1,0,0]
    point2 = [1,0,0]
    point3 = [0,0,0]
    point4 = [1,0,0]
    V1 =[p1,p2,0]
    V2 =[p1,p2,0]
    V3 =[-2*p1,-2*p2,0]
    V4 =[p1,p2,0]
    points = np.array([[point1,point2,point3,point4]])
    velocities = np.array([[V1,V2,V3,V4]])
    
    data_x =[]
    data_y =[]
    data_z =[]
    for j in range(len(Masses)):
        #print(j)
        #print(((j+1)%len(Masses)))
        N=100
        delta = 0.0001
        
        E = np.array([])
        E_d = np.array([])
        Lz = np.array([])
        
        posx_1 = np.array([])
        posy_1 = np.array([])
        posz_1 = np.array([])
        r = np.array([])
        
        velx_h_1 = np.array([])
        vely_h_1 = np.array([])
        velz_h_1 = np.array([])
        
        velx_1 = np.array([])
        vely_1 = np.array([])
        velz_1 = np.array([])
        
        # initaialisation of data 
        
        posx_1 = np.append(posx_1, points[0,j,0])
        velx_1 = np.append(velx_1, velocities[0,j,0])
        
        posy_1 = np.append(posy_1, points[0,j,1])
        vely_1 = np.append(vely_1, velocities[0,j,1])
        
        posz_1 = np.append(posz_1, points[0,j,2])
        velz_1 = np.append(velz_1, velocities[0,j,2])
        #print(posx_1,posy_1,posz_1)
        posx_2 = np.array([])
        posy_2 = np.array([])
        posz_2 = np.array([])
        
        velx_h_2 = np.array([])
        vely_h_2 = np.array([])
        velz_h_2 = np.array([])
        
        velx_2 = np.array([])
        vely_2 = np.array([])
        velz_2 = np.array([])
        
        # initaialisation of data 
        posx_2 = np.append(posx_2, points[0,((j+1)%len(Masses)),0])
        velx_2 = np.append(velx_2, velocities[0,((j+1)%len(Masses)),0])
        
        posy_2 = np.append(posy_2, points[0,((j+1)%len(Masses)),1])
        vely_2 = np.append(vely_2, velocities[0,((j+1)%len(Masses)),1])
        
        posz_2 = np.append(posz_2, points[0,((j+1)%len(Masses)),2])
        velz_2 = np.append(velz_2, velocities[0,((j+1)%len(Masses)),2])
        def Force(x1,x2,M1,M2,r):
            G = 1
            F = -((G*(M1))*(x1-x2))/((r)**3+0.01)-((G*(M2))*(x1-x2))/((r)**3+0.01)
            return F
        
        for i in range(N-1):
            r = np.append(r, np.sqrt((posx_1[i]-posx_2[i])**2+(posy_1[i]-posy_2[i])**2+(posz_1[i]-posz_2[i])**2))
            # x position particle 1
            velx_h_1 = np.append(velx_h_1,velx_1[i]+0.5*delta*Force(posx_1[i],posx_2[i],Masses[j],Masses[(j+1)%len(Masses)],r[i]))
            posx_1 = np.append(posx_1, posx_1[i]+delta*velx_h_1[i])
            # x position particle 2
            velx_h_2 = np.append(velx_h_2,velx_2[i]+0.5*delta*Force(posx_2[i],posx_1[i],Masses[(j+1)%len(Masses)],Masses[j],r[i]))
            posx_2 = np.append(posx_2, posx_2[i]+delta*velx_h_2[i])
            
            # y position particle 1
            vely_h_1 = np.append(vely_h_1,vely_1[i]+0.5*delta*Force(posy_1[i],posy_2[i],Masses[j],Masses[(j+1)%len(Masses)],r[i]))
            posy_1 = np.append(posy_1, posy_1[i]+delta*vely_h_1[i])
            # y position particle 2
            vely_h_2 = np.append(vely_h_2,vely_2[i]+0.5*delta*Force(posy_2[i],posy_1[i],Masses[(j+1)%len(Masses)],Masses[j],r[i]))
            posy_2 = np.append(posy_2, posy_2[i]+delta*vely_h_2[i])
            
            # z position particle 1
            velz_h_1 = np.append(velz_h_1,velz_1[i]+0.5*delta*Force(posz_1[i],posz_2[i],Masses[j],Masses[(j+1)%len(Masses)],r[i]))
            posz_1 = np.append(posz_1, posz_1[i]+delta*velz_h_1[i])
            # z position particle 2
            velz_h_2 = np.append(velz_h_2,velz_2[i]+0.5*delta*Force(posz_2[i],posz_2[i],Masses[(j+1)%len(Masses)],Masses[j],r[i]))
            posz_2 = np.append(posz_2, posz_2[i]+delta*velz_h_2[i])
            
            r = np.append(r, np.sqrt((posx_1[i+1]-posx_2[i+1])**2+(posy_1[i+1]-posy_2[i+1])**2+(posz_1[i+1]-posz_2[i+1])**2))
            # next x position particle 1 and 2
            velx_1 = np.append(velx_1, velx_h_1[i]+0.5*delta*Force(posx_1[i+1],posx_2[i+1], Masses[j], Masses[(j+1)%len(Masses)], r[i+1]))
            velx_2 = np.append(velx_2, velx_h_2[i]+0.5*delta*Force(posx_2[i+1],posx_1[i+1], Masses[(j+1)%len(Masses)], Masses[j], r[i+1]))
            # next x position particle 1 and 2
            vely_1 = np.append(vely_1, vely_h_1[i]+0.5*delta*Force(posy_1[i+1],posy_2[i+1], Masses[j], Masses[(j+1)%len(Masses)], r[i+1]))
            vely_2 = np.append(vely_2, vely_h_2[i]+0.5*delta*Force(posy_2[i+1],posy_1[i+1], Masses[(j+1)%len(Masses)], Masses[j], r[i+1]))
            # next z position particle 1 and 2
            velz_1 = np.append(velz_1, velz_h_1[i]+0.5*delta*Force(posz_1[i+1],posz_2[i+1], Masses[j], Masses[(j+1)%len(Masses)], r[i+1]))
            velz_2 = np.append(velz_2, velz_h_2[i]+0.5*delta*Force(posz_2[i+1],posz_1[i+1], Masses[(j+1)%len(Masses)], Masses[j], r[i+1]))
            
            E = np.append(E,energy(velx_1[i], vely_1[i], velz_1[i], Masses[j],Masses[(j+1)%len(Masses)], r[i]))
            Lz = np.append(Lz,angular_momentum_z(velx_1[i],vely_1[i],Masses[j],posx_1[i],posy_1[i]))
            E_d = np.append(E_d,((energy(velx_1[i], vely_1[i], velz_1[i], Masses[j],Masses[(j+1)%len(Masses)], r[i])-energy(velx_1[0], vely_1[0], velz_1[0], Masses[j],Masses[(j+1)%len(Masses)], r[0]))/energy(velx_1[0], vely_1[0], velz_1[0], Masses[j],Masses[(j+1)%len(Masses)], r[i]))*100)
            
        data_x.append( posx_1)
        data_x.append( posx_2)
        #print(posx_1,posx_2)    
        data_y.append( posy_1)
        data_y.append( posy_2)
            
        data_z.append( posy_1)
        data_z.append( posy_2)
        
        print(data_x[j],data_y[j],data_z[j])
        ts = np.linspace(0,(N*delta),N-1)
        ax.plot(data_x[j], data_y[j], data_z[j], label="body "+str(j+1))
        ax.legend(loc='upper left')
        #ax.plot(data_x[1], data_y[1], data_z[1], label="body "+str(j))
        Plot_function_w(ts, E, "Energy mass " + str(j), "Energy", "time")
        Plot_function_w(ts, Lz, "angular momentum mass " + str(j), "Lz", "time")
        Plot_function_w(ts, E_d, "Energy diffrence mass " + str(j), "Energy diffrence (%)", "time")
    #ax.scatter(posx_1, posy_1, posz_1, label="Body 1")
    #ax.scatter(posx_2, posy_2, posz_2, label="Body 2")
    ax.legend(loc='upper left')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=45., azim=-35, roll=0)
    plt.show()
"""
def Verlet(N,delta,posx1,velx1,posy1,vely1,posz1,velz1,Masses[j],Masses[j+1],r1,r2,r3):
    E = np.array([])
    E_d = np.array([])
    Lz = np.array([])
    
    posx_1 = np.array([])
    posy_1 = np.array([])
    posz_1 = np.array([])
    r = np.array([])
    
    velx_h_1 = np.array([])
    vely_h_1 = np.array([])
    velz_h_1 = np.array([])
    
    velx_1 = np.array([])
    vely_1 = np.array([])
    velz_1 = np.array([])
    
    # initaialisation of data 
    posx_1 = np.append(posx_1, posx1)
    velx_1 = np.append(velx_1, velx1)
    
    posy_1 = np.append(posy_1, posy1)
    vely_1 = np.append(vely_1, vely1)
    
    posz_1 = np.append(posz_1, posz1)
    velz_1 = np.append(velz_1, velz1)
    
    posx_2 = np.array([])
    posy_2 = np.array([])
    posz_2 = np.array([])
    
    velx_h_2 = np.array([])
    vely_h_2 = np.array([])
    velz_h_2 = np.array([])
    
    velx_2 = np.array([])
    vely_2 = np.array([])
    velz_2 = np.array([])
    
    # initaialisation of data 
    posx_2 = np.append(posx_2, r1)
    velx_2 = np.append(velx_2, 0.5)
    
    posy_2 = np.append(posy_2, r2)
    vely_2 = np.append(vely_2, 0)
    
    posz_2 = np.append(posz_2, r3)
    velz_2 = np.append(velz_2, 0)
    
    for i in range(N-1):
        r = np.append(r, np.sqrt((posx_1[i]-posx_2[i])**2+(posy_1[i]-posy_2[i])**2+(posz_1[i]-posz_2[i])**2))
        
        velx_h_1 = np.append(velx_h_1,velx_1[i]+0.5*delta*Force(posx_1[i],Masses[j],Masses[j+1],r[i]))
        posx_1 = np.append(posx_1, posx_1[i]+delta*velx_h_1[i])
        velx_1 = np.append(velx_1, velx_h_1[i]+0.5*delta*Force(posx_1[i+1], Masses[j], Masses[j+1], r[i]))
        
        vely_h_1 = np.append(vely_h_1,vely_1[i]+0.5*delta*Force(posy_1[i],Masses[j],Masses[j+1],r[i]))
        posy_1 = np.append(posy_1, posy_1[i]+delta*vely_h_1[i])
        vely_1 = np.append(vely_1, vely_h_1[i]+0.5*delta*Force(posy_1[i+1], Masses[j], Masses[j+1], r[i]))
    
        velz_h_1 = np.append(velz_h_1,velz_1[i]+0.5*delta*Force(posz_1[i],Masses[j],Masses[j+1],r[i]))
        posz_1 = np.append(posz_1, posz_1[i]+delta*velz_h_1[i])
        velz_1 = np.append(velz_1, velz_h_1[i]+0.5*delta*Force(posz_1[i+1], Masses[j], Masses[j+1], r[i]))
        
        velx_h_2 = np.append(velx_h_2,velx_2[i]+0.5*delta*Force(posx_2[i],Masses[j],Masses[j+1],r[i]))
        posx_2 = np.append(posx_2, posx_2[i]+delta*velx_h_2[i])
        velx_2 = np.append(velx_2, velx_h_2[i]+0.5*delta*Force(posx_2[i+1], Masses[j], Masses[j+1], r[i]))
        
        vely_h_2 = np.append(vely_h_2,vely_2[i]+0.5*delta*Force(posy_2[i],Masses[j],Masses[j+1],r[i]))
        posy_2 = np.append(posy_2, posy_2[i]+delta*vely_h_2[i])
        vely_2 = np.append(vely_2, vely_h_2[i]+0.5*delta*Force(posy_2[i+1], Masses[j], Masses[j+1], r[i]))
    
        velz_h_2 = np.append(velz_h_2,velz_2[i]+0.5*delta*Force(posz_2[i],Masses[j],Masses[j+1],r[i]))
        posz_2 = np.append(posz_2, posz_2[i]+delta*velz_h_2[i])
        velz_2 = np.append(velz_2, velz_h_2[i]+0.5*delta*Force(posz_2[i+1], Masses[j], Masses[j+1], r[i]))
        
        E = np.append(E,energy(velx_1[i], vely_1[i], velz_1[i], Masses[j],Masses[j+1], r[i]))
        Lz = np.append(Lz,angular_momentum_z(velx_1[i],vely_1[i],Masses[j],posx_1[i],posy_1[i]))
        E_d = np.append(E_d,((energy(velx_1[i], vely_1[i], velz_1[i], Masses[j],Masses[j+1], r[i])-energy(velx_1[0], vely_1[0], velz_1[0], Masses[j],Masses[j+1], r[0]))/energy(velx_1[0], vely_1[0], velz_1[0], Masses[j],Masses[j+1], r[0]))*100)
    return posx_1,posy_1,velx_1,vely_1,posz_1,velz_1,E,Lz,E_d,posx_2,posy_2,velx_2,vely_2,posz_2,velz_2
"""
def energy(vx,vy,vz,M1,M2,r):
    E = 0.5*(M1*M2)/(M1+M2)*(vx**2+vy**2+vz**2)+Potential(M1, M2, r)
    return E    

def Potential(M1,M2,r):
    G = 1
    U = -(G*M1*M2)/(r)
    return U
def angular_momentum_z(vx,vy,M1,x,y):
    Px = vx*M1
    Py = vy*M1
    L = x*Py-Px*y
    return L
def Plot_function(xs,ys,z,n_y,n_x):
    """plots a function given the parametrs allows the user to edit the legend and x and y labels for a specfic graph"""
    graph = plt.figure()
    ax = graph.add_subplot()
    ax.set_ylabel(str(n_y)) # y axis label
    ax.set_xlabel(str(n_x)) # x axis label
    ax.plot(xs,ys, linestyle='-', color='black', label=str(z))
    ax.scatter(0,0)
    ax.legend(loc='upper left')
    plt.grid()
def Plot_function_w(xs,ys,z,n_y,n_x):
    """plots a function given the parametrs allows the user to edit the legend and x and y labels for a specfic graph"""
    graph = plt.figure()
    ax = graph.add_subplot()
    ax.set_ylabel(str(n_y)) # y axis label
    ax.set_xlabel(str(n_x)) # x axis label
    ax.plot(xs,ys, linestyle='-', color='black', label=str(z))
    ax.legend(loc='upper left')
    plt.grid()
    
def Plot_3d(xs,ys,zs,z):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xs, ys, zs, color='black', label=str(z))
    ax.scatter(0,0,0)
    ax.legend(loc='upper left')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=90., azim=-35, roll=0)
    plt.show()

    
#-----
# main program

N_body()
    