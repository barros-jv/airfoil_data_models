
import os
import subprocess
import numpy as np
import pandas as pd

def xfoil_polar(airfoil, alpha, Re, n_iter=200):

    if os.path.exists("polar_file.txt"):
        os.remove("polar_file.txt")

    if os.path.exists("input_file.txt"):
        os.remove("input_file.txt")

    input_file = open("input_file.txt", 'w')
    input_file.write("PLOP\n")
    input_file.write("G F\n\n")
    input_file.write("LOAD "+airfoil+"\n")
    input_file.write("airfoil\n")
    input_file.write("PANE\n") # Set current-airfoil panel nodes ( 140 ) based on curvature
    #input_file.write("PCOP\n") # Set current-airfoil panel nodes directly from buffer airfoil points
    input_file.write("OPER\n")
    input_file.write("Visc {0}\n".format(Re))
    input_file.write("VPAR\n")
    #input_file.write("N 10\n")
    input_file.write("GB\n") # Default A = 6.7 and B = 0.75; Best A = 6.75 and B = 0.83
    input_file.write("6.7\n") # A
    input_file.write("0.75\n\n") # B
    input_file.write("PACC\n")
    input_file.write("polar_file.txt\n\n")
    input_file.write("ITER {0}\n".format(n_iter))
    input_file.write("ASeq {0} {1} {2}\n".format(alpha[0], alpha[1],alpha[2]))
    input_file.write("\n\n")
    input_file.write("quit\n")
    input_file.close()

    try:
        subprocess.call("xfoil.exe < input_file.txt", shell=True, timeout = 10)

        # Extract data from the loaded dataBuffer array
        polar_data = np.loadtxt("polar_file.txt", skiprows=12)

    except subprocess.TimeoutExpired:

        os.system('wmic process where name="xfoil.exe" delete')
        polar_data =  np.zeros((10, 3))
        pass

    # Extract data from the loaded dataBuffer array
    polar_data = np.loadtxt("polar_file.txt", skiprows=12)

    try:
        Alpha  = polar_data[:,0]
        Cl  = polar_data[:,1]
        Cd = polar_data[:,2]
    except IndexError:
        Alpha  = [0] * int((abs(alpha[0])+abs(alpha[1]))/alpha[2])
        Cl  = [0] * int((abs(alpha[0])+abs(alpha[1]))/alpha[2])
        Cd = [0] * int((abs(alpha[0])+abs(alpha[1]))/alpha[2])

    Airfoil_data = {
                    'Alpha': Alpha, 
                    'Cl': Cl,
                    'Cd': Cd,
                    }
    df_airfoil = pd.DataFrame(data=Airfoil_data)

    # Delete file after loading
    if os.path.exists('polar_file.txt'):
        os.remove('polar_file.txt')

    if os.path.exists("input_file.txt"):
        os.remove("input_file.txt")

    return Alpha, Cl, Cd