import numpy as np
from numpy.linalg import inv
from math import floor, pi
from scipy.interpolate import interp1d

def nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n/k * nchoosek(n-1, k-1)
    return round(r)

def Coord2Bezier(airfoil, cps):

    # Entradas
    xf=np.array(airfoil.x)
    yf=np.array(airfoil.y)
    N=xf.shape[0]
    deg = cps-1                       # degree
    #cps=deg+1                        # the number of control points which is degree+1
    t=np.linspace(0,1,floor((N+1)/2)) # t is in [0,1]

    #
    xsup = np.zeros( (floor((N+1)/2)), dtype='float')
    ysup = np.zeros( (floor((N+1)/2)), dtype='float')
    xinf = np.zeros( (floor((N+1)/2)), dtype='float')
    yinf = np.zeros( (floor((N+1)/2)), dtype='float')

    for i in np.arange(0,(N+1)/2, dtype=int):
        xsup[i]=xf[int((N+1)/2-1-i)]
        ysup[i]=yf[int((N+1)/2-1-i)]

    for i in np.arange((N+1)/2-1,N, dtype=int):
        xinf[int(i-(N+1)/2+1)]=xf[i]
        yinf[int(i-(N+1)/2+1)]=yf[i]
    
    J  = np.zeros( (floor((N+1)/2),cps), dtype='float')
    xU = np.zeros( (floor((N+1)/2)), dtype='float')
    yU = np.zeros( (floor((N+1)/2)), dtype='float')
    xL = np.zeros( (floor((N+1)/2)), dtype='float')
    yL = np.zeros( (floor((N+1)/2)), dtype='float')

    for i in np.arange(0, (N+1)/2, dtype=int):
        for s in np.arange(0,cps, dtype=int):
            J[i,s]=nchoosek(deg,s)*(t[i]**(s))*((1-t[i])**(deg-s))

    xsupB = np.zeros( (cps), dtype='float')
    ysupB = np.zeros( (cps), dtype='float')
    xinfB = np.zeros( (cps), dtype='float')
    yinfB = np.zeros( (cps), dtype='float')

    # BX and BY are x-coordinate and y-coordinate of Bezier control % points

    xsupB = inv(np.dot(np.transpose(J),J))@J.transpose()@xsup
    ysupB = inv(np.dot(np.transpose(J),J))@J.transpose()@ysup

    xinfB =inv(np.dot(np.transpose(J),J))@J.transpose()@xinf
    yinfB =inv(np.dot(np.transpose(J),J))@J.transpose()@yinf

    # Control points at leading edge and trailing edge
    # Upper surface 
    xsupB[0]=xf[int((N+1)/2-1)]
    xsupB[-1]=xf[0]
    ysupB[0]=yf[int((N+1)/2-1)]
    ysupB[-1]=yf[0]

    # Lower surface
    xinfB[0]=xf[int((N+1)/2-1)]
    xinfB[-1]=xf[-1]
    yinfB[0]=yf[int((N+1)/2-1)]
    yinfB[-1]=yf[-1]

    return xsupB,ysupB,xinfB,yinfB

def Bezier2Coord(xsupB,ysupB,xinfB,yinfB,nPts = 200, dist = 'Regular'):

    cps=len(xsupB)
    deg=cps-1                     # degree of Bezier curve (can be changed)
    if dist == 'Regular':
        t = np.linspace(0,1,nPts)
    else:
        Beta = np.linspace(0,pi,nPts)
        t = 0.5*(1-np.cos(Beta))

    J  = np.zeros( (nPts,cps), dtype='float')

    for i in np.arange(0, nPts, dtype=int):
        for s in np.arange(0,cps, dtype=int):
            J[i,s]=nchoosek(deg,s)*(t[i]**(s))*((1-t[i])**(deg-s))

    xBezierSup = np.zeros( (nPts), dtype='float')
    yBezierSup = np.zeros( (nPts), dtype='float')
    xBezierInf = np.zeros( (nPts), dtype='float')
    yBezierInf = np.zeros( (nPts), dtype='float')

    xBezierSup = J@xsupB
    yBezierSup = J@ysupB
    xBezierInf = J@xinfB
    yBezierInf = J@yinfB

    return xBezierSup, yBezierSup, xBezierInf, yBezierInf

def comparar_superficies(x_aero, y_aero, x_bezier, y_bezier):
    # Interpolação da curva de Bézier nos pontos x do aerofólio
    interp_bezier = interp1d(x_bezier, y_bezier, kind='linear', fill_value="extrapolate")
    y_bezier_interp = interp_bezier(x_aero)

    # Cálculo do erro absoluto
    erro_absoluto = np.abs(np.array(y_aero) - y_bezier_interp)

    # Cálculo do erro quadrático médio (MSE)
    erro_quadratico_medio = np.mean(erro_absoluto**2)

    return erro_absoluto, erro_quadratico_medio, y_bezier_interp