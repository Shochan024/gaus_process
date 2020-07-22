#-*-coding:utf-8-*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaus( X , dataset ):
    ###########################
    #   define basic matrix   #
    ###########################
    U = np.mean( dataset , axis=0 )
    S = np.cov( dataset.T )

    ####################################
    #    calculate gaus dev items      #
    ####################################
    X_U = X.T - U
    inv_cov = np.linalg.inv( S )
    gaus_item = -1/2 * X_U.dot( inv_cov ).dot( X_U.T )
    const_term = 1 / np.sqrt( 1*np.pi )**X.shape[0] * np.linalg.det( S )

    return  const_term * np.exp( np.diag( gaus_item ) )

###########################
#    define matrix        #
###########################
X = np.arange(-10,10,0.5)
Y = np.arange(-10,10,0.5)
X,Y = np.meshgrid( X , Y )
dataset = np.random.randint(-10,10,(100,2))

Z = np.array([X.ravel(),Y.ravel()])
Z = gaus( Z , dataset ).reshape( X.shape )

###########################
#   plot gaus dev         #
###########################
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.plot_surface( X,Y,Z , cmap="plasma" )

plt.savefig( "./imgs/muti_gaus_dev.png" )
