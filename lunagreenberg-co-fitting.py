import numpy as np;
import matplotlib.pyplot as plt;
import scipy.optimize as sp;
import math;

FILE_NAME_1 = 'violet1.txt'
FILE_NAME_2 = 'brightred2.txt'

#SI Units
BOLTZMANN_CONST = 1.3806503e-23
PATH_LENGTH = .027
WAVELENGTH_EST = 650e-9
GAMMA_EST = 11/2
NUM_DENSITY_EST = 2.5e25
POLARIZABILITY_EST = 4.49e-30
PROP_CONSTANT_EST = 1e-14

INITIAL_INTENSITY_EST = 3.5 

critical_pt = 0.0 #C

PARAM_LIST = np.array( [ 'wavelength', 'initial intensity', 'gamma', 'number density', 'polarizability', 'prop_constant' ] );
def intensity( temperature, wavelength, initial_intensity, gamma, number_density, polarizability, prop_constant ):
    C_l = 8 * np.pi**3 * BOLTZMANN_CONST * ( number_density * polarizability )**2 * prop_constant / 3;
    dt = temperature - critical_pt;
    denom = dt**gamma;
    
    #return a fit with a high chi-squared value so it ditches it for negative values
    if ( gamma < 0 or wavelength < 0 ):
        return [ 0 for _ in temperature ];
    
    exp = -C_l * critical_pt / wavelength**4 * PATH_LENGTH / denom;
    finalval = initial_intensity * np.exp( exp );
    return finalval;
    


def chi_squared( function, data, *params ):
    """
    Calculates the chi-squared value for an arbitrary function

    Args:
        function (lambda): predictive function f(x, ...) where ... are params
        data (array): 2d array, rows are x/y/sigma and cols are each data point
        params: various function parameters
    """
    predicted = function( data[ 0 ], *params );
    observed = data[ 1 ];
    error = data[ 2 ];
    
    return np.sum((np.abs( predicted - observed ) / error)**2)

def read_data(file_name):
    """
    Reads in data given file name.

    Parameters
    ----------
    file_name : string

    Returns
    -------
    data : np.array of floats
        Data should be in format [x, y, uncertainty on y]
    """
    # Write a function that reads the data accordingly.
    # Extra: what does it do if it cannot find the file?
    input_file = open( file_name, 'r' )
        
    data = np.zeros((0, 2))
    SKIPPED_FIRST_LINE = False
    for line in input_file:
        if not SKIPPED_FIRST_LINE:
            SKIPPED_FIRST_LINE = True

        else:
            split_up = line.split( '\t' )
            try:
                temp = np.array([float(split_up[0]), float(split_up[1])])
                for i in range( 0, 2 ):
                    if ( math.isnan( temp[ i ] ) or 0 == temp[ i ] ):
                        temp = None;
                if ( temp is None ):
                    continue;
            except:
                continue;

            data = np.vstack((data, temp))

    input_file.close()

    return data

def plot_result( fitfunc, xdata, ydata, sigma, result):
    """
    Plots result.

    Parameters
    ----------
    data : numpy array of floats
        Should be in format (x, y, y_uncertainty)
    result : float
        Optimum value for coefficient

    Returns
    -------
    None.

    """

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.errorbar( xdata, ydata, yerr=sigma, fmt='o')
    x_line = np.linspace( xdata[ 0 ], xdata[ -1 ], 1000000 );
    ax.plot( x_line, fitfunc( x_line, *result[ 0 ] ) )

    ax.set_title('Data and line of best fit')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.show()
    
def remove_outliers( f, x, values, errors ):
    result = sp.curve_fit( f, xdata=x, ydata=values, sigma=errors, nan_policy='raise', absolute_sigma=True );
    predicted = f( x, *result[ 0 ] );
    differences = np.abs( predicted - values );
    
    if ( len( values[ differences < 3 * errors ] ) != len( values ) ):
        #values are deleted here via bool mask
        return remove_outliers( f, x[ np.max( differences ) != differences ],
                                values[ np.max( differences ) != differences ], 
                                errors[ np.max( differences ) != differences ] );
    else: return x, values, errors;
    
    

    
#data = np.vstack( (read_data( FILE_NAME_1 ), read_data( FILE_NAME_2 )) );
data = read_data( FILE_NAME_1 );

xdata = data[:, 0];
ydata = data[:, 1];
sigma_val = np.std( ydata[ xdata > 45 ] );

critical_pt = xdata[ ydata == np.min( ydata ) ];
try:
    critical_pt = critical_pt[ -1 ];
except:
    _ = 1;
    
ydata = ydata[ xdata > critical_pt ];
xdata = xdata[ xdata > critical_pt ];

ydata = ydata[ xdata < critical_pt + .5 ];
xdata = xdata[ xdata < critical_pt + .5 ];

sigma = [];
for _ in ydata:
    sigma.append( sigma_val );
sigma = np.array( sigma );


#xdata, ydata, sigma = remove_outliers( intensity, xdata, ydata, sigma );

result = sp.curve_fit( intensity, xdata, ydata, sigma=sigma, nan_policy='raise', absolute_sigma=True, method='lm', maxfev=100000,\
    p0=[ WAVELENGTH_EST, INITIAL_INTENSITY_EST, GAMMA_EST, NUM_DENSITY_EST, POLARIZABILITY_EST, PROP_CONSTANT_EST ] );
parameters = result[ 0 ];
sigmas = np.sqrt( np.diag( result[ 1 ] ) );
CHI_SQUARED = chi_squared( intensity, [xdata, ydata, sigma], *result[ 0 ] );
RCS = CHI_SQUARED / len( xdata );

print( 'Chi-squared: {0:.3f}, reduced: {1:.3f}'.format( CHI_SQUARED, RCS ) );
for parameter in parameters:
    param = PARAM_LIST[ parameters == parameter ][ 0 ];
    sigma = sigmas[ parameters == parameter ][ 0 ];
    print( '{2}: {0} +/- {1}'.format( parameter, sigma, param ) );


plot_result( intensity, xdata, ydata, sigma, result );