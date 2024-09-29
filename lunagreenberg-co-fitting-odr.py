import numpy as np;
import matplotlib.pyplot as plt;
import scipy.optimize as sp;
import math;
from scipy.odr import ODR, Model, Data, RealData;
import scipy.stats as stats;

#https://colorcodes.io
#https://405nm.com/color-to-wavelength/
FILE_DETAILS = [
    [ 'deepred1.txt', 737e-9 ],
    [ 'brightred1.txt', 678e-9 ],
    [ 'brightred2.txt', 678e-9 ],
    [ 'emeraldgreen1.txt', 502e-9 ],
    [ 'turquoisegreen1.txt', 498e-9 ],
    [ 'royalblue1.txt', 459e-9 ],
    [ 'violet1.txt', 400e-9 ],
    [ 'ultraviolet1.txt', 300e-9 ]
]

#SI Units
BOLTZMANN_CONST = 1.3806503e-23
PATH_LENGTH = .027
WAVELENGTH_EST = 650e-9
GAMMA_EST = 3/2
NUM_DENSITY_EST = 2.5e25
POLARIZABILITY_EST = 4.49e-25
PROP_CONSTANT_EST = 1e-10

PROP_CONST_EST = 8 * np.pi**3 * BOLTZMANN_CONST * ( NUM_DENSITY_EST * POLARIZABILITY_EST )**2 * PROP_CONSTANT_EST / 3 / WAVELENGTH_EST**4;

INITIAL_INTENSITY_EST = 3.5 

critical_pt = 0.0 #C - to be adjusted

PARAM_LIST = np.array( [ 'prop const', 'initial intensity', 'gamma' ] ); 
def intensity( beta, temperature ):
    prop_const = beta[ 0 ];
    initial_intensity = beta[ 1 ];
    gamma = beta[ 2 ];
    #C_l = 8 * np.pi**3 * BOLTZMANN_CONST * ( number_density * polarizability )**2 * prop_constant / 3;
    dt = temperature - critical_pt;
    denom = dt**gamma;
    
    #return a fit with a high chi-squared value so it ditches it for negative values
    if ( gamma <= 0 or prop_const <= 0 ):
        return [ 0 for _ in temperature ];
    
    #exp = -C_l * critical_pt / wavelength**4 * PATH_LENGTH / denom;
    exp = -prop_const * critical_pt * PATH_LENGTH / denom;
    finalval = initial_intensity * np.exp( exp );
    return finalval;

def prop_const_func( beta, wavelength ):
    C_l = beta[ 0 ];
    try:
        if ( beta.ndim > 1 ):
            wavelength = wavelength[ :, np.newaxis ];
        return C_l / wavelength**4;
    except:
        return C_l / wavelength**4;
def prop_const_func_curvefit( wavelength, C_l ):
    if ( C_l.ndim > 1 ):
        wavelength = wavelength[ :, np.newaxis ];
    return C_l / wavelength**4;
    


def chi_squared( function, data, params ):
    """
    Calculates the chi-squared value for an arbitrary function

    Args:
        function (lambda): predictive function f(x, ...) where ... are params
        data (array): 2d array, rows are x/y/sigma and cols are each data point
        params: various function parameters
    """
    predicted = function( params, data[ 0 ] );
    observed = np.array( data[ 1 ] );
    error = np.array( data[ 2 ] );
    if ( predicted.ndim > 1 ):
        observed = observed[ :, np.newaxis ];
        error = error[ :, np.newaxis ];
    
    temp = (np.abs( predicted - observed ) / error)**2
    return np.sum( temp, 0 );

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
    
    #aggregate all rows with the same x value, taking the average y value and combining uncertainties
    xvals = np.unique( data[ :, 0 ] );
    newdata = np.zeros( ( 0, 3 ) );
    for xval in xvals:
        yvals = data[ data[ :, 0 ] == xval, 1 ];
        yerr = np.std( yvals );
        yval = np.mean( yvals );
        newdata = np.vstack( ( newdata, [ xval, yval, yerr ] ) );

    return newdata

def plot_result( fitfunc, xdata, ydata, xerr, yerr, result, title, xaxis, yaxis ):
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

    ax.errorbar( xdata, ydata, xerr=xerr, yerr=yerr, fmt='o')
    x_line = np.linspace( np.min( xdata ), np.max( xdata ), 1000000 );
    ax.plot( x_line, fitfunc( result, x_line ) )

    ax.set_title( title )
    #ax.set_xlabel('Temperature (C)')
    #ax.set_ylabel('Voltage (V)')
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    plt.savefig( title[ 0:-4 ] + '.png' );
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
    
def degrees_of_freedom( func, xdata, ydata ):
    h = .0001;
    hatmatrix_diag = [];
    for y_i in range( 0, len( ydata ) ):
        ydata_plus = ydata.copy();
        ydata_plus[ y_i ] += h;
        odr_data_plus = RealData( xdata, ydata_plus, xerr, yerr );
        odr_plus = ODR( odr_data_plus, odr_model, \
                    beta0 = [ PROP_CONST_EST, \
                    INITIAL_INTENSITY_EST, \
                    GAMMA_EST ] );
        odr_plus.set_job( fit_type=0 );
        output_plus = odr_plus.run();
        
        yhat = func( parameters, xdata[ y_i ] );
        yhat_plus = func( output_plus.beta, xdata[ y_i ] );
        hatmatrix_diag.append( ( yhat_plus - yhat ) / h );
    hatmatrix_diag = np.array( hatmatrix_diag );
    effective_parameters = np.sum( hatmatrix_diag );
    return len( xdata ) - effective_parameters;
    
    

    
prop_constants = [];
prop_constants_errors = [];
for FILE_DETAIL in FILE_DETAILS:
    FILE_NAME = FILE_DETAIL[ 0 ];
    data = read_data( FILE_NAME );

    xdata = data[:, 0];
    ydata = data[:, 1];
    sigma_val = np.sqrt( np.std( ydata[ xdata > 45 ] )**2 + data[:, 2]**2 );

    critical_pt = xdata[ ydata == np.min( ydata ) ];
    try:
        critical_pt = critical_pt[ -1 ];
    except:
        _ = 1;
        
    ydata = ydata[ xdata > critical_pt ];
    sigma_val = sigma_val[ xdata > critical_pt ];
    xdata = xdata[ xdata > critical_pt ];

    ydata = ydata[ xdata < critical_pt + .5 ];
    sigma_val = sigma_val[ xdata < critical_pt + .5 ];
    xdata = xdata[ xdata < critical_pt + .5 ];



    xerr = [ .005 for _ in xdata ];
    yerr = sigma_val;
    #xdata, ydata, sigma = remove_outliers( intensity, xdata, ydata, yerr );

    odr_data = RealData( xdata, ydata, xerr, yerr );
    odr_model = Model( intensity );
    odr = ODR( odr_data, odr_model, \
                beta0 = [ PROP_CONST_EST, \
                INITIAL_INTENSITY_EST, \
                GAMMA_EST ] );
    odr.set_job( fit_type=0 );
    output = odr.run();

    parameters = output.beta;
    sigmas = output.sd_beta;

    dof = degrees_of_freedom( intensity, xdata, ydata );
    #dof = len( xdata ) - 3;

    CHI_SQUARED = output.res_var;
    RCS = CHI_SQUARED / ( dof );

    print( FILE_NAME );
    print( 'Effective parameters: ' + str( -dof + len( xdata ) ) );
    print( 'Chi-squared: {0:.3f}, reduced: {1:.3f}'.format( CHI_SQUARED, RCS ) );
    for parameter in parameters:
        param = PARAM_LIST[ parameters == parameter ][ 0 ];
        sigma = sigmas[ parameters == parameter ][ 0 ];
        print( '{2}: {0} +/- {1}'.format( parameter, sigma, param ) );
    print( '\n' );
    
    prop_constants.append( parameters[ PARAM_LIST == 'prop const' ][ 0 ] );
    prop_constants_errors.append( sigmas[ PARAM_LIST == 'prop const' ][ 0 ] );

    plot_result( intensity, xdata, ydata, xerr, yerr, parameters, FILE_NAME, 'Temperature (C)', 'Voltage (V)' );
    
wavelengths = np.array( [ float( stringval ) for stringval in np.array( FILE_DETAILS )[ :, 1 ] ] );
wavelengths_err = np.array( [ 5e-9 for _ in wavelengths ] );

odr_data = RealData( wavelengths, prop_constants, wavelengths_err, np.array( prop_constants_errors ) );
odr_model = Model( prop_const_func );
odr = ODR( odr_data, odr_model, beta0 = [ PROP_CONSTANT_EST ] );
odr.set_job( fit_type=0 );
output = odr.run();

dof = degrees_of_freedom( prop_const_func, wavelengths, np.array( prop_constants ) );

C_l = output.beta[ 0 ];
C_l_err = output.sd_beta[ 0 ];
RCS = chi_squared( prop_const_func, [ wavelengths, np.array( prop_constants ), np.array( prop_constants_errors ) ], [ C_l ] ) / dof;
print( '{2}: {0} +/- {1} - Chi squared: {3}, reduced {4}, effective parameters {5}'.format( C_l, C_l_err, 'C_l', RCS * dof, RCS, -dof + len( xdata ) ) );


plot_result( prop_const_func, wavelengths, prop_constants, wavelengths_err, np.array( prop_constants_errors ), [ C_l ], 'Rayleigh\'s law', 'Wavelength (m)', 'Proportionality Constant' );