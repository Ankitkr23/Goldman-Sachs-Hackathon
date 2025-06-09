import numpy as np
import pandas as pd
from scipy.optimize import minimize
import io
from scipy.stats import norm

# ----------------------- SETUP -----------------------
SIM_COUNT = 5800
TIME_DIVISIONS = 52
RNG_SEED = 987
np.random.seed(RNG_SEED)

# ----------------------- MARKET PARAMETERS -----------------------
ASSETS = ['ASSET_A', 'ASSET_B', 'ASSET_C']
INITIAL_PRICE = 100.0
INTEREST = 0.05
OPTION_PRICES = np.array([50, 75, 100, 125, 150])
EXPIRIES = np.array([1, 2, 5])

# ----------------------- ASSET CORRELATION -----------------------
correlation_matrix = np.array([
    [1.0, 0.75, 0.5],
    [0.75, 1.0, 0.25],
    [0.5, 0.25, 1.0]
])
chol_decomp = np.linalg.cholesky(correlation_matrix)

# ----------------------- PRICING MODELS -----------------------
def vanilla_option_price(asset_price, strike, time, rate, volatility):
    """Analytical formula for standard option pricing"""
    if time <= 0:
        return np.maximum(asset_price - strike, 0)
        
    d1_term = (np.log(asset_price/strike) + (rate + 0.5*volatility**2)*time) / (volatility*np.sqrt(time))
    d2_term = d1_term - volatility*np.sqrt(time)
    
    return asset_price * norm.cdf(d1_term) - strike * np.exp(-rate*time) * norm.cdf(d2_term)

def calculate_volatility(asset_price, strike, time, rate, market_price):
    """Binary search for implied volatility"""
    vol_min, vol_max = 0.01, 3.0
    
    for _ in range(20):
        vol_mid = (vol_min + vol_max)/2
        calculated_price = vanilla_option_price(asset_price, strike, time, rate, vol_mid)
        
        if calculated_price > market_price:
            vol_max = vol_mid
        else:
            vol_min = vol_mid
            
    return (vol_min + vol_max)/2

# ----------------------- VOLATILITY SURFACE -----------------------
def build_vol_surface(asset_name, calibration_data):
    """Creates a volatility surface for the given asset"""
    asset_data = calibration_data[calibration_data['Stock'] == asset_name]
    vol_grid = np.full((len(OPTION_PRICES), len(EXPIRIES)), 0.2)
    
    for i, strike in enumerate(OPTION_PRICES):
        for j, expiry in enumerate(EXPIRIES):
            data_row = asset_data[(asset_data['Strike'] == strike) & 
                                 (asset_data['Maturity'] == f"{expiry}y")]
            
            if not data_row.empty:
                market_price = float(data_row['Price'])
                vol_grid[i, j] = calculate_volatility(INITIAL_PRICE, strike, expiry, 
                                                   INTEREST, market_price)

    def objective_function(flat_vol_array):
        vol_matrix = flat_vol_array.reshape((len(OPTION_PRICES), len(EXPIRIES)))
        squared_errors = []
        
        for _, row in asset_data.iterrows():
            strike_idx = np.where(OPTION_PRICES == row['Strike'])[0][0]
            expiry_idx = np.where(EXPIRIES == int(row['Maturity'][:-1]))[0][0]
            
            theoretical_price = vanilla_option_price(
                INITIAL_PRICE, row['Strike'], 
                int(row['Maturity'][:-1]), INTEREST, 
                vol_matrix[strike_idx, expiry_idx]
            )
            
            squared_errors.append((theoretical_price - row['Price'])**2)
            
        return np.sum(squared_errors)

    optimization = minimize(
        objective_function, 
        vol_grid.flatten(), 
        bounds=[(0.01, 2.0)] * vol_grid.size, 
        options={'maxiter': 20}
    )
    
    return optimization.x.reshape((len(OPTION_PRICES), len(EXPIRIES)))

def create_vol_interpolator(vol_surface): 
    """Creates a function to interpolate volatility at any price and time"""
#     actually it is returning the nearest index of price and time i chose 
    def vol_at_point(price, time):
        # Ensure values are within bounds
        bounded_price = np.clip(price, OPTION_PRICES[0], OPTION_PRICES[-1])
        bounded_time = np.clip(time, EXPIRIES[0], EXPIRIES[-1])
        
        # Find interpolation indices
        price_idx = np.searchsorted(OPTION_PRICES, bounded_price) - 1
        time_idx = np.searchsorted(EXPIRIES, bounded_time) - 1
        
        # Ensure indices are valid
        price_idx = np.clip(price_idx, 0, len(OPTION_PRICES)-2)
        time_idx = np.clip(time_idx, 0, len(EXPIRIES)-2)
        
        return vol_surface[price_idx, time_idx]
        
    return vol_at_point

# ----------------------- PATH GENERATION -----------------------
def generate_asset_paths(vol_interpolators, duration, steps, path_count, correlation):
    """Generate correlated paths for multiple assets"""
    time_step = duration / steps
    sqrt_time_step = np.sqrt(time_step)
    path_values = np.full((steps+1, path_count, 3), INITIAL_PRICE)

    for step in range(1, steps+1):
        current_values = path_values[step-1]
        current_time = (step-1) * time_step

        # Calculate volatilities for current prices and time
        asset_vols = np.stack([
            vol_interpolators[asset_idx](current_values[:, asset_idx], current_time) 
            for asset_idx in range(3)
        ], axis=1)
        
        # Generate correlated random numbers
        random_samples = np.random.normal(size=(path_count, 3))
        correlated_samples = random_samples @ correlation.T
        
        # Calculate price movements
        drift_component = (INTEREST - 0.5 * asset_vols**2) * time_step
        random_component = asset_vols * sqrt_time_step * correlated_samples
        
        # Apply to prices
        path_values[step] = current_values * np.exp(drift_component + random_component)

    return path_values

# ----------------------- OPTION VALUATION -----------------------
def value_basket_knockout(vol_interpolators, correlation, option_params, path_count=SIM_COUNT):
    """Price a basket option with knockout feature"""
    duration = int(option_params['Maturity'][:-1])
    steps = duration * TIME_DIVISIONS
    knockout_level = float(option_params['KnockOut'])
    strike_price = float(option_params['Strike'])
    option_style = option_params['Type']

    # Generate price paths
    asset_paths = generate_asset_paths(vol_interpolators, duration, steps, path_count, correlation)
    
    # Calculate basket values (average across assets)
    basket_values = asset_paths.mean(axis=2)
    
    # Check knockout condition
    knockout_triggered = (basket_values >= knockout_level).any(axis=0)
    final_basket_values = basket_values[-1]

    # Calculate payoffs
    if option_style.lower() == 'call':
        payoffs = np.where(~knockout_triggered, np.maximum(final_basket_values - strike_price, 0), 0)
    else:  # put
        payoffs = np.where(~knockout_triggered, np.maximum(strike_price - final_basket_values, 0), 0)

    # Apply discount factor
    present_value = np.exp(-INTEREST * duration) * payoffs.mean()
    
    return round(present_value, 2)

def process_data():
    """Main function to process calibration data and price basket options"""
    # Load calibration data
    calibration_data = '''
    CalibIdx,Stock,Type,Strike,Maturity,Price
    1,ASSET_A,Call,50,1y,52.44
    2,ASSET_A,Call,50,2y,54.77
    3,ASSET_A,Call,50,5y,61.23
    4,ASSET_A,Call,75,1y,28.97
    5,ASSET_A,Call,75,2y,33.04
    6,ASSET_A,Call,75,5y,43.47
    7,ASSET_A,Call,100,1y,10.45
    8,ASSET_A,Call,100,2y,16.13
    9,ASSET_A,Call,100,5y,29.14
    10,ASSET_A,Call,125,1y,2.32
    11,ASSET_A,Call,125,2y,6.54
    12,ASSET_A,Call,125,5y,18.82
    13,ASSET_A,Call,150,1y,0.36
    14,ASSET_A,Call,150,2y,2.34
    15,ASSET_A,Call,150,5y,11.89
    16,ASSET_B,Call,50,1y,52.45
    17,ASSET_B,Call,50,2y,54.9
    18,ASSET_B,Call,50,5y,61.87
    19,ASSET_B,Call,75,1y,29.11
    20,ASSET_B,Call,75,2y,33.34
    21,ASSET_B,Call,75,5y,43.99
    22,ASSET_B,Call,100,1y,10.45
    23,ASSET_B,Call,100,2y,16.13
    24,ASSET_B,Call,100,5y,29.14
    25,ASSET_B,Call,125,1y,2.8
    26,ASSET_B,Call,125,2y,7.39
    27,ASSET_B,Call,125,5y,20.15
    28,ASSET_B,Call,150,1y,1.26
    29,ASSET_B,Call,150,2y,4.94
    30,ASSET_B,Call,150,5y,17.46
    31,ASSET_C,Call,50,1y,52.44
    32,ASSET_C,Call,50,2y,54.8
    33,ASSET_C,Call,50,5y,61.42
    34,ASSET_C,Call,75,1y,29.08
    35,ASSET_C,Call,75,2y,33.28
    36,ASSET_C,Call,75,5y,43.88
    37,ASSET_C,Call,100,1y,10.45
    38,ASSET_C,Call,100,2y,16.13
    39,ASSET_C,Call,100,5y,29.14
    40,ASSET_C,Call,125,1y,1.96
    41,ASSET_C,Call,125,2y,5.87
    42,ASSET_C,Call,125,5y,17.74
    43,ASSET_C,Call,150,1y,0.16
    44,ASSET_C,Call,150,2y,1.49
    45,ASSET_C,Call,150,5y,9.7
    '''
    calib_df = pd.read_csv(io.StringIO(calibration_data))
    
    # Rename the asset identifiers to match the new ASSETS constant
    calib_df['Stock'] = calib_df['Stock'].replace({
        'DTC': 'ASSET_A',
        'DFC': 'ASSET_B', 
        'DEC': 'ASSET_C'
    })

    # Load basket option parameters
    basket_options = '''Id,Asset,KnockOut,Maturity,Strike,Type
    1,Basket,150,2y,50,Call
    2,Basket,175,2y,50,Call
    3,Basket,200,2y,50,Call
    4,Basket,150,5y,50,Call
    5,Basket,175,5y,50,Call
    6,Basket,200,5y,50,Call
    7,Basket,150,2y,100,Call
    8,Basket,175,2y,100,Call
    9,Basket,200,2y,100,Call
    10,Basket,150,5y,100,Call
    11,Basket,175,5y,100,Call
    12,Basket,200,5y,100,Call
    13,Basket,150,2y,125,Call
    14,Basket,175,2y,125,Call
    15,Basket,200,2y,125,Call
    16,Basket,150,5y,125,Call
    17,Basket,175,5y,125,Call
    18,Basket,200,5y,125,Call
    19,Basket,150,2y,75,Put
    20,Basket,175,2y,75,Put
    21,Basket,200,2y,75,Put
    22,Basket,150,5y,75,Put
    23,Basket,175,5y,75,Put
    24,Basket,200,5y,75,Put
    25,Basket,150,2y,100,Put
    26,Basket,175,2y,100,Put
    27,Basket,200,2y,100,Put
    28,Basket,150,5y,100,Put
    29,Basket,175,5y,100,Put
    30,Basket,200,5y,100,Put
    31,Basket,150,2y,125,Put
    32,Basket,175,2y,125,Put
    33,Basket,200,2y,125,Put
    34,Basket,150,5y,125,Put
    35,Basket,175,5y,125,Put
    36,Basket,200,5y,125,Put'''
    basket_df = pd.read_csv(io.StringIO(basket_options))

    # Build volatility surfaces for each asset
    vol_surfaces = [build_vol_surface(asset, calib_df) for asset in ASSETS]
    
    # Create volatility interpolators
    vol_interpolators = [create_vol_interpolator(surface) for surface in vol_surfaces]

    # Price each basket option
    pricing_results = []
    for _, option in basket_df.iterrows():
        option_price = value_basket_knockout(vol_interpolators, chol_decomp, option)
        pricing_results.append({'Id': int(option['Id']), 'Price': option_price})

    # Output results
    print('Id,Price')
    for result in pricing_results:
        print(f"{result['Id']},{result['Price']}")

if __name__ == "__main__":
    process_data()
