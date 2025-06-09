# Goldman-Sachs-Hackathon
Goldman Sachs India Hackathon 2025

                                                      Optimal Hedging Strategy 

                                                          Problem Set -1
Description: Market making is a critical activity performed by financial institutions. These firms provide liquidity to financial markets by quoting bid and ask prices for various assets, enabling investors to buy and sell securities efficiently. Market makers generate revenue primarily through the bid-ask spread. They also profit from inventory management and taking positions based on market views.

However, market making activities inherently expose firms to market risk. Market risk refers to the potential for losses arising from changes in market conditions, such as fluctuations in asset prices, interest rates, or exchange rates etc. For an equities trader, this risk can manifest as potential losses due to adverse movements in the prices of the stocks they hold in their inventory or are obligated to trade.

To mitigate market risk, financial institutions employ various hedging strategies. Hedging involves taking offsetting positions in related assets to reduce the overall exposure to adverse market movements. The costs associated with hedging include transaction fees, bid-ask spreads, and the capital cost of holding the hedges.

Objective: In this hackathon, you'll be stepping into the shoes of an equities trader facing market risk from an unhedged portfolio comprised of both equity stock and derivative products. You are given an unhedged portfolio's historical Profit and Loss (P&L) distribution, simulated using the last year's historical returns of the portfolio's constituent assets.

Ideally, hedging can be done with liquid equity stock to minimize transaction costs and ensure the ability to adjust positions quickly. Your objective is to identify a set of hedging stocks from a universe of publicly traded equity stocks and determine the quantity of each stock needed to minimize risk, while also considering the cost of implementing the hedge. These equity stocks have known risk profiles, represented by their historical daily returns, and a known hedging cost per share. The daily return of a stock is the percentage change in its price from one day to the next. For example, if a stock's price was 100 yesterday and 101 today, the daily return would be (101-100)/100 = 0.01 or 1%.

The effectiveness of your hedge will be evaluated based on the historical Value at Risk (VaR) of the hedged portfolio. Value at Risk (VaR) is a statistical measure of the potential loss in value of an asset or portfolio over a defined period for a given confidence level. VaR (confidence level) = The (1 - confidence level)th percentile of the portfolio's return distribution.

Here's how you can calculate 95%ile VaR using Python:

import numpy as np

portfolio_pnl = np.array([-10, -5, 0, 2, 5, -20, -12, -3, 7, 10]) # Example Portfolio P&L

VaR_95 = np.percentile(portfolio_pnl, 5)
print(VaR_95)

# In this code, np.percentile(portfolio_pnl, 5) calculates the 5th percentile of the portfolio_pnl array, which represents the 95% VaR.
Hedging can involve taking long or short positions in the hedging stocks. Going long means buying the stock (profiting from price increases), while going short means selling borrowed stock (profiting from price decreases). The cost of hedging can be assumed to be a simple linear sum of absolute value of quantity times cost for each stock. In other words, if you buy 'x' shares of Stock A, which costs C_A per share, and ′y′ shares of Stock B, which costs C_B per share the total cost is:

Total Cost = (Abs(x) * C_A) + (Abs(y) * C_B).

In this example, the VaR of hedged portfolio can be calculated as: VaR(Unhedged Portfolio P&L + x * Return_A + y * Return_B)

Training Dataset: The training dataset comprises information on various publicly traded stocks. Each stock has metadata and a historical daily returns time series. Here's how you can read this dataset using Python (please note the returns are quoted in percentage terms - so to calculate P&L you'll need to divide the returns data by 100):

import pandas as pd

returns = pd.read_csv("stocks_returns.csv")/100

metadata = pd.read_csv("stocks_metadata.csv")
Data Schema: The dataset is provided in two CSV files:

Stocks Metadata: Contains information about each stock. The file includes columns like:

stock_id: Unique identifier for the stock (string).
stock_exchange: The exchange where the stock is listed (e.g., "NASDAQ", "NYSE" ) (string).
country: Country where the company is based (string).
rating: Credit rating of the company (string).
region: Geographic region where the company operates (string).
sector: Industry sector of the company (e.g., "Technology", "Finance") (string).
market_cap: Market capitalization of the company (string).
capital_cost: Cost of acquiring one unit of stock (numeric).
Dataset link: stocks metadata

Historical Returns: Contains daily historical returns for each stock. The file has the following structure:

date: Date of the return (YYYYMMDD) (date).
stock_1: Daily return for stock with ID 'stock_1' on the given date (numeric).
stock_2: Daily return for stock with ID 'stock_2' on the given date (numeric).
...and so on for all stocks. Each row represents a specific date, and the columns contain the corresponding daily returns for each stock.
Dataset link: stocks returns

Evaluation Criteria: The evaluation criteria will be based on the 95%ile VaR of the hedged portfolio and the total capital cost of implementing the hedge. The goal is to minimize both VaR and cost. A test case is considered passing if both the VaR and the total cost of your hedged portfolio are less than or equal to the VaR and cost values specified in the test case.

You will earn 0.5 point if only the VaR criteria is met and 1 point if both criteria are met. No points are awarded for meeting the cost criteria alone.

Note: This problem statement has additional test cases that'll run post your submission. You final score is subject to change, your current score is not the final score.

Input Format

portfolio_id_1 -5000 -2000 1000 3000 -10000 …

Sample input includes a portfolio identifier portfolio_id_1 and 250 numeric values, each representing the simulated P&L of the unhedged portfolio based on historical returns.

Constraints

-

Output Format

stock_1 1000

stock_2 -2000

The output consists of multiple rows, each indicating the stock_id and the quantity of that stock to include in the hedge. A positive quantity means a long position, and a negative quantity means a short position. For example, if stock_1 has a cost of USD 10 per share and stock_2 has a cost of USD 20 per share, then the total cost of this hedge would be: Total Cost = (1000 * 10) + (2000 * 10) = USD 30,000. Please note there can be multiple possible combinations that can pass the evaluation criteria - the output will be tested to see if the overall VaR and Cost of the portfolio are within required thresholds

Sample Input 0

Portfolio_0 64.519 -27.184 -7.486 -23.389 -59.758 -32.51 10.709 30.091 23.171 8.287 0.685 -39.95 -8.559 -41.953 -35.853 -24.688 -35.236 56.744 79.524 -4.436 31.402 -14.708 -5.235 24.395 18.312 -50.646 -61.176 76.292 51.243 -38.15 32.702 3.362 -72.519 -12.151 13.526 -5.445 11.683 20.39 71.556 16.058 59.782 23.123 63.49 -73.324 14.685 29.85 -14.55 12.929 -42.592 8.43 53.359 -20.301 -53.67 34.028 46.344 -22.197 33.436 8.191 -72.934 -58.64 8.572 -43.509 53.446 73.564 -64.034 0.063 90.673 23.256 -35.493 4.473 -0.408 101.233 2.074 1.881 86.152 13.618 48.136 -28.199 24.667 39.558 4.197 40.523 -20.493 6.912 32.867 -37.858 16.801 -19.829 -16.219 64.059 5.848 22.903 43.011 -2.536 -28.976 -70.6 70.003 44.239 -22.269 -1.339 24.281 -45.394 -35.547 -17.417 -14.146 15.323 -26.936 -29.636 -19.969 -35.087 71.375 4.233 -5.272 -11.89 24.386 -1.989 -18.999 -21.665 -6.07 36.756 -13.734 66.281 -41.605 16.535 -38.988 61.495 19.85 -27.184 -8.556 13.381 -38.734 -15.543 22.72 23.5 51.58 20.624 -26.402 45.586 -91.622 -65.812 -4.672 67.459 -3.269 -1.89 33.056 -23.23 -35.702 121.729 -19.507 -42.295 5.357 -6.842 8.53 78.413 20.732 -11.487 -16.505 -11.483 -2.928 22.009 -17.077 54.633 -22.158 9.628 -22.968 21.34 -17.681 15.522 1.615 -50.879 -37.472 -6.231 -45.596 -9.018 14.151 22.85 -19.619 -53.855 52.707 28.541 -8.261 -79.694 78.373 15.453 32.206 1.269 18.53 1.585 -33.701 -2.377 -8.972 -24.417 -12.871 -2.071 2.126 -78.813 -17.967 -44.014 3.921 12.522 -52.362 -43.616 -25.943 -71.369 -47.09 -21.327 52.426 -64.284 -4.786 -44.108 17.379 -81.296 19.933 -54.466 26.926 -61.874 -10.773 98.22 39.328 -26.459 8.75 3.878 9.435 -45.276 -47.71 -51.774 51.038 28.418 14.255 44.815 -14.741 29.208 -61.269 13.115 -28.616 8.015 27.876 45.914 -47.445 -43.606 2.966 36.819 51.178 19.035 40.15 5.011 27.612 6.392 -27.091 -21.816
Sample Output 0

stock_286 -1000
Sample Input 1

Portfolio_2 -538.72 238.18 128.025 -204.53 -305.72 3.77 217.61 96.445 -391.92 -222.47 746.785 -279.58 -43.335 93.36 -151.925 -1.885 525.9 417.56 223.405 -150.66 25.19 -247.055 -0.18 172.41 114.31 -223.035 -613.89 326.825 -124.405 208.16 -58.82 -139.625 -219.755 6.675 -298.135 11.13 -5.715 -166.475 66.23 194.235 114.44 -21.48 111.985 -387.525 -83.595 163.195 133.46 196.795 -409.04 15.205 754.325 339.95 53.135 69.325 285 73.96 561.92 20.805 -356.57 -185.88 -7.065 -361.25 199.335 -32.44 173.54 125.07 74.515 -183.66 -623.11 290.445 -78.895 85.08 90.12 105.11 105.88 -34.765 353.98 18.185 -229.455 -44.71 -344.08 368.04 -231.125 -206.695 59.27 -340.345 -152.695 -246.265 -73.375 303.5 -19.19 -248.565 110.695 -338.925 -63.19 -306.165 -113.885 67.68 -334.73 -57.455 -33.09 -49.08 102.445 -301.68 -396.385 -138.27 -484.795 -12.905 43.54 86.485 -276.21 276.165 -234.77 33.44 -12.41 56.83 -268.61 89.575 -150.34 -67.17 263.275 193.89 -114.89 -104.66 145.79 134.81 -175.425 -212.735 -373.6 428.135 236.57 -130.685 -187.545 -251.815 93.865 -353.675 117.18 -75.735 325.84 95.025 181.825 180.91 69.495 3.09 181.745 -655.295 -33.06 536.48 263.91 270.34 -152.9 -287.75 -212.31 211.96 -395.78 161.375 248.005 -61.41 365.295 280.52 -217.26 -256.14 -262.045 -225.245 -159.81 597.26 16.065 -592.625 64.505 -485.195 -232.195 171.835 -245.205 -34.53 -15.52 135.35 36.82 29.745 -26.31 -9.84 -56.73 -156.26 394.515 144.675 -314.075 260.25 129.04 -360.405 125.915 137.24 244.195 237.16 -362.265 -111.595 239.495 95 -131.065 -347.055 183.98 148.825 -57.13 187.12 -103.29 -468.99 87.53 -40.695 -25.94 -449.275 68.09 -517.7 467.93 46.43 -289.65 -43.365 265.52 -145.825 92.865 -213.035 375.665 -229.815 248.805 61.825 35.275 -278.25 -74.99 -1.02 -260.515 -221.82 244.485 317.54 -8.845 -394.21 129.465 -66.26 -222.825 179.175 -129.005 449.46 3.765 -244.82 -134.64 -102.01 -248.725 -141.27 114.405 -183.515 -25.13 226.205 -303.79 -226.275
Sample Output 1

stock_120 -5000
stock_280 -5000
Sample Input 2

Portfolio_7 -474.759 -930.042 -158.064 -184.248 95.178 133.95 -453.465 -15.765 210.438 55.95 -172.425 180.804 -103.506 133.791 167.73 205.065 -312.33 36.873 105.81 42.543 62.565 -146.691 23.307 -494.463 -491.202 -233.589 54.735 115.443 109.941 59.991 -318.939 304.662 -142.329 -3.687 11.766 -82.959 2.793 -165.09 -134.637 106.275 43.779 -24.627 -356.307 -56.304 255.003 294.429 59.028 151.989 145.611 235.764 237.459 -311.133 -85.824 158.823 125.19 148.341 -355.884 -479.121 116.82 358.842 247.806 172.425 114.912 301.752 109.683 -28.422 109.203 245.625 140.781 -82.23 -39.447 38.925 206.298 -338.805 -13.152 -337.608 -93.081 -34.596 -124.494 209.208 265.8 -151.971 -167.172 -169.818 91.818 -49.611 -184.179 231.303 -218.79 -86.007 -103.392 -73.074 388.383 454.671 -131.547 -184.959 55.404 -185.112 18.453 46.908 -153.735 453.681 -10.689 154.614 384.564 142.515 74.316 181.092 -487.494 128.835 62.091 -257.439 -105.744 -40.644 -20.946 12.657 -177.318 -44.781 -194.673 -81.873 -227.775 -337.695 -75.609 405.024 -318.591 -45.579 142.692 -293.688 626.238 151.215 -217.449 -128.58 -201.411 -8.508 -92.943 160.14 121.653 133.521 -207.27 -205.815 -58.92 367.59 379.38 -85.944 0.846 122.211 -60.141 110.907 12.087 41.64 -387.264 -93.618 243.825 -362.934 191.733 191.184 143.508 -74.256 -363.681 278.391 324.708 38.889 188.892 101.061 183.909 -239.745 223.116 278.529 454.713 75.219 383.433 -65.415 154.221 374.316 -168.441 25.512 -58.914 -46.188 -163.14 151.185 -75.741 -231.228 115.974 -272.391 191.388 51.057 209.943 309.807 -107.826 -306.915 -48.756 -5.502 215.847 -167.103 432.42 -84.408 79.989 -118.101 453.033 2.232 -32.889 -142.5 -309.552 -285.162 153.297 59.649 -18.861 260.73 -126.126 -377.274 -251.847 -190.257 -40.566 43.389 -76.569 -230.919 -119.457 -168.63 136.53 -35.349 -93.24 -392.82 628.542 -14.418 156 234.312 76.539 -191.043 -84.666 107.619 -268.254 13.548 -196.122 -5.994 114.066 194.334 105.369 261.255 96.828 277.077 -28.914 344.433 -17.133 9.204 -200.727 21.591 18.048 -231.159 -263.742 118.86
Sample Output 2

stock_127 -3000
stock_130 6000
Sample Input 3

Portfolio_15 -143.947242 -54.696053 -194.050843 -114.379762 -250.914222 94.525127 -38.578749 87.924437 71.989914 11.229715 294.581902 -280.166856 164.614297 -65.917684 5.897927 150.181106 263.998383 174.068129 88.060921 -223.012161 266.605458 80.822122 158.287567 -9.023034 22.444253 -33.883753 -261.471284 -91.303422 -83.428574 33.292501 -73.951395 -176.659194 -135.428729 62.436845 -136.566712 -173.370305 -283.001479 107.665112 -53.090612 95.636076 139.235617 -19.10865 85.199601 -209.1056 -310.955306 -2.165499 35.976801 7.50174 73.062716 -203.179266 469.00358 127.236235 3.716109 252.211904 174.502566 6.919277 339.092547 4.422581 5.965394 -173.426425 -30.290419 -185.455906 65.674427 116.629944 -109.198075 78.179203 -115.110464 -177.391446 -119.218134 166.293029 -76.239738 171.411876 -141.504239 139.137473 -25.01957 -20.231675 55.088898 2.876409 -44.997561 -9.161336 -72.560697 283.933358 -34.321216 -135.798794 110.626876 -192.54237 119.526443 -1.670631 27.555595 -40.781872 -235.559946 107.050236 -56.895824 -147.281255 -156.770645 -75.445617 82.642868 265.12137 -196.013666 87.095452 56.390496 -72.333542 -88.827275 8.286346 -170.446357 -148.334462 -280.558408 4.362755 46.400015 -6.847279 91.234663 165.416968 20.210035 145.796151 22.934237 -112.741297 31.012431 -22.119119 68.527919 95.182339 42.33652 140.188056 9.100858 103.606242 180.677244 206.1336 -320.051696 107.214837 -275.824155 101.115325 -11.629589 -118.879814 179.147753 -67.162632 196.397848 47.642698 30.792682 41.573022 44.183258 -16.676944 -247.958614 339.049795 -155.912637 -138.184113 152.162531 -361.782844 -18.957799 239.77323 293.220672 -58.43556 269.09757 121.316655 -226.5609 91.962242 -58.319922 131.039805 207.790177 5.537777 176.764472 110.922365 -199.574635 -235.721358 -267.523291 33.088489 -56.281509 133.278441 -48.862124 -167.373615 19.464542 -160.257798 -149.425146 297.139127 -178.140481 42.800719 -88.711313 -141.499563 -22.497015 -178.339219 89.77678 -76.572525 135.606513 -126.414028 -13.139711 139.544421 -166.822281 164.313625 -98.407028 -264.842666 238.055878 139.604803 -61.628126 -1.396826 -193.065595 -165.36715 -114.776197 -140.0188 -149.45157 62.355782 -143.783067 -9.581666 7.968838 230.406302 -165.656265 -74.449799 -49.687282 -200.097204 51.320385 -169.522404 72.792048 -140.198797 296.193869 -166.100215 -235.234293 -208.255771 114.320455 -32.605264 96.040885 124.16604 109.84381 -100.553086 92.345497 -24.857538 -155.040899 225.620404 -27.926196 126.356083 -262.033456 158.129422 -64.263922 34.925883 78.358089 -16.682323 -80.531757 126.170745 -122.257938 97.482713 4.582632 140.674958 -242.783119 -79.575146 62.659014 -45.79654 30.429683 -416.711845 230.05549 -205.179518 46.296581 155.942441 -128.055691 -95.994111
Sample Output 3

stock_1 -708
stock_12 -101
stock_120 -645
stock_132 -8
stock_15 -634
stock_159 -831
stock_160 -479
stock_182 -62
stock_184 -191
stock_195 -363
stock_211 -898
stock_213 -758
stock_225 -898
stock_247 -475
stock_282 -246
stock_290 -294
stock_5 -115
stock_59 -805
stock_78 -282
stock_98 -358
Sample Input 4

Portfolio_4 -73.081271 -84.211868 8.377751 -33.494525 0.330646 19.475247 47.404192 -26.92179 8.308454 9.586998 33.051007 -31.415246 28.211781 -5.698875 -31.472106 5.658291 27.214006 4.982148 -19.553847 -56.171168 17.279384 36.240806 -3.969569 5.81317 20.925095 24.501942 0.397532 20.571512 -7.335735 -23.525076 1.899284 8.203912 -29.122669 -27.4116 -4.858685 -43.906612 1.575939 -65.111499 -9.143756 24.605195 -39.669326 18.577153 25.6869 -44.695191 10.985917 -33.452485 41.080617 16.576369 1.423955 16.063231 29.559079 4.525803 23.993845 10.135635 31.887597 -17.511973 27.640138 -2.226123 -48.683312 -0.925848 -14.870336 16.236237 1.734242 -5.147925 4.6196 -29.823412 -8.43822 -23.726393 -6.953149 -29.999034 -10.090757 -32.690671 37.713222 -28.07623 -3.584532 -20.549662 21.326194 -19.284161 19.750502 1.189747 -36.447698 -1.052589 -38.709351 -4.423969 -20.205825 -56.211555 25.121753 -17.97476 17.586008 34.181568 -16.893854 -62.648492 -8.73967 -16.388064 27.997725 1.699665 -19.078063 20.283508 -28.101385 -46.618362 -8.272773 -33.829369 -7.528099 -15.521029 -20.974837 -67.451321 2.2753 -29.67227 34.324876 -21.470781 -18.203054 -19.27622 9.930011 47.399165 -2.253188 15.897249 -17.376423 46.265583 -23.478491 -4.962818 -52.207165 -32.302544 81.412873 -9.116124 17.875857 -11.348157 -32.748756 -22.867892 -36.827272 -8.092363 26.076708 10.143215 27.058365 -23.612956 -7.418975 1.366519 25.235845 -15.164643 38.183663 32.795393 -25.904542 27.217621 -45.424937 -77.380392 -8.702781 -2.526311 0.861588 -27.938599 45.455864 25.373931 34.463257 49.054841 11.985695 -4.72214 18.151276 20.906993 -13.332188 -44.148517 53.320829 -15.973302 13.205902 -10.681005 -34.979349 -5.863152 -39.078532 -0.531924 33.774832 -9.122297 44.054664 -42.288186 -37.143983 -5.33027 -33.995735 -24.856274 -10.360636 2.88493 -2.376223 44.747408 -17.480633 -56.126285 -27.282663 9.089984 43.687333 3.972792 29.879047 15.961869 27.911502 -17.617793 28.57794 15.381394 64.113803 -4.344112 -16.25435 1.057378 19.593166 47.570853 25.815816 0.323386 19.50585 1.377868 -11.142047 53.716047 11.971496 -38.769258 42.945247 -43.331965 31.471377 -36.613866 -19.407595 -24.451452 7.886392 -1.606662 -54.666071 -14.426928 24.165189 36.436051 -34.936017 4.968173 1.25402 -28.195535 26.735183 3.656922 -34.970591 18.49586 17.173991 -11.164549 -33.25186 -21.941739 -21.931041 -45.045033 -36.37223 -21.602888 -15.485485 -5.348782 -25.106616 19.26621 45.253593 71.814144 25.437867 -24.323015 19.700412 -23.761719 6.224272 -11.193765 -43.136471 17.409156 6.904642 -2.51 1.911441 -23.239876
Sample Output 4

stock_125: -1500
Sample Input 5

Portfolio_5 -512.001237 -1134.459445 85.501137 -411.361606 -361.82312 44.498178 201.840862 17.098748 322.904941 -16.471364 134.421137 -220.170172 -174.608089 -131.33062 -769.874202 -72.59143 -176.537822 253.203233 660.96373 -36.230896 94.954468 237.042056 22.554502 265.495861 111.517239 -89.008565 -352.946215 330.997123 501.732873 167.419484 -100.384273 -406.603768 -492.097 -156.312756 407.499819 -270.990078 -127.364928 -25.175428 222.976346 85.077008 97.487893 120.689809 184.585963 -431.404018 -5.728746 208.927708 289.288996 244.852596 -214.056288 -159.029945 307.291613 -198.888283 -47.286733 117.850332 163.94208 -257.183546 216.121524 239.537843 -661.248874 -34.143489 -42.851407 -44.430847 -79.339942 263.410097 -17.385747 -122.668944 282.548564 -269.562797 -473.575389 -157.04785 59.639286 362.365274 132.587887 -208.866851 505.567777 -117.968806 -29.803275 -410.001432 -37.574952 -71.119789 -307.500918 45.202618 -513.439238 60.065718 212.175188 -208.181699 -102.716701 61.367039 -94.808264 175.412205 145.377085 362.704725 467.526733 -306.793249 -122.677754 -270.322265 -155.325029 -133.781691 -390.05705 352.892041 -2.422335 -251.606968 -121.990152 -83.08988 491.799048 -109.769907 -316.977428 -932.113794 46.881934 -169.603562 71.15014 361.928829 -320.838309 183.579363 219.635017 156.069269 -77.23208 -404.125259 -54.877664 177.805739 -36.982283 387.815651 309.548005 157.205255 3.623067 277.57912 -87.365498 -472.375519 -434.957216 269.198674 -92.49263 48.591412 147.839694 -126.114745 -79.442517 -54.390256 93.84833 225.799554 124.062728 223.512475 434.039683 109.54129 -78.115292 -304.11428 116.83377 -234.077844 -466.107616 672.396258 231.190399 109.417175 198.097935 -125.80488 125.023035 485.328544 -157.351065 265.948867 136.144295 -291.327897 73.777352 251.061295 93.480994 440.536568 19.13503 98.579012 -226.452972 252.401169 48.447744 160.95807 -41.471462 -518.530877 -362.432142 43.016731 -737.26844 247.909622 289.482897 -176.954519 -160.100252 -20.551075 148.662143 194.975437 -645.948135 -377.68926 734.182727 -18.419962 253.043626 -406.198543 -23.157251 -144.831758 -203.449804 48.988181 14.19586 -423.846527 189.615471 -174.800655 689.141286 -189.245655 -49.237916 -289.088927 226.495579 -16.503798 -130.869592 280.209943 131.42194 -413.04061 -81.173716 106.761972 -101.526236 -75.203949 178.026323 -186.725449 55.410012 -339.271017 -105.288177 148.435182 164.831449 -460.508108 -29.748356 244.188795 185.577381 -177.26761 -264.938306 -166.550465 420.457891 -108.568269 -293.16241 -311.849899 164.561243 -339.620068 310.554041 353.481987 -181.165085 222.263451 -21.826778 1.363298 411.470544 -78.517053 -111.736625 387.159476 -275.77238 -213.885024 197.647415 414.150628 141.82591 -37.594828 422.752865 218.219382 106.129575 17.348787 -698.524584 -430.265881
Sample Output 5

stock_296: -6500
Sample Input 6

Portfolio_8 600.184521 11150.208489000001 815.607081 3057.203721 2616.4944 3862.828401 6843.8889 957.5504 331.3929 -83.621791 1995.8084 -97.307119 801.8009 -43.569856 360.231209 177.855561 2214.379664 9445.290000000001 1028.355281 1931.214761 893.321289 283.611396 -33.479664 1710.077025 1288.158564 1748.742009 5352.936336 461.121344 702.758889 189.748484 1060.901184 242.953361 2867.852484 881.130329 -37.1151 -2.029596 -88.241959 31.974144 -99.916479 5382.217764 114.095424 84.090624 94.463025 3566.181401 0.881936 1530.140625 -43.884919 247.039641 1299.882225 11.830625 2349.458064 16.834481 281.967936 826.167489 5243.756201 1335.424769 476.624169 207.932304 4270.796544 491.656976 1980.910689 -10.148559 -41.660956 2803.162161 -41.553975 134.457344 1892.997449 42.683025 864.972096 -58.796439 546.939225 14836.506224999999 -80.560719 279.431441 257.172201 2724.072164 2417.229584 -50.057511 -62.472124 -57.541744 -99.728559 2397.700529 2511.925449 574.285089 -87.896559 369.068964 3128.398761 1917.8064 -98.439999 554.182929 -98.182896 1104.922944 1877.669841 2740.250436 360.789156 398.003856 327.827856 -28.343775 5707.506849 351.605001 174.498624 546.990096 2917.524624 -99.91 5207.851025 -73.539264 794.6081 405.620196 2345.005809 499.711121 2181.495225 504.1764 -71.398896 554.745744 3095.301729 496.189889 371.975625 -15.599031 -67.121244 267.527241 794.01 190.600209 81.818256 99.911321 319.266576 1460.25 -92.897775 266.837409 4317.197444 -97.339839 18935.996841 780.664976 56.375025 -95.053824 140.467049 100.703889 227.935881 63.405089 -77.674375 2064.203441 -97.180959 2315.7225 7659.8481 1620.839289 2124.537225 -99.494479 1886.395761 3515.496641 3140.227929 3240.377616 518.5169 229.531409 278.3025 1163.944704 23.076836 2508.962084 -89.103399 1079.853801 455.262096 436.987929 -11.470719 2203.520025 4336.359236 -97.853775 295.134884 883.888689 -96.5775 3503.6009 251.900081 -92.736975 2382.630276 172.547081 3529.098564 1878.114576 1880.517009 -78.414684 423.4944 -94.589724 933.108164 140.498064 -80.666391 1896.391761 3957.053025 3402.390761 -66.5916 -60.874975 190.975364 1570.111689 837.400689 351.052644 216.1284 75.907169 1288.829289 373.193009 705.821769 796.822809 -7.340124 -95.903424 246.406544 -88.676775 1218.270864 911.558025 523.900484 -60.824919 3800.252304 21.330225 749.197881 61.645796 2782.723481 414.473124 2293.655625 -64.466479 1818.002025 305.780736 755.913536 705.083876 511.721289 223.856016 321.152484 1547.5481 498.878784 585.601856 -13.621564 -47.654775 606.336929 1618.185401 -31.788919 376.592561 38.650625 1780.436496 -57.437424 604.796304 903.178929 546.278084 1145.3841 1069.776804 554.080625 66.255236 1140.4484 2448.533289 323.001489 -58.539279 -74.0919 3340.174409 1410.565956 1342.784256 -99.8064 2727.261584 2142.211904 1578.868676
Sample Output 6

stock_239: -20000 
Sample Input 7

Portfolio_9 -341.461528 -852.774922 4.064719 2.782886 4.157859 -178.963582 -98.990801 -66.915281 -53.266738 4.850701 -12.643811 -12.19956 -60.192994 -110.574704 3.705104 -309.059826 -1.84216 -2.979634 -102.374414 -38.123971 4.914064 -12.938074 -189.270144 -1.373205 -54.989552 -8.864455 1.819171 -26.180039 -309.908416 -422.470318 -41.424092 -75.512819 -53.725072 -291.257531 -8.94784 -189.295078 -256.885319 -3.734454 -21.03762 -447.286094 -114.643644 -73.526845 4.943182 -255.230622 4.908063 -0.441331 -5.28178 1.899219 4.897755 3.255958 2.615358 -74.40511 0.092568 -4.31476 -28.613059 -119.335884 4.084508 -100.267073 -38.179788 -4.863101 -3.999453 -276.430029 -31.660904 -63.342742 -66.2418 -190.22501 -26.402686 -208.22409 -145.212286 -5.161677 -6.17662 -12.219968 -13.089922 0.746814 -184.875869 -222.522342 4.586431 -95.36096 0.154783 -39.166892 -9.031475 -46.107242 -130.231202 -26.8756 -49.16657 2.048576 -24.821704 1.195464 -183.848142 -18.65095 0.068762 -104.661811 -42.139851 -9.867777 -119.01694 -2.600212 3.795086 -153.816844 -15.959234 -29.510899 -268.20832 2.24718 -164.524999 2.25163 -8.195001 -210.102405 -74.297549 -631.327367 -5.579306 2.96478 4.60238 -128.12284 -4.94473 -40.48128 -15.379786 4.882035 -387.781571 -25.978094 -35.578307 -185.387692 -920.004011 -3.84583 -20.90088 -42.311956 -58.988954 -73.59423 -193.059592 -22.563172 -55.994766 -2.470086 4.13763 -362.318776 -27.0045 -2.010464 0.710768 -157.535511 4.999048 -34.886177 -8.185256 -3.590983 -27.808973 -94.788314 -141.497432 -130.772605 2.381292 -36.308002 4.997536 -8.647172 -35.863687 -21.68743 -164.35618 -71.82416 -78.183047 3.711204 -20.866751 4.217503 -7.691431 -2.818751 -20.62848 -15.30717 -7.526279 -20.456705 -456.7605 -144.019782 -23.828807 -1.221701 -14.516952 -48.46123 -28.486132 -9.992193 1.487104 -29.945748 -197.680711 -68.137826 -78.828746 3.080479 -131.409091 -61.630751 3.554394 -122.456956 -79.091005 -13.587848 -133.217046 -150.040961 -167.86612 -208.642234 -34.643648 -13.765656 -356.462531 3.174108 -56.81128 -63.298384 4.978154 -230.531298 -110.041654 -208.936487 -24.921918 0.973366 -40.228289 -21.932884 -32.708072 -154.567303 -27.693802 -2.81625 -1.74425 -274.310854 -105.82632 1.183242 -31.701532 2.758849 -188.790477 -120.926711 -201.859552 -56.846445 4.367743 3.696438 -9.812647 -42.204354 -64.419106 -216.671453 -2.015201 2.003846 -6.386914 -84.985546 -9.752466 2.019308 4.828505 -133.627451 -22.216778 -47.861261 4.043844 -197.127178 -10.986508 -118.907398 -9.945476 -13.870416 -0.483139 -521.379041 0.9482 -30.855742 -590.195051 2.45102 -262.92664 -241.26162 -62.653133 -6.035037 4.990926 -477.240563 -20.519923 -16.01045
Sample Output 7

stock_84: 1000 

                                                           Automated Market Making
                                                                Problem set -2 
                                                                
Overview: You’ll build an adaptive quoting strategy that mimics a market-maker in a volatile, real‑time environment. Every timestamp, your code posts:

• bid_price
• ask_price

based on the L2 order book, recent public trades, and your current inventory. Maximize total PnL while keeping inventory within ±20 units.

Data Provided:

orderbook_train.csv (training) - Link

• timestamped L2 snapshots: bid/ask prices & sizes at multiple levels.
• Columns: timestamp, bid_price_1, bid_size_1, …, ask_price_2, ask_size_2, etc.

public_trades.csv (training) - Link

• timestamp, price, size, side (buy/sell)

orderbook_test.csv

public_trades_test.csv

Tick Size: All quotes must be multiples of 0.1. Any quote not on this grid is invalid and will incur hidden penalties.

PLEASE NOTE - PICK ONLY TOP 3000 rows from all datasets for submission

Submission Files:

submission.csv
– Rows only at timestamps when you post or update a quote.
– Columns: timestamp, bid_price, ask_price
– Example: 0, 100.0, 100.5
Update strategy function in class AutomatedMarketMaking
Detailed Flow: At each timestamp t:

Inventory is known from all fills up to t (inclusive).
You call strategy(…), passing current inventory, order book & trades up to t, and TICK_SIZE.
You output a new quote for t (must satisfy bid \< ask, and multiples of tick size).
– If you don’t post a quote at t, your previous quote remains active.
– Any time your inventory changes or a fill occurs at t, you must send a new quote on that side at t, else that side will deactivate.
The simulator applies any fills at t+1 (based on the quote you posted at t) to update inventory.
Fills are simulated based on public trades
Inventory Risk & Rules: – Keep |inventory| ≤ 20
– bid_price < ask_price
– Quotes active from t+1 to next update or end of test
– If your inventory changes at t, update that side in your quote at t, or that side deactivates

Please note that at time T, all the quotes should be based on time from 0 to T-1, any attempts at using future data (coming up for a quote at T using data from T+1 to N) will result in submission failures with error "future-peek detected"
Evaluation Criteria:

PnL Score: realized PnL, unrealized PnL (mark-to-market), and penalties.
PnL = realized PnL + unrealized PnL - penalties, where unrealized PnL is calculated using the last mid from order book
Risk Management:
Quotes crossing the market (bid >= ask) incur penalties.
Inventory beyond the limit of ±20 units incurs penalties.
Input Format

Orderbook Snapshot (L2)

Timestamp	Bid Price 1	Bid Size 1	Bid Price 2	Bid Size 2	Ask Price 1	Ask Size 1	Ask Price 2	Ask Size 2
1738368005000000	99.8	10	99.7	20	100.2	15	100.3	25
1738368006000000	99.9	12	99.8	30	100.3	18	100.4	20
1738368007000000	100.0	20	99.9	30	100.4	20	100.5	25
Public Trades

Timestamp	Price	Size	Side
1738368005000000	100.3	5	buy
1738368006000000	99.9	10	sell
1738368007000000	100.4	6	buy
Constraints

As a market maker you can hold/short a maximum of 20 units.

Output Format

Timestamp	Bid Price	Ask Price
1738368005000000	99.9	100.4
1738368006000000	100.0	100.5
Sample Input 0

orderbook_train.csv
public_trades_train.csv

                                             Exotic Option Pricing using Monte Carlo Simulation
                                             Problem Statement 3

Imagine you're a quantitative analyst on an Equity Volatility trading desk at a major investment bank. Your client, a large hedge fund, has requested price quotes for custom European basket options with knock-out features. The basket is created on three hypothetical assets: DummyTechCorp (DTC), DummyFinCorp(DFC) and DummyEnergyCorp(DEC).

You are now tasked with building this pricing model and providing prices for all the options. Your submission will be rated based on the aggregate accuracy of the submitted prices.

Background:

Vanilla Options

An option is a financial contract that gives the buyer the right, but not the obligation, to buy or sell an asset at a specified price (called the strike price) on or before a specified date (called the maturity). Options are widely used for hedging, speculation, and portfolio management. We concern ourselves only with 'European Options' in this challenge. European options can only be exercised on the specified maturity data and not before.

Generally, there are two types of European vanilla options.

Call Options:

A call option gives the holder the right to buy the underlying stock at the strike price K on the maturity date.
The payoff of a call option is given by max⁡(S(t) – K,0) where S(t) is the price of underlying stock on maturity date.
Put Options:

A put option gives the holder the right to sell the underlying asset at the strike price K.
The payoff of a put option is given by max⁡(K - S(t),0) where S(t) is the price of underlying stock on maturity date.
Basket Options

Basket options are comparable to the vanilla options but their payoff depends on the weighted average price of a collection of assets (the "basket"). They offer investors a way to gain exposure to a diversified portfolio with a single transaction. However, pricing these options accurately presents a significant challenge and require more sophisticated techniques to account for the correlations between the assets in the basket with complex real-world diffusions

Knock-Out Call & Put Options

Options can be made more investor friendly by adding a knock-out feature which gives the holder the right to buy(sell) the underlying asset at a specified strike price ( K ), but only if the price of the underlying asset does not breach a predefined barrier level ( B ) during the life of the option. If the barrier is breached, the option becomes void (or "knocks out"), and the holder receives no payoff

The payoff of the knock-out call option is:

max⁡(S(T) – K,0) when S(t) < B for all t ≤ T
0 if S(t) >= B at any time t ≤ T
The payoff of the knock-out put option is:

max⁡(K - S(T),0) when S(t) < B for all t ≤ T
0 if S(t) >= B at any time t ≤ T
Model Assumptions:

Your desk applies the following assumptions in pricing Exotic Equity Options

Each stock's price evolves over time in a way that is influenced by randomness, similar to how stock prices fluctuate in real life. This randomness is modeled using a Diffusion Process with a drift component and a random shock component which also accounts for volatility of the stock.

The spot price for each stock evolves according to the following stochastic differential equation ( SDE ) under the risk-neutral probability measure. The drift for all stocks is the same but each stock can have a different local volatility function under the risk-neutral probability measure:

dS(t)= r * S(t) * dt+ σ(S(t), t) * dW(t)
where, - r = Risk-free rate - S(t) = Stock price at time t - σ(S(t),t) = The local volatility function which depends on S(t) and t - dW(t) = Brownian motion increment for at time t

The stocks are correlated. This dependance is usually measured with their historical return correlation. The Brownian Motion shocks for all 3 stocks should account for this correlation provided.

Monte Carlo simulation based pricing models work by simulating a large number of possible price paths for the underlying assets based on their diffusion processes, evaluating the payoff on each path and then averaging the discounted payoffs of the option across all simulated paths.

σ(S(t),t) is calibrated to ensure that the Monte Carlo Pricing Pricing Model ( with stock correlations captured ) is able to match Vanilla European Options provided.

σ(S(t),t) is assumed to be piece-wise constant in time for each calibration option maturity date and piece-wise constant for each calibration option strike ‘K’ for any time t. The calibration options for this challenge use 5 strikes and 3 maturity dates for each stock. So, the local volatility function for each stock can be thought of as a 5 * 3 matrix.

Diffused Stock prices for all the assets in the basket are generated post calibration. These paths can be then used to obtain the Diffused Basket prices and evaluate the basked option prices.

Market inputs:

All 3 stocks have a current market spot value of $100
The risk-free rate to be used in 5%
The pairwise correlations for 3 stocks is given
Stock1,Stock2,Correlation
DTC,DFC,0.75
DTC,DEC,0.5,
DFC,DEC,0.25
To calibrate the Local Volatility model for each asset, you are given the prices of European call options (in $) for different strikes and maturities. Each row below corresponds to a different calibration option.
CalibIdx,Stock,Type,Strike ,Maturity,Price
1,DTC,Call,50,1y,52.44
2,DTC,Call,50,2y,54.77
3,DTC,Call,50,5y,61.23
4,DTC,Call,75,1y,28.97
5,DTC,Call,75,2y,33.04
6,DTC,Call,75,5y,43.47
7,DTC,Call,100,1y,10.45
8,DTC,Call,100,2y,16.13
9,DTC,Call,100,5y,29.14
10,DTC,Call,125,1y,2.32
11,DTC,Call,125,2y,6.54
12,DTC,Call,125,5y,18.82
13,DTC,Call,150,1y,0.36
14,DTC,Call,150,2y,2.34
15,DTC,Call,150,5y,11.89
16,DFC,Call,50,1y,52.45
17,DFC,Call,50,2y,54.9
18,DFC,Call,50,5y,61.87
19,DFC,Call,75,1y,29.11
20,DFC,Call,75,2y,33.34
21,DFC,Call,75,5y,43.99
22,DFC,Call,100,1y,10.45
23,DFC,Call,100,2y,16.13
24,DFC,Call,100,5y,29.14
25,DFC,Call,125,1y,2.8
26,DFC,Call,125,2y,7.39
27,DFC,Call,125,5y,20.15
28,DFC,Call,150,1y,1.26
29,DFC,Call,150,2y,4.94
30,DFC,Call,150,5y,17.46
31,DEC,Call,50,1y,52.44
32,DEC,Call,50,2y,54.8
33,DEC,Call,50,5y,61.42
34,DEC,Call,75,1y,29.08
35,DEC,Call,75,2y,33.28
36,DEC,Call,75,5y,43.88
37,DEC,Call,100,1y,10.45
38,DEC,Call,100,2y,16.13
39,DEC,Call,100,5y,29.14
40,DEC,Call,125,1y,1.96
41,DEC,Call,125,2y,5.87
42,DEC,Call,125,5y,17.74
43,DEC,Call,150,1y,0.16
44,DEC,Call,150,2y,1.49
45,DEC,Call,150,5y,9.7
Evaluation Criteria:

The client has provided the following basket options for you to price.Each row below corresponds to a different option you are supposed to price. Your responses will be rated based on aggregate accuracy across all options.

The Basket Underlier for all these options is the equal weighted portfolio of DTC, DFC and DEC stocks. Weights of each stock = 1/3 such that the Basket spot price = $100.

Knock out feature is up and out only.

Id,Asset,KnockOut,Maturity,Strike,Type
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
36,Basket,200,5y,125,Put
Key things you should consider to improve your submission accuracy are:

Ensure the Local Volatility parameterization is as mentioned above. Different Local Volatility parameterizations can lead to different answers and will impact your score.

Ensure pricing error on the calibration options is minimal. Pricing an exotic with poorly calibrated SDEs will lead to erroneous results.

Ensure Monte-Carlo simulation is converging by using sufficiently large Monte-Carlo Paths and fine discretization in time dimension

Input Format

Your can load the Basket Option details by reading the text provided above

import pandas as pd
import io

data = '''Id,Asset,KnockOut,Maturity,Strike,Type
DummyId1,Basket,150,2y,50,Call
DummyId2,Basket,175,2y,50,Put'''

data_input = io.StringIO(data)

data_df = pd.read_csv(data_input,sep = ',')
Constraints

Option Price should be non-negative
Output Format

Your submission should print the result in the following format.

Id,Price
DummyId1,20
DummyId2,50
Even though you may use some cached data to optimize runtime, please provide an option in the code to regenerate the cached data for evaluation purposes.. Submissions which produce some result but incomplete code to generate the output will not be considered

Sample Input 0

Id,Asset,KnockOut,Maturity,Strike,Type
1,Basket,150,2y,50,Call
2,Basket,175,2y,50,Put
Sample Output 0

Id,Price
1,100
2,100
Explanation 0

Check that Output Format is as expected
