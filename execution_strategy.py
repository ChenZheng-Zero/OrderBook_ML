import numpy as np
import pandas as pd
from utils import time_to_int
import pdb

def detect_buy(value):
	buy_orders = pd.DataFrame(columns=value.columns)
	pdb.set_trace()



def execution(trd_filename, full_test_data, full_test_timestamp, full_test_labels, max_holdings=100, unit=1, tick_increment=0.01):
	# read-in the RMD trade data
	trades = pd.read_excel(trd_filename).sort_values('Time').reset_index()
	#pdb.set_trace()
	cash = 0
	position = 0
	d = {}
	sell_orders = []
	i = 0
	trd_i = 0
	# TO-DO check quantity match
	while i < len(full_test_labels)-1:
		# buy at best ask if predict price up
		if full_test_labels[i] == 1 and position < max_holdings:
			quant_less = 0
			for price in sell_orders:
				# prevent from sell low buy high
				if price <= full_test_data[i][0]:
					quant_less = quant_less + d[price]

			#pdb.set_trace()
			if quant_less == 0:
				cash = cash - unit * full_test_data[i][0] 
				position = position + unit
				if full_test_data[i][0] + tick_increment not in d:
					d[full_test_data[i][0] + tick_increment] = unit
				else:
					d[full_test_data[i][0] + tick_increment] = d[full_test_data[i][0] + tick_increment] + unit
				sell_orders = sorted(d.keys())
				print("Buy at price {} for {} units at time {}. Current cash is {} and current holding is {}".format(full_test_data[i][0], unit, full_test_timestamp[i], format(cash, '.3f'), position))
		# want to sell with a positive position
		if position > 0:
			# sell the bought units (transact with the trd_file)
			#pdb.set_trace()
			while trd_i < trades.shape[0]-1 and time_to_int(trades['Time'][trd_i]) < full_test_timestamp[i]:
				trd_i = trd_i+1
			while trd_i < trades.shape[0]-1 and time_to_int(trades['Time'][trd_i]) == full_test_timestamp[i]:
				if trades['BUY_SELL_FLAG'][trd_i] == 0:
					for price in sell_orders:
						if trades['PRICE'][trd_i] >= price:
							shares = d[price]
							cash = cash + shares * price
							position = position - shares
							print(d)
							del d[price]
							print(d)
							print("Sell at price {} for {} units from trades at time {}. Current cash is {} and current holding is {}".format(price, shares, trades['Time'][trd_i], format(cash, '.3f'), position))
						else:
							break
					sell_orders = sorted(d.keys())	
				trd_i = trd_i+1

			# sell the bought units (transact with the OB)
			for price in sell_orders:
				if full_test_data[i][2] >= price:
					shares = d[price]
					cash = cash + shares * price
					position = position - shares
					print(d)
					del d[price]
					print(d)
					print("Sell at price {} for {} units. Current cash is {} and current holding is {}".format(price, shares, format(cash, '.3f'), position))
				else:
					break
			sell_orders = sorted(d.keys())	
		i = i+1
	# liquidate in the end
	cash = cash + position * full_test_data[i][2]
	print("Liquidate all at price {} for {} units. Current cash is {}".format(full_test_data[i][2], position, format(cash, '.3f')))
	return cash