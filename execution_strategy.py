import numpy as np
import pdb

def execution(full_test_data, full_test_labels, max_holdings=100, unit=1, tick_increment=0.01):
	cash = 0
	position = 0
	d = {}
	sell_orders = []
	i = 0
	# TO-DO check quantity match
	while i < len(full_test_labels)-1:
		if full_test_labels[i] == 1 and position < max_holdings:
			# buy at best ask
			cash = cash - unit * full_test_data[i][0] 
			position = position + unit
			if full_test_data[i][0] + tick_increment not in d:
				d[full_test_data[i][0] + tick_increment] = unit
			else:
				d[full_test_data[i][0] + tick_increment] = d[full_test_data[i][0] + tick_increment] + unit
			sell_orders = sorted(d.keys())
			print("Buy at price {} for {} units. Current cash is {} and current holding is {}".format(full_test_data[i][0], unit, cash, position))
		elif full_test_labels[i] == 0 and position > 0:
			# sell the bought units
			for price in sell_orders:
				if full_test_data[i][2] >= price:
					shares = d[price]
					cash = cash + d[price] * full_test_data[i][2]
					position = position - d[price]
					del d[price]
					print(d)
					print("Sell at price {} for {} units. Current cash is {} and current holding is {}".format(full_test_data[i][2], shares, cash, position))
				else:
					break
			sell_orders = sorted(d.keys())	
		i = i+1
	# liquidate in the end
	cash = cash + position * full_test_data[i][2]
	print("Liquidate all at price {} for {} units. Current cash is {}".format(full_test_data[i][2], position, cash))
	return cash