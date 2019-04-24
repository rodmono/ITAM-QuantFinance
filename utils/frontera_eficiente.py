#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################
#-------------------------------------#
# Module: Frontera Eficiente          #
#-------------------------------------#
# Creado:                             #
#     20. 04. 2019                    #
# Ult. modificacion:                  #
#     23. 04. 2019                    #
#-------------------------------------#
# Autor: Rodrigo Lugo Frias           #
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#######################################
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(font_scale=1.5)
import datetime as dt
import matplotlib.pylab as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os
import pywt
from statsmodels.robust import mad
import statsmodels.formula.api as sm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import scipy.optimize as sco
#########################################
#---------------------------------------#
# Aux. functions
#---------------------------------------#
#########################################
def alpha_0( Num_Days ):
    return 2./(Num_Days + 1.0)

def STDerror( m, b, sdata ):
	time = [t for t in range(0,len(sdata))]
	x    = [(m*t + b) for t in time]
	mt   = np.mean(time)
	num_slope  = 0
	den_slope  = 0
	for i in range(0,len(sdata)):
		num_slope += (sdata[i] - x[i])**2
		den_slope += (i - mt)**2
	num_slope = np.sqrt(num_slope/(len(sdata)-2))
	den1= np.sqrt(den_slope)
	den2 = np.sqrt(len(sdata)*den_slope)
	return [num_slope/den1, num_slope/den2]

def Slope(x1, y1, x2, y2):
    slope = (y2-y1)/(x2-x1)
    return slope

def YInt(x1, y1, x2, y2):
    m = Slope(x1, y1, x2, y2)
    return  y1 - m*x1
#########################################
# END: Aux. functions     				#
#########################################

#########################################
#---------------------------------------#
# getData 							    #
#---------------------------------------#
#########################################
class getData:
	def __init__( self, file ):
		self.file = file
		# ----- #
		df = pd.read_csv(self.file, index_col = 0)
		df = self.index_to_datetime(df)
		self.n = 22 # Days to ATR
		# ----- #
		self.timeseries = df
		self.truerange  = self.truerange()
		self.atr 	    = self.atr()
		self.atr_return = self.atr_return()
		self.cum_sum    = self.cum_sum()
		self.dataframe	= self.dataframe()

	def index_to_datetime( self, df ):
		df.index = df.index.astype('str')
		df.index = df.index.to_datetime()
		return df

	def truerange( self ):
		adf = self.timeseries
		s1 = pd.Series(np.abs(adf.DHigh - adf.DLow))
		s2 = pd.Series(np.abs(adf.DHigh - adf.DClose.shift()))
		s3 = pd.Series(np.abs(adf.DLow  - adf.DClose.shift()))
		TR = pd.Series(pd.concat([s1,s2,s3],axis=1).max(axis=1), name = 'TrueRange')
		return TR

	def atr( self ):
		n = self.n
		TR = self.truerange
		ATR = pd.Series(pd.ewma(TR, span = n, min_periods = n), name = 'ATR_{}'.format(n))
		return ATR

	def atr_return( self ):
		tday    = self.timeseries.DClose
		yday    = self.timeseries.DClose.shift()
		atryday = self.atr.shift()
		atr_ret = (tday - yday) / atryday
		atr_ret = atr_ret.rename('ATR_RET')
		return atr_ret

	def cum_sum( self ):
		atr_ret = self.atr_return
		cum_sum = atr_ret.cumsum(axis = 0)
		cum_sum = cum_sum.rename('PATR')
		return cum_sum

	def dataframe( self ):
		cols =  ['DOpen', 'DHigh', 'DLow', 'DClose', 'TrueRange', 'ATR_{}'.format(22)]
		cols += ['ATR_RET', 'PATR']
		adf = self.timeseries.join([self.truerange,self.atr,self.atr_return,self.cum_sum])
		adf = adf[cols]
		return adf

	def plot( self, Series, *args):
		fig, ax = plt.subplots(1,figsize=(10, 7))
		ser = self.dataframe[Series]
		ser.plot()
		plt.xlabel('Year')
		plt.ylabel(Series)
		if len(args) != 0:
			plt.title(args[0])
		plt.show()
#########################################
# END: getData  						#
#########################################

#########################################
#---------------------------------------#
# Regression						    #
#---------------------------------------#
#########################################
class Regression:
	def __init__( self, data ):
		self.time   = range(0,len(data))
		self.data   = data
		self.simple = self.SimpleRegression(self.time, self.data)

	def Results( self ):
		txts = 'Simple:\n\tSlope: {0:.5f}\tIntercept: {1:.5f}\n'.format(self.simple.slope, self.simple.intercept)
		txts += '\tSSE: {0:.5f}\tISE: {1:.5f}\n\t'.format(self.simple.sse, self.simple.ise)
		print ( txts )

	class SimpleRegression:
		def __init__(self, time, data):
			X = data
			y = [t for t in range(0,len(data))]
			df = pd.concat([pd.Series(y,index=X.index,name='time'),X],axis=1)
			model = sm.ols(formula='time ~ PATR', data=df)
			result = model.fit()
			self.slope = result.params[1]
			self.intercept = result.params[0]
			self.sse = STDerror(self.slope, self.intercept, data)[0] # Compared to the initial data
			self.ise = STDerror(self.slope, self.intercept, data)[1] # Compared to the initial data

#########################################
# END: Regression  						#
#########################################

#########################################
#---------------------------------------#
# RegressionML						    #
#---------------------------------------#
#########################################
class RegressionML:
	def __init__( self, data ):
		self.time   = range(0,len(data))
		self.data   = data
		self.model  = linear_model.LinearRegression()
		self.simple = self.SimpleRegression(self.model, self.time, self.data)

	def Results( self ):
		txts = 'Simple Regression:\n\tSlope: {0:.5f}\tIntercept: {1:.5f}\n'.format(self.simple.slope, self.simple.intercept)
		txts += '\tSSE: {0:.5f}\tISE: {1:.5f}\n\t'.format(self.simple.sse, self.simple.ise)
		print ( txts )

	def Plot( self, *args ):
		fig, ax1 = plt.subplots(1,figsize=(10, 7))
		ax1.plot(self.data,linestyle='-.',color='g',label='ATR Return (cumsum)')
		ti = self.data.index[0]
		tf = self.data.index[-1]
		if len(args) == 0:
			plt.xticks(rotation=30)
			plt.legend()
			plt.show()
		else:
			if args[0] == 's':
				yi = self.simple.intercept
				yf = self.simple.slope*(len(self.data)) + self.simple.intercept
				ax1.plot([ti,tf],[yi,yf],color='r', label = 'Simple Regression')
				plt.xticks(rotation=30)
				plt.legend()
				plt.show()

	class SimpleRegression:
		def __init__(self, model, time, data):
			t = time
			x = data
			X_train, X_test, y_train, y_test = train_test_split(t, x, test_size=0., random_state=1)
			X_train = [[i] for i in  X_train]
			model.fit(X_train,y_train)
			self.slope = model.coef_[0]
			self.intercept = model.intercept_
			self.sse = STDerror(self.slope, self.intercept, data)[0] # Compared to the initial data
			self.ise = STDerror(self.slope, self.intercept, data)[1] # Compared to the initial data
#########################################
# END: RegressionML						#
#########################################

#######################################
#-------------------------------------#
# Portfolio							  #
#-------------------------------------#
#######################################
def portfolio( weights, mean_ret, cov_mat, riskfreerate):
    mu = mean_ret.dot(weights)*250
    sigma = np.sqrt(weights.dot(cov_mat.dot(weights)))*np.sqrt(250)
    sharpe = (mu-riskfreerate)/sigma
    return mu, sigma, sharpe # Expected value, Volatility, Sharpe ratio

def getReturn( weights, mean_ret, cov_mat, riskfreerate):
    return portfolio(weights,mean_ret,cov_mat,riskfreerate)[0]

def getVolatility( weights, mean_ret, cov_mat, riskfreerate):
    return portfolio(weights,mean_ret,cov_mat,riskfreerate)[1]

def negSharpeRatio( weights, mean_ret, cov_mat, riskfreerate):
    return -portfolio(weights,mean_ret,cov_mat,riskfreerate)[2]

def random_weights(n):
    k = np.random.random(n)
    return k / sum(k)
#######################################
#-------------------------------------#
#######################################

#######################################
#-------------------------------------#
# Simulation						  #
#-------------------------------------#
#######################################
class simulation:
	def __init__( self, stocks, data, riskfreerate, n_portfolios ):
		self.stocks     = stocks
		self.rfr        = riskfreerate
		self.data	    = data
		self.n_portfolios = n_portfolios
		self.returns    = data.pct_change()[1:]
		self.mean_ret   = self.returns.mean()
		self.cov_mat    = self.returns.cov()
		self.simulation = self.do_simulation()
		self.results    = self.simulation[0]
		self.max_sharpe_portfolio = self.simulation[1]
		self.min_volatility_portfolio = self.simulation[2]

	def do_simulation( self ):
		means,stdvs,shrps,weights = [],[],[],[]
		for i in range(self.n_portfolios):
			w = random_weights(len(self.stocks))
			p = portfolio(w,self.mean_ret,self.cov_mat,self.rfr)
			means.append(p[0])
			stdvs.append(p[1])
			shrps.append(p[2])
			weights.append(w)
		# Convert to DataFrame
		wght = {}
		for i in range(len(self.stocks)):
			wght[self.stocks[i]] = [j[i] for j in weights]
		aux = {'Returns': means,'Volatility': stdvs,'Sharpe Ratio': shrps}
		results = {**wght, **aux}
		df = pd.DataFrame.from_dict(results)
		max_sha_port = df.iloc[df['Sharpe Ratio'].idxmax()]
		min_vol_port    = df.iloc[df['Volatility'].idxmin()]
		return df, max_sha_port, min_vol_port

	def plot( self ):
		df = self.simulation[0]
		max_sh = self.simulation[1]
		min_vol= self.simulation[2]
		# Scatter plot colored by Sharpe Ratio
		plt.style.use('seaborn-dark')
		fig, ax = plt.subplots(figsize=(10,7))
		df.plot(ax= ax, kind='scatter',x='Volatility', y='Returns', c='Sharpe Ratio', cmap='RdYlGn', edgecolors='black', grid=True, label = 'MC Simulation')
		# Maximum Sharpe Ratio
		ax.scatter(x=max_sh['Volatility'],y=max_sh['Returns'],marker='D',c='r',s=100,label='Maximum Sharpe Ratio')
		# Minimum variance
		ax.scatter(x=min_vol['Volatility'],y=min_vol['Returns'],marker='D',c='b',s=100,label='Minimum Volatility')
		plt.legend()
		ax.set_xlabel('Volatility (Std. Deviation)', fontsize=15)
		ax.set_ylabel('Expected Returns', fontsize=15)
		ax.set_title('Efficient Frontier', fontsize=22)
		plt.show()

	def print( self ):
		max_sh = self.simulation[1]
		min_vol= self.simulation[2]
		print('Maximum Sharpe Ratio:\n{}'.format(
					max_sh[max_sh.index.tolist()[0:len(self.stocks)]].to_frame(name='Weights').T))
		print('{}'.format(max_sh[max_sh.index.tolist()[len(self.stocks):]].to_frame(name='Results').T))
		print('\nMinimum Volatility:\n{}'.format(
					min_vol[min_vol.index.tolist()[0:len(self.stocks)]].to_frame(name='Weights').T))
		print('{}'.format(min_vol[min_vol.index.tolist()[len(self.stocks):]].to_frame(name='Results').T))
#######################################
# END: Simulation					  #
#######################################

#######################################
#-------------------------------------#
# Theory							  #
#-------------------------------------#
#######################################
def MaxSharpeRatio(meanReturns, covMatrix, riskFreeRate):
	numAssets = len(meanReturns)
	args = (meanReturns, covMatrix, riskFreeRate)
	constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
	bounds = tuple( (0,1) for asset in range(numAssets))
	opts = sco.minimize(negSharpeRatio, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
	return opts

def MinVolatility(meanReturns, covMatrix, riskFreeRate):
	numAssets = len(meanReturns)
	args = (meanReturns, covMatrix, riskFreeRate)
	constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
	bounds = tuple( (0,1) for asset in range(numAssets))
	opts = sco.minimize(getVolatility, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
	return opts

def EfficientReturn(meanReturns, covMatrix, riskFreeRate, targetReturn):
	numAssets = len(meanReturns)
	args = (meanReturns, covMatrix, riskFreeRate)
	def getPortfolioReturn(weights):
		return portfolio(weights, meanReturns, covMatrix, riskFreeRate)[0]
	constraints = ({'type': 'eq', 'fun': lambda x: getPortfolioReturn(x) - targetReturn},
					{'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
	bounds = tuple((0,1) for asset in range(numAssets))
	opts = sco.minimize(getVolatility, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
	return opts

def EfficientFrontier(meanReturns, covMatrix, riskFreeRate, rangeOfReturns):
	efficientPortfolios = []
	for ret in rangeOfReturns:
		efficientPortfolios.append(EfficientReturn(meanReturns, covMatrix, riskFreeRate, ret))
	return efficientPortfolios

class theory:
	def __init__( self, stocks, data, riskfreerate, n_portfolios ):
		self.stocks     = stocks
		self.rfr        = riskfreerate
		self.data	    = data
		self.n_portfolios = n_portfolios
		self.returns    = data.pct_change()[1:]
		self.mean_ret   = self.returns.mean()
		self.cov_mat    = self.returns.cov()
		self.theory     = self.do_theory()
		self.results    = self.theory[0]
		self.max_sharpe_portfolio = self.theory[1]
		self.min_volatility_portfolio = self.theory[2]

	def do_theory( self ):
		target    = np.linspace(0.01, 0.30, self.n_portfolios)
		eff_front = EfficientFrontier(self.mean_ret, self.cov_mat, self.rfr, target)
		x = np.array([p['fun'] for p in eff_front])
		df = pd.DataFrame({'Volatility':x, 'Returns':target})
		# Create max_sharpe_port
		max_sh = MaxSharpeRatio(self.mean_ret, self.cov_mat, self.rfr)['x']
		x = dict(zip(self.stocks,max_sh))
		port_max = portfolio(max_sh,self.mean_ret, self.cov_mat, self.rfr)
		y = {'Returns' : port_max[0], 'Sharpe Ratio' : port_max[2], 'Volatility' : port_max[1]}
		z = {**x, **y}
		max_sharpe_port = pd.Series(z)
		# Create min_vol_port
		min_vo = MinVolatility(self.mean_ret, self.cov_mat, self.rfr)['x']
		x_vo = dict(zip(self.stocks,min_vo))
		port_min = portfolio(min_vo,self.mean_ret, self.cov_mat, self.rfr)
		y_vo = {'Returns' : port_min[0], 'Sharpe Ratio' : port_min[2], 'Volatility' : port_min[1]}
		z_vo = {**x_vo, **y_vo}
		min_vol_port = pd.Series(z_vo)
		return df, max_sharpe_port, min_vol_port

	def plot( self ):
		df = self.theory[0]
		df = df.loc[df['Volatility'] < np.round(df['Volatility'].iloc[-1],7)]
		max_sh = self.theory[1]
		min_vol= self.theory[2]
		# Scatter plot colored by Sharpe Ratio
		plt.style.use('seaborn-dark')
		fig, ax = plt.subplots(figsize=(10,7))
		df.plot(ax= ax, kind='scatter',x='Volatility', y='Returns',edgecolors='black', grid=True, label = 'Theory')
		# Maximum Sharpe Ratio
		ax.scatter(x=max_sh['Volatility'],y=max_sh['Returns'],marker='o',c='r',s=100,label='Maximum Sharpe Ratio')
		# Minimum variance
		ax.scatter(x=min_vol['Volatility'],y=min_vol['Returns'],marker='o',c='b',s=100,label='Minimum Volatility')
		plt.legend()
		ax.set_xlabel('Volatility (Std. Deviation)', fontsize=15)
		ax.set_ylabel('Expected Returns', fontsize=15)
		ax.set_title('Efficient Frontier', fontsize=22)
		plt.show()

	def print( self ):
		max_sh = self.theory[1]
		min_vol= self.theory[2]
		print('Maximum Sharpe Ratio:\n{}'.format(
					max_sh[max_sh.index.tolist()[0:len(self.stocks)]].to_frame(name='Weights').T))
		print('{}'.format(max_sh[max_sh.index.tolist()[len(self.stocks):]].to_frame(name='Results').T))
		print('\nMinimum Volatility:\n{}'.format(
					min_vol[min_vol.index.tolist()[0:len(self.stocks)]].to_frame(name='Weights').T))
		print('{}'.format(min_vol[min_vol.index.tolist()[len(self.stocks):]].to_frame(name='Results').T))
#######################################
# END: Theory 						  #
#######################################

#-------------------------------------#
# Plot All		       				  #
#-------------------------------------#
def plot_all( simulation, theory ):
	# Scatter plot colored by Sharpe Ratio
	plt.style.use('seaborn-dark')
	fig, ax = plt.subplots(figsize=(10,7))
	# Simulation
	df = simulation.results
	max_sh = simulation.max_sharpe_portfolio
	min_vol= simulation.min_volatility_portfolio
	df.plot(ax= ax, kind='scatter',x='Volatility', y='Returns', c='Sharpe Ratio', cmap='RdYlGn', edgecolors='black', grid=True, label = 'MC Simulation',alpha=0.5)
	# Maximum Sharpe Ratio
	ax.scatter(x=max_sh['Volatility'],y=max_sh['Returns'],marker='D',c='r',s=100,label='Maximum Sharpe Ratio (MC)')
	# Minimum variance
	ax.scatter(x=min_vol['Volatility'],y=min_vol['Returns'],marker='D',c='b',s=100,label='Minimum Volatility (MC)')
	# Theory
	df = theory.results
	df = df.loc[df['Volatility'] < np.round(df['Volatility'].iloc[-1],7)]
	max_sh = theory.max_sharpe_portfolio
	min_vol= theory.min_volatility_portfolio
	df.plot(ax= ax, kind='scatter',x='Volatility', y='Returns',edgecolors='black', label = 'Theory', grid=True)
	# Maximum Sharpe Ratio
	ax.scatter(x=max_sh['Volatility'],y=max_sh['Returns'],marker='o',c='r',s=100,label='Maximum Sharpe Ratio (theory)',alpha=0.5)
	# Minimum variance
	ax.scatter(x=min_vol['Volatility'],y=min_vol['Returns'],marker='o',c='b',s=100,label='Minimum Volatility (theory)',alpha=0.5)
	plt.legend(loc=7)
	ax.set_xlabel('Volatility (Std. Deviation)', fontsize=15)
	ax.set_ylabel('Expected Returns', fontsize=15)
	ax.set_title('Efficient Frontier', fontsize=22)
	plt.show()
