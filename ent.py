#!/usr/bin/env python3
# file: ent.py
# vim:fileencoding=utf-8:fdm=marker:ft=python
#
# Modified on May 2019 by Corentin Rafflin <corentin.rafflin@eurecom.fr>.
# Copyright © 2018 R.F. Smith <rsmith@xs4all.nl>.
# SPDX-License-Identifier: MIT
"""
Updated version of the ‘ent’ python implementation by R.F. Smith which was a
partial implementation of the ‘ent’ program by John "Random" Walker.

See http://www.fourmilab.ch/random/ for the original.
See https://github.com/rsmith-nl/ent for the partial implementation by R.F. Smith
"""
from __future__ import division, print_function
import argparse
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler 

__version__ = '1.0'
PI = 3.14159265358979323846
filename = "scaler_lb_clf.sav"

bytescount = [str(i) for i in range(0,256)]
tests = ['File_type','File_bytes','Entropy','Chi_square','Mean','Monte_Carlo_Pi','Serial_Correlation'] 

def main(argv):
    """
    Calculate and print figures about the randomness of the input files.

    Arguments:
        argv: Program options.
    """

    #Passing arguments
    opts = argparse.ArgumentParser(prog='ent', description=__doc__)
    opts.add_argument('-c', action='store_true', help="print byte occurrence counts")
    opts.add_argument('-t', action='store_true', help="terse output in CSV format")
    opts.add_argument('-n', action='store_true', help="remove header when printing in CSV format")
    opts.add_argument('-f', action='store_true', help="plot the histogram of the byte distribution")
    opts.add_argument('-p', action='store_true', help="predict file type")
    opts.add_argument('-v', '--version', action='version', version=__version__)
    opts.add_argument("files", metavar='file', nargs='*', help="one or more files to process")
    args = opts.parse_args(argv)

    #Checking the input format
    if not(args.files):
	    sys.stderr.write("Input error, see help : %s --help \n"%sys.argv[0])
	    sys.exit(1)

    for fname in args.files:
        extension = os.path.splitext(fname)[1][1:]
        if len(extension)==0:
            extension="unknown"
        data, cnts = readdata(fname)
        e = entropy(cnts)
        c = pearsonchisquare(cnts)
        p = pochisq(c)
        d = math.fabs(p * 100 - 50)
        m = monte_carlo(data)
        try:
            scc = correlation(data)
            es = f"{scc:.6f}"
        except ValueError:
            es = 'undefined'
        if args.t: 
            terseout(extension, data, e, c, p, d, es, m, cnts, args.c, not(args.n))
       	else:
       		textout(data, e, c, p, d, es, m, cnts, args.c)
        if args.p:
            predictType(extension, data, e, c, p, d, es, m, cnts)
        if args.f:
            plotHist(cnts)
        
def plotHist(cnts):
    fig, ax = plt.subplots(1,1, figsize=(18,12))
    ax.bar([i for i in range(len(cnts))], cnts, width=1.0, edgecolor='black')
    ax.set_ylabel('Count', size=15)
    ax.set_xlabel('Byte', size=15)
    ax.set_title('Byte distribution', size=30)
    plt.show()

def toFraction(data, cnts):
    n = len(data)
    bytesFraction = []
    for byte in range(256):
       bytesFraction.append(round(cnts[byte]/n,6))
    return bytesFraction

def terseout(extension, data, e, chi2, p, d, scc, mc, cnts, withOcurrence, withHeader):
    """
    Print the results in terse CSV.

    Arguments:
        data: file contents
        e: Entropy of the data in bits per byte.
        chi2: Χ² value for the data.
        p: Probability of normal z value.
        d: Percent distance of p from centre.
        scc: Serial correlation coefficient.
        mc: Monte Carlo approximation of π.
       	cnts: numpy array containing the occurance of each byte.
    """
    n = len(data)
    m = data.mean()
    if withOcurrence:
        bytesFraction = toFraction(data, cnts)
        bytesString = ","
        bytesFractionString = ","
        for i in range(256):
            bytesString+=str(i) + ","
            bytesFractionString+= str(bytesFraction[i]) + ","
        if withHeader:
            print('File-type, File-bytes,Entropy,Chi-square,Mean,Monte-Carlo-Pi,Serial-Correlation' + bytesString[:-1])    
        print(f'{extension},{n},{e:.6f},{chi2:.6f},{m:.6f},{mc:.6f},{scc}' + bytesFractionString[:-1])
    else:
        if withHeader:
            print('File-type, File-bytes,Entropy,Chi-square,Mean,Monte-Carlo-Pi,Serial-Correlation')    
        print(f'{extension},{n},{e:.6f},{chi2:.6f},{m:.6f},{mc:.6f},{scc}')


def textout(data, e, chi2, p, d, scc, mc, cnts, withOcurrence):
    """
    Print the results in plain text.

    Arguments:
        data: file contents
        e: Entropy of the data in bits per byte.
        chi2: Χ² value for the data.
        p: Probability of normal z value.
        d: Percent distance of p from centre.
        scc: Serial correlation coefficient.
        mc: Monte Carlo approximation of π.
       	cnts: numpy array containing the occurance of each byte.
    """
    print(f'- Entropy is {e:.6f} bits per byte.')
    print('- Optimum compression would reduce the size')
    red = (100 * (8 - e)) / 8
    n = len(data)
    print(f'  of this {n} byte file by {red:.0f}%.')
    print(f'- χ² distribution for {n} samples is {chi2:.2f}, and randomly')
    pp = 100 * p
    print(f'  would exceed this value {pp:.2f}% of the times.')
    print("  According to the χ² test, this sequence", end=' ')
    if d > 49:
        print("is almost certainly not random")
    elif d > 45:
        print("is suspected of being not random.")
    elif d > 40:
        print("is close to random, but not perfect.")
    else:
        print("looks random.")
    m = data.mean()
    print(f'- Arithmetic mean value of data bytes is {m:.4f} (random = 127.5).')
    err = 100 * (math.fabs(PI - mc) / PI)
    print(f'- Monte Carlo value for π is {mc:.9f} (error {err:.2f}%).')
    print(f'- Serial correlation coefficient is {scc} (totally uncorrelated = 0.0).')  
    
    if withOcurrence:
        bytesFraction = toFraction(data, cnts)
        print('\nByte distribution :')
        for byte in range(256):
            print(f'{byte},' + str(int(cnts[byte])) + ',' + str(bytesFraction[byte]))


def readdata(name):
    """
    Read the data from a file and count byte occurences.

    Arguments:
        name: Path of the file to read

    Returns:
        data: numpy array containing the byte values.
        cnts: numpy array containing the occurance of each byte.
    """
    data = np.fromfile(name, np.ubyte)
    bincount = np.bincount(data)
    cnts = np.zeros(256)
    cnts[:bincount.shape[0]] = bincount
    return data, cnts


def entropy(counts):
    """
    Calculate the entropy of the data represented by the counts array.

    Arguments:
        counts: numpy array of counts for all byte values.

    Returns:
        Entropy in bits per byte.
    """
    counts = np.trim_zeros(np.sort(counts))
    sz = sum(counts)
    p = counts / sz
    ent = -sum(p * np.log(p) / math.log(256))
    return ent * 8


def pearsonchisquare(counts):
    """
    Calculate Pearson's χ² (chi square) test for an array of bytes.

    See [http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    #Discrete_uniform_distribution]

    Arguments:
        counts: Numpy array of counts.

    Returns:
        χ² value
    """
    np = sum(counts) / 256
    return sum((counts - np)**2 / np)


def correlation(d):
    """
    Calculate the serial correlation coefficient of the data.

    Arguments:
        d: numpy array of unsigned byte values.

    Returns:
        Serial correlation coeffiecient.
    """
    totalc = len(d)
    a = np.array(d, np.float64)
    b = np.roll(a, -1)
    scct1 = np.sum(a * b)
    scct2 = np.sum(a)**2
    scct3 = np.sum(a * a)
    scc = totalc * scct3 - scct2
    if scc == 0:
        raise ValueError
    scc = (totalc * scct1 - scct2) / scc
    return scc


def poz(z):
    """
    Calculate probability of normal z value.

    Adapted from http://en.wikipedia.org/wiki/Normal_distribution,
    integration by parts of cumulative distribution function.

    Arguments:
        z: normal z value

    Returns:
        Cumulative probability from -∞ to z.
    """
    if z > 3:
        return 1
    elif z < -3:
        return 0
    cnt = 10  # number of expansion elements to use.
    exp = np.array([2 * i + 1 for i in range(0, cnt + 1)])
    za = np.ones(cnt + 1) * z
    num = np.power(za, exp)
    denum = np.cumprod(exp)
    fact = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return 0.5 + fact * np.sum(num / denum)


def pochisq(x, df=255):
    """
    Compute probability of χ² test value.

    Adapted from: Hill, I. D. and Pike, M. C.  Algorithm 299 Collected
    Algorithms for the CACM 1967 p. 243 Updated for rounding errors based on
    remark in ACM TOMS June 1985, page 185.

    According to http://www.fourmilab.ch/random/:

      We interpret the percentage (return value*100) as the degree to which
      the sequence tested is suspected of being non-random. If the percentage
      is greater than 99% or less than 1%, the sequence is almost certainly
      not random. If the percentage is between 99% and 95% or between 1% and
      5%, the sequence is suspect. Percentages between 90% and 95% and 5% and
      10% indicate the sequence is “almost suspect”.

    Arguments:
        x: Obtained chi-square value.
        df: Degrees of freedom, defaults to 255 for random bytes.

    Returns:
        The degree to which the sequence tested is suspected of being
        non-random.
    """
    # Check arguments first
    if not isinstance(df, int):
        raise ValueError('df must be an integer')
    if x <= 0.0 or df < 1:
        return 1.0
    # Constants
    LOG_SQRT_PI = 0.5723649429247000870717135  # log(√π)
    I_SQRT_PI = 0.5641895835477562869480795  # 1/√π
    BIGX = 20.0
    a = 0.5 * x
    even = df % 2 == 0
    if df > 1:
        y = math.exp(-a)
    s = y if even else 2.0 * poz(-math.sqrt(x))
    if df > 2:
        x = 0.5 * (df - 1.0)
        z = 1.0 if even else 0.5
        if a > BIGX:
            e = 0 if even else LOG_SQRT_PI
            c = math.log(a)
            while z <= x:
                e = math.log(z) + e
                s += math.exp(c * z - a - e)
                z += 1.0
            return s
        else:
            e = 1.0 if even else I_SQRT_PI / math.sqrt(a)
            c = 0.0
            while z <= x:
                e = e * a / z
                c = c + e
                z += 1.0
            return c * y + s
    else:
        return s


def monte_carlo(d):
    """
    Calculate Monte Carlo value for π.

    Arguments:
        d: numpy array of unsigned byte values.

    Returns:
        Approximation of π
    """
    MONTEN = 6
    incirc = (256.0**(MONTEN // 2) - 1)**2
    d = np.array(d, copy=True, dtype=np.float64)
    d = d[:len(d) // MONTEN * MONTEN]
    values = np.sum(d.reshape((-1, MONTEN // 2)) * np.array([256**2, 256, 1]), axis=1)
    montex = values[0::2]
    montey = values[1::2]
    dist2 = montex * montex + montey * montey
    inmont = np.count_nonzero(dist2 <= incirc)
    montepi = 4 * inmont / len(montex)
    return montepi


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def predictType(extension, data, e, chi2, p, d, scc, mc, cnts):
 
    m = data.mean()
    bytesFraction = toFraction(data, cnts)
    
    my_input = np.array([e, chi2, m, mc, scc] + bytesFraction)
    input_type = extension

    file_path = resource_path(filename)
    modlist_loaded = pickle.load(open(file_path, 'rb'))

    scaler = modlist_loaded[0]
    lbencoder = modlist_loaded[1]
    clf = modlist_loaded[2]

    x_input = scaler.transform(my_input.reshape(1,-1))
    y_input_pred = clf.predict(x_input)
    pred = lbencoder.inverse_transform(y_input_pred)[0]
    probas = clf.predict_proba(x_input)
    print("\nThe classifier trained with {jpeg, mp3, zip, pdf, png} predicts that there is a " + str(round(probas.max()*100,3)) + "% chance that this is a " + pred + ' file.')
    if (probas.max()*100<80):
        print("As the probability is under 80%, we can consider that the classifier does not know this object.")
        pred = 'unknown'

    if (extension!='unknown'):
        if (extension==pred):
            isTrue = True
        else:
            isTrue = False
        print("The extension of the file shows that this is a " + str(extension) + " file, therefore the prediction may be " + str(isTrue))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main(sys.argv[1:])
